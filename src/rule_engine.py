#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""规则引擎: 用声明式规则 (rules/gates.json) 裁决所有阶段的判据。

设计要点:
  * 判据与代码分离 —— 阈值写在 JSON 里, 换芯片/换模型只改规则, 不改脚本;
  * 表达式受限求值 —— 只暴露 len/sum/min/max/abs/round/all/any 与几个聚合助手,
    不允许 import / 下标写入 / 函数定义;
  * 三级严重度 —— error 影响退出码 (可挂 CI), warn 只提示, info 仅记录;
  * 产物新鲜度 —— 规则可以声明"结果文件必须比依赖源码新", 防止拿旧结果当结论。

用法:
    python3 src/rule_engine.py                       # 全量裁决 -> results/RULES.md + rules_report.json
    python3 src/rule_engine.py --only S2             # 只看某个阶段
    python3 src/rule_engine.py --json                # 只打印 JSON 摘要
    python3 src/rule_engine.py --rules rules/xxx.json
退出码 = error 级失败数 + 过期产物数 (0 = 全绿)。
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_RULES = os.path.join(ROOT, "rules", "gates.json")


# ----------------------------------------------------------------- 上下文
class Dot:
    """让 JSON 能写成 s5.metrics_bc.logits_cosine_prefill 这种属性访问。"""

    __slots__ = ("_v",)

    def __init__(self, v):
        self._v = v

    def __getattr__(self, k):
        v = self._v
        if isinstance(v, dict) and k in v:
            return wrap(v[k])
        raise AttributeError("字段不存在: %s" % k)

    def __getitem__(self, k):
        return wrap(self._v[k])

    def __len__(self):
        return len(self._v)

    def __iter__(self):
        return iter(wrap(x) for x in self._v)

    def __repr__(self):
        return repr(self._v)


def wrap(v):
    return Dot(v) if isinstance(v, (dict, list)) else v


def raw(x):
    return x._v if isinstance(x, Dot) else x


# ----------------------------------------------------------------- 表达式助手
def count_true(lst, key="passed"):
    return sum(1 for x in raw(lst) if raw(x).get(key) is True)


def all_true(lst, key="passed"):
    items = raw(lst)
    return bool(items) and all(raw(x).get(key) is True for x in items)


def min_of(lst, key, default=None):
    vals = [raw(x).get(key) for x in raw(lst)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return min(vals) if vals else default


def max_of(lst, key, default=None):
    vals = [raw(x).get(key) for x in raw(lst)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return max(vals) if vals else default


def pick(lst, key, value):
    """从列表里挑出某个字段等于 value 的元素 (返回原 dict 列表)。"""
    return [raw(x) for x in raw(lst) if raw(x).get(key) == value]


HELPERS = {"len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
           "all": all, "any": any, "int": int, "float": float, "str": str, "sorted": sorted,
           "count_true": count_true, "all_true": all_true, "min_of": min_of, "max_of": max_of,
           "pick": pick, "True": True, "False": False, "None": None}

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
       "<": lambda a, b: a < b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def evaluate(expr, ctx):
    env = dict(HELPERS)
    env["os"] = os
    env["ROOT"] = ROOT
    env.update(ctx)
    return eval(expr, {"__builtins__": {}}, env)      # noqa: S307 - 受限求值, 规则文件由本仓库维护


# ----------------------------------------------------------------- 规则加载
ALIAS = {"s1_model_analysis": "s1", "s2_ops_alignment": "s2", "s2_gates": "s2g",
         "s3_quant": "s3", "s3_pack": "s3_pack", "s5_e2e": "s5", "s6_perf": "s6",
         "s7_regression": "s7", "s7_fault_injection": "s7fi", "s8_vision": "s8",
         "summary": "s9"}


def load_sources(names=None):
    """把 results/ 下的 JSON 读进上下文 (按文件名与短别名两种键注册)。

    names=None 时加载全部 —— 规则表达式里写 s5 还是 s5_e2e 都能命中。
    """
    src = {}
    if names is None:
        names = [f for f in sorted(os.listdir(RESULTS)) if f.endswith(".json")]
    for n in names:
        p = os.path.join(RESULTS, n)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                stem = n[:-5] if n.endswith(".json") else n
                data = wrap(json.load(f))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print("  [warn] %s 解析失败: %s" % (n, e), file=sys.stderr)
            continue
        src[stem] = data
        if stem in ALIAS:
            src[ALIAS[stem]] = data
    return src


def mtime(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return os.path.getmtime(p) if os.path.exists(p) else 0.0


def run(spec, only=None):
    rules = spec["rules"]
    fres = spec.get("freshness", [])
    ctx = load_sources()

    out = []
    for r in rules:
        if only and r.get("stage") != only:
            continue
        entry = {k: r.get(k) for k in ("id", "stage", "desc", "severity", "evidence")
                 if k in r}
        try:
            val = evaluate(r["expr"], ctx)
            entry["value"] = val
            if "ok_expr" in r:
                ok = bool(evaluate(r["ok_expr"], ctx))
                entry["limit"] = r.get("limit", r["ok_expr"])
            else:
                op, thr = r["op"], r["threshold"]
                if isinstance(thr, str) and thr.startswith("expr:"):
                    thr = evaluate(thr[5:], ctx)
                ok = OPS[op](val, thr)
                entry["limit"] = "%s %s" % (op, thr)
            entry["ok"] = bool(ok)
        except Exception as e:                                  # noqa: BLE001
            entry.update({"value": None, "ok": False, "limit": r.get("limit", ""),
                          "error": "%s: %s" % (type(e).__name__, e)})
        entry.setdefault("severity", "error")
        out.append(entry)

    fresh = []
    for f in fres:
        if only and f.get("stage") != only:
            continue
        rt = mtime(f["result"])
        stale = [s for s in f.get("sources", []) if mtime(os.path.join(ROOT, "src", s)) > rt + 1.0]
        fresh.append({"stage": f["stage"], "result": f["result"], "exists": rt > 0,
                      "stale_by": stale, "ok": rt > 0 and not stale})
    return out, fresh


def render(rules, fresh, spec):
    n_err = sum(1 for r in rules if not r["ok"] and r["severity"] == "error")
    n_warn = sum(1 for r in rules if not r["ok"] and r["severity"] == "warn")
    n_fresh = sum(1 for f in fresh if not f["ok"])
    lines = ["# 判据裁决总表 (规则引擎)",
             "",
             "- 规则文件: `%s` (v%s)" % (os.path.relpath(DEFAULT_RULES, ROOT), spec.get("version", "?")),
             "- 生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "- 结果: **%d/%d PASS**; error 级失败 %d, warn 级 %d, 过期/缺失产物 %d"
             % (len(rules) - n_err - n_warn, len(rules), n_err, n_warn, n_fresh),
             "",
             "| # | 判据 | 阶段 | 严重度 | 说明 | 实测 | 门槛 | 判定 |",
             "|---|------|------|--------|------|------|------|------|"]
    for i, r in enumerate(rules, 1):
        mark = "PASS" if r["ok"] else ("**FAIL**" if r["severity"] == "error" else "*WARN*")
        val = r["value"]
        if isinstance(val, float):
            val = ("%.6f" % val).rstrip("0").rstrip(".")
        lines.append("| %d | `%s` | %s | %s | %s | %s | %s | %s |"
                     % (i, r["id"], r.get("stage", ""), r["severity"], r.get("desc", ""),
                        val, r.get("limit", ""), mark))
    lines += ["", "## 产物新鲜度（结果是否由当前代码产生）", "",
              "| 阶段 | 产出 | 状态 | 备注 |", "|------|------|------|------|"]
    for f in fresh:
        if not f["exists"]:
            st, note = "**缺失**", "尚未运行"
        elif f["stale_by"]:
            st, note = "**过期**", "源码更新晚于产出: " + ", ".join(f["stale_by"])
        else:
            st, note = "OK", "产出晚于依赖源码"
        lines.append("| %s | `%s` | %s | %s |" % (f["stage"], f["result"], st, note))
    lines += ["", "> error 级失败与过期产物会让退出码非 0，可直接挂 CI/门禁；",
              "> warn 级只提示，不阻断。修改阈值请编辑 `rules/gates.json`，不要改脚本。"]
    return "\n".join(lines) + "\n", {"n_error": n_err, "n_warn": n_warn, "n_stale": n_fresh,
                                     "n_pass": len(rules) - n_err - n_warn, "n_total": len(rules)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--only", default=None, help="只看某个阶段 (如 S2/S5/S7/S8)")
    ap.add_argument("--json", action="store_true", help="只打印 JSON 摘要")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    with open(a.rules, encoding="utf-8") as f:
        spec = json.load(f)
    rules, fresh = run(spec, a.only)
    md, summary = render(rules, fresh, spec)

    if not a.no_write:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, "RULES.md"), "w", encoding="utf-8") as f:
            f.write(md)
        with open(os.path.join(RESULTS, "rules_report.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rules": rules, "freshness": fresh,
                       "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=2, default=str)

    if a.json:
        print(json.dumps({"summary": summary,
                          "failed": [r["id"] for r in rules if not r["ok"]],
                          "stale": [f["stage"] for f in fresh if not f["ok"]]},
                         ensure_ascii=False, indent=2))
    else:
        for r in rules:
            tag = "PASS " if r["ok"] else ("FAIL " if r["severity"] == "error" else "WARN ")
            print("%s %-26s %-28s [%s]" % (tag, r["id"], r["value"], r.get("limit", "")))
            if not r["ok"] and r.get("error"):
                print("      └─ %s" % r["error"])
        for f in fresh:
            if not f["ok"]:
                print(("MISS  " if not f["exists"] else "STALE ") + f["stage"])
        print("\n%d/%d PASS | error %d | warn %d | 过期产物 %d -> results/RULES.md"
              % (summary["n_pass"], summary["n_total"], summary["n_error"],
                 summary["n_warn"], summary["n_stale"]))
    sys.exit(summary["n_error"] + summary["n_stale"])


if __name__ == "__main__":
    main()
