#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model2npu MCP server (stdio, 无第三方依赖)。

MCP (Model Context Protocol) 的最小实现: initialize / tools/list / tools/call / ping,
消息为按行分隔的 JSON-RPC 2.0。工具把仓库里的阶段脚本包成智能体可调用的能力。

用法:
    python3 mcp/server.py --stdio            # MCP 客户端用 (stdio transport)
    python3 mcp/server.py --list-tools       # 人肉看工具清单
    python3 mcp/server.py --call verify_all  # 命令行直接调一个工具
    python3 mcp/server.py --call run_stage --args '{"stage":"s2_align"}'
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")
KNOWLEDGE = os.path.join(ROOT, "knowledge")
RULES = os.path.join(ROOT, "rules", "gates.json")

# stage -> (脚本, 说明, 产物, 默认参数)
STAGES = {
    "s1_analyze_model": ("s1_analyze_model.py", "模型分析/部署规划/roofline",
                         "results/s1_model_analysis.json", []),
    "s2_align": ("s2_align.py", "算子数值对齐 (Python↔C++ native↔torch)",
                 "results/s2_ops_alignment.json", []),
    "s3_quant": ("s3_quant.py", "量化 RTN/AWQ/GPTQ + 混合精度",
                 "results/s3_quant.json", ["--gptq", "sample"]),
    "s3_pack": ("s3_pack.py", "int4 权重打包 + 设备镜像导出",
                "results/s3_pack.json", ["--variant", "w4a16_rtn_mix8"]),
    "s4_gen_program": ("s4_gen_program.py", "模型 → ISA 微码",
                       "build/npu_program_w4a16_rtn_mix8_kvfp16.mtx", []),
    "s5_e2e": ("s5_e2e.py", "端到端三端对齐 (prefill + 连续 decode)",
               "results/s5_e2e.json", ["--variant", "w4a16_rtn_mix8", "--decode", "8"]),
    "s6_perf": ("s6_perf.py", "性能与瓶颈归因 (TTFT/TPOT/带宽/存储)",
                "results/s6_perf.json", ["--variant", "w4a16_rtn_mix8"]),
    "s7_regress": ("s7_regress.py", "回归门禁 + 故障注入逐层定位",
                   "results/s7_regression.json", ["--allow-fail"]),
    "s8_vision": ("s8_vision.py", "ViT 视觉编码器 + 视觉 token 多模态链路",
                  "results/s8_vision.json", []),
    "s9_report": ("s9_report.py", "汇总报告 (一页纸结论 + 覆盖矩阵)",
                  "results/SUMMARY.md", []),
}


def run(cmd, timeout=3600):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"))
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip(), time.time() - t0
    except subprocess.TimeoutExpired:
        return 124, "超时 (> %ds)" % timeout, time.time() - t0


def python_exe():
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    for c in ("python3", "/usr/local/python3.11.15/bin/python"):
        if subprocess.run(["bash", "-lc", "command -v %s" % c], capture_output=True).returncode == 0:
            try:
                if subprocess.run([c, "-c", "import numpy, torch, tokenizers, PIL"],
                                  capture_output=True, env=env).returncode == 0:
                    return c
            except FileNotFoundError:
                pass
    return None


def tool_env_check(_):
    py = python_exe()
    lines = ["python(带 numpy+torch): %s" % (py or "缺失"),
             "g++: %s" % ("可用" if subprocess.run(["bash", "-lc", "command -v g++"],
                                                 capture_output=True).returncode == 0 else "缺失"),
             "ISS: %s" % ("已编译" if os.path.exists(os.path.join(ROOT, "build", "npu_iss"))
                          else "未编译 (先跑 install.sh)"),
             "模型: " + ", ".join("%s %s" % (d, "存在" if os.path.isdir(os.path.join(ROOT, "models", d))
                                            else "缺失")
                                 for d in ("qwen2.5-0.5b-instruct", "deit-tiny-patch16-224")),
             "CANN/atc: %s" % ("可用" if subprocess.run(["bash", "-lc", "command -v atc"],
                                                       capture_output=True).returncode == 0 else "未安装"),
             "npu-smi: %s" % ("可用" if subprocess.run(["bash", "-lc", "command -v npu-smi"],
                                                      capture_output=True).returncode == 0 else "未安装")]
    return "\n".join(lines)


def tool_list_stages(_):
    return "\n".join("%-16s %-42s -> %s" % (k, v[1], v[2]) for k, v in STAGES.items())


def tool_run_stage(a):
    name = a.get("stage")
    if name not in STAGES:
        return "未知阶段 %r; 可选: %s" % (name, ", ".join(STAGES))
    script, desc, artifact, default_args = STAGES[name]
    extra = a.get("args") or default_args
    py = python_exe() or "python3"
    rc, out, dt = run([py, os.path.join(SRC, script)] + list(extra),
                      timeout=a.get("timeout_s", 3600))
    art = os.path.join(ROOT, artifact)
    return ("阶段 %s (%s) 退出码 %d, 用时 %.1fs\n产物: %s (%s)\n---- 输出末尾 ----\n%s"
            % (name, desc, rc, dt, artifact,
               "存在" if os.path.exists(art) else "缺失", "\n".join(out.splitlines()[-40:])))


def tool_verify_all(_):
    py = python_exe() or "python3"
    rc, out, dt = run([py, os.path.join(SRC, "verify_all.py")], timeout=1800)
    gates = {}
    p = os.path.join(RESULTS, "ALL_GATES.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            gates = json.load(f)
    bad = [g["id"] for g in gates.get("gates", []) if not g["ok"]]
    stale = [s["stage"] for s in gates.get("stages", []) if not s["ok"]]
    return ("verify_all 退出码 %d, %.1fs\nFAIL 判据: %s\n过期/缺失阶段: %s\n----\n%s"
            % (rc, dt, bad or "无", stale or "无", "\n".join(out.splitlines()[-30:])))


def tool_read_gate_table(_):
    p = os.path.join(RESULTS, "ALL_GATES.json")
    if not os.path.exists(p):
        return "还没有 results/ALL_GATES.json, 先调用 verify_all"
    with open(p, encoding="utf-8") as f:
        return json.dumps(json.load(f), ensure_ascii=False, indent=2)


def tool_read_report(a):
    name = a.get("file") or "SUMMARY.md"
    p = os.path.join(RESULTS, os.path.basename(name))
    if not os.path.exists(p):
        return "没有 %s" % p
    text = open(p, encoding="utf-8").read()
    limit = int(a.get("max_chars") or 8000)
    return text if len(text) <= limit else text[:limit] + "\n... (截断)"


def tool_ascend_build_om(a):
    onnx = a.get("onnx")
    if not onnx:
        return "需要参数 onnx=<模型路径>"
    out = a.get("out_prefix") or "/tmp/model2npu_smoke"
    cmd = ["bash", os.path.join(SRC, "ascend_build_om.sh"), onnx, out]
    if a.get("input_shape"):
        cmd.append(a["input_shape"])
    rc, o, dt = run(cmd, timeout=1800)
    return "atc 退出码 %d, %.1fs\n%s" % (rc, dt, "\n".join(o.splitlines()[-20:]))


def tool_ascend_run_om(a):
    om = a.get("om")
    if not om:
        return "需要参数 om=<模型路径>"
    py = python_exe() or "python3"
    rc, o, dt = run([py, os.path.join(SRC, "ascend_acl_run.py"), om, str(a.get("device_id", 0))],
                    timeout=600)
    return "acl 执行退出码 %d, %.1fs\n%s" % (rc, dt, o)


# ---------------------------------------------------------------- 规则引擎
def load_rules():
    with open(RULES, encoding="utf-8") as f:
        return json.load(f)


def tool_list_rules(a):
    spec = load_rules()
    stage = a.get("stage")
    rows = ["规则文件: rules/gates.json (v%s), 共 %d 条 + %d 条新鲜度规则"
            % (spec.get("version"), len(spec["rules"]), len(spec.get("freshness", [])))]
    for r in spec["rules"]:
        if stage and r.get("stage") != stage:
            continue
        rows.append("[%s] %-26s %-6s %s | 门槛: %s"
                    % (r.get("stage", "-"), r["id"], r.get("severity", "error"),
                       r.get("desc", ""), r.get("limit", "%s %s" % (r.get("op", ""), r.get("threshold", "")))))
    return "\n".join(rows)


def tool_run_rules(a):
    py = python_exe() or "python3"
    cmd = [py, os.path.join(SRC, "rule_engine.py")]
    if a.get("stage"):
        cmd += ["--only", a["stage"]]
    if a.get("json"):
        cmd.append("--json")
    rc, out, dt = run(cmd, timeout=a.get("timeout_s", 600))
    return "规则引擎退出码 %d (error 失败数 + 过期产物数), %.1fs\n%s" % (rc, dt, out.strip())


def tool_read_gate_table(_):
    for name in ("rules_report.json", "ALL_GATES.json"):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    return "还没有判据报告, 先调用 run_rules / verify_all"


# ---------------------------------------------------------------- 知识库
def kb_entries():
    with open(os.path.join(KNOWLEDGE, "index.json"), encoding="utf-8") as f:
        return json.load(f)["entries"]


def tool_kb_list(_):
    rows = []
    for e in kb_entries():
        rows.append("%-22s %s\n%-22s 阶段 %s | tags: %s | 文件 %s"
                    % (e["id"], e["title"], "", ",".join(e.get("stage", [])),
                       ", ".join(e.get("tags", [])), e["file"]))
    return "\n\n".join(rows)


def tool_kb_search(a):
    q = (a.get("query") or "").strip()
    if not q:
        return "需要参数 query"
    terms = [t for t in q.replace(",", " ").split() if t]
    scored = []
    for e in kb_entries():
        blob = " ".join([e["title"], " ".join(e.get("tags", [])), " ".join(e.get("symptoms", [])),
                         " ".join(e.get("stage", [])), e.get("verdict", "")])
        score = sum(blob.count(t) for t in terms)
        if score:
            scored.append((score, e))
    if not scored:
        return "没有命中; 可用 kb_list 看全部条目"
    scored.sort(key=lambda x: -x[0])
    rows = []
    for score, e in scored[:5]:
        rows.append("score=%d  %s\n  标题: %s\n  适用阶段: %s\n  症状: %s\n  结论: %s\n  全文: kb_get id=%s"
                    % (score, e["id"], e["title"], ",".join(e.get("stage", [])),
                       " / ".join(e.get("symptoms", [])), e.get("verdict", ""), e["id"]))
    return "\n\n".join(rows)


def tool_kb_get(a):
    eid = a.get("id")
    for e in kb_entries():
        if e["id"] == eid:
            p = os.path.join(ROOT, e["file"])
            if not os.path.exists(p):
                return "条目 %s 的文件缺失: %s" % (eid, e["file"])
            text = open(p, encoding="utf-8").read()
            limit = int(a.get("max_chars") or 8000)
            return text if len(text) <= limit else text[:limit] + "\n... (截断)"
    return "没有 id=%s 的条目; 用 kb_list 查看" % eid


TOOLS = {
    "env_check": (tool_env_check, "检查 Python/g++/ISS/模型/CANN 是否就绪", {}),
    "list_stages": (tool_list_stages, "列出所有可跑阶段与产物路径", {}),
    "run_stage": (tool_run_stage, "跑某个阶段 (s1_analyze_model/s2_align/s3_quant/...)", {
        "type": "object", "properties": {
            "stage": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}},
            "timeout_s": {"type": "integer"}}, "required": ["stage"]}),
    "verify_all": (tool_verify_all, "跑 20 条判据总闸口, 返回 FAIL 与过期产物清单", {}),
    "read_gate_table": (tool_read_gate_table, "读 results/ALL_GATES.json", {}),
    "read_report": (tool_read_report, "读 results/ 下的报告 (默认 SUMMARY.md)", {
        "type": "object", "properties": {"file": {"type": "string"},
                                         "max_chars": {"type": "integer"}}}),
    "list_rules": (tool_list_rules, "列出声明式判据规则 (rules/gates.json)", {
        "type": "object", "properties": {"stage": {"type": "string"}}}),
    "run_rules": (tool_run_rules, "运行规则引擎裁决所有判据, 返回失败与过期产物清单", {
        "type": "object", "properties": {"stage": {"type": "string"}, "json": {"type": "boolean"},
                                         "timeout_s": {"type": "integer"}}}),
    "kb_list": (tool_kb_list, "列出知识库条目 (knowledge/index.json)", {}),
    "kb_search": (tool_kb_search, "按关键词/症状检索知识库 (定位经验)", {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    "kb_get": (tool_kb_get, "读取知识库条目全文", {
        "type": "object", "properties": {"id": {"type": "string"},
                                         "max_chars": {"type": "integer"}}, "required": ["id"]}),
    "ascend_build_om": (tool_ascend_build_om, "真机: ONNX → ATC → .om", {
        "type": "object", "properties": {"onnx": {"type": "string"}, "out_prefix": {"type": "string"},
                                         "input_shape": {"type": "string"}}, "required": ["onnx"]}),
    "ascend_run_om": (tool_ascend_run_om, "真机: pyACL 在 Ascend 上执行 .om", {
        "type": "object", "properties": {"om": {"type": "string"}, "device_id": {"type": "integer"}},
        "required": ["om"]}),
}


def list_resources():
    res = []
    for e in kb_entries():
        res.append({"uri": "kb://%s" % e["id"], "name": e["title"],
                    "description": "知识库: " + e.get("verdict", ""), "mimeType": "text/markdown"})
    for f in sorted(os.listdir(RESULTS)):
        if f.endswith((".md", ".json")) and not f.startswith(("npu_profile_", "rules_report")):
            res.append({"uri": "report://%s" % f, "name": f,
                        "description": "阶段报告 / 原始判据数据", "mimeType": "text/plain"})
    return res


def read_resource(uri):
    if uri.startswith("kb://"):
        return tool_kb_get({"id": uri[5:], "max_chars": 60000})
    if uri.startswith("report://"):
        p = os.path.join(RESULTS, os.path.basename(uri[9:]))
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()[:60000]
        return "不存在: " + uri
    return "不支持的 uri: " + uri


def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        res = {"protocolVersion": "2024-11-05",
               "capabilities": {"tools": {}, "resources": {}},
               "serverInfo": {"name": "model2npu", "version": "0.1.0"}}
    elif method in ("notifications/initialized", "notifications/cancelled"):
        return None
    elif method == "ping":
        res = {}
    elif method == "resources/list":
        res = {"resources": list_resources()}
    elif method == "resources/read":
        uri = (msg.get("params") or {}).get("uri", "")
        text = read_resource(uri)
        res = {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": text}]}
    elif method == "tools/list":
        res = {"tools": [{"name": n, "description": d, "inputSchema": s or {"type": "object"}}
                         for n, (f, d, s) in TOOLS.items()]}
    elif method == "tools/call":
        name = (msg.get("params") or {}).get("name")
        args = (msg.get("params") or {}).get("arguments") or {}
        fn = TOOLS.get(name)
        if fn is None:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": "未知工具 %s" % name}}
        try:
            text = fn[0](args)
            res = {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as e:                                  # noqa: BLE001
            res = {"content": [{"type": "text", "text": "工具执行异常: %r" % e}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": "不支持的方法 %s" % method}}
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdio", action="store_true", help="MCP stdio 模式")
    ap.add_argument("--list-tools", action="store_true")
    ap.add_argument("--call", help="直接调用某个工具")
    ap.add_argument("--args", default="{}", help="工具参数 (JSON)")
    a = ap.parse_args()

    if a.list_tools:
        for n, (f, d, s) in TOOLS.items():
            print("%-18s %s" % (n, d))
        return
    if a.call:
        if a.call not in TOOLS:
            print("未知工具; 可选: %s" % ", ".join(TOOLS))
            sys.exit(2)
        print(TOOLS[a.call][0](json.loads(a.args)))
        return

    for line in sys.stdin:                       # stdio MCP: 一行一个 JSON-RPC
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except json.JSONDecodeError as e:
            reply = {"jsonrpc": "2.0", "id": None,
                     "error": {"code": -32700, "message": "解析失败: %s" % e}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
