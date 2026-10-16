#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S7 回归判据与问题定位。

对应 能力项 5 的 "可重复的精度与性能回归流程" + 职责 2/5 的 "跨模型、算子、系统链路定位问题":
  1. 门禁回归: 算子对齐 / 权重格式 / 端到端精度 / 性能, 逐条 PASS-FAIL, 不达标返回非 0;
  2. 故障注入定位: 故意注入 3 类典型错误 (RoPE theta、attention scale、量化 zero-point),
     用「逐层误差曲线」把问题定位到层/算子 —— 这是现场最需要的能力;
  3. 可重复性: 同一条微码跑两次, 计数与 logits 必须逐 bit 一致。

产出: results/s7_regression.json / .md, results/s7_fault_injection.{json,md}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_ops_ref as R
import mtx_qwen as Q
import s3_quant as S3


# ------------------------------------------------------------------ 门禁
GATES = {
    "ops.op_alignment":       {"desc": "结构算子与参考实现对齐 (12 用例)", "op": "all_pass"},
    "ops.quant_alignment":    {"desc": "量化算子 bit 级一致 (13 用例)", "op": "all_pass"},
    "pack.bit_exact":         {"desc": "硬件权重格式解包逐 bit 一致", "op": ">=", "thr": 100.0},
    "e2e.logits_cosine_bc":   {"desc": "ISS 输出 vs 量化模拟 (prefill logits cosine)", "op": ">=", "thr": 0.999},
    "e2e.decode_cosine_min":  {"desc": "ISS 输出 vs 量化模拟 (各 decode 步 cosine 最小值)", "op": ">=", "thr": 0.999},
    "e2e.gen_match_bc":       {"desc": "ISS 与量化模拟的生成 token 一致率", "op": ">=", "thr": 0.9},
    "e2e.logits_cosine_ab":   {"desc": "量化 vs fp32 参考 (logits cosine)", "op": ">=", "thr": 0.90},
    "e2e.top1_ab":            {"desc": "量化 vs fp32 参考 (top-1 一致率)", "op": ">=", "thr": 0.60},
    "perf.tpot_ms":           {"desc": "TPOT (ms, 越小越好)", "op": "<=", "thr": 15.0},
    "perf.ttft_ms":           {"desc": "TTFT (ms, 越小越好)", "op": "<=", "thr": 50.0},
    "repeat.deterministic":   {"desc": "同一微码两次运行逐 bit 一致", "op": "==", "thr": 1.0},
}


def check(val, spec):
    op = spec["op"]
    if op == "all_pass":
        return bool(val)
    if op == ">=":
        return val >= spec["thr"]
    if op == "<=":
        return val <= spec["thr"]
    if op == "==":
        return val == spec["thr"]
    raise ValueError(op)


def fmt(val):
    return f"{val:.4f}" if isinstance(val, float) else str(val)


def run_gates(variant, results):
    rows, data = [], {}
    for key, spec in GATES.items():
        val = results.get(key)
        if val is None:
            rows.append([key, spec["desc"], "-", "SKIP", "无数据"])
            continue
        ok = check(val, spec)
        data[key] = val
        rows.append([key, spec["desc"], fmt(val),
                     "PASS" if ok else "FAIL",
                     f"{spec['op']} {spec.get('thr', '')}".strip()])
    return rows, data


def collect(variant):
    res = {}
    p2 = os.path.join(C.RESULTS, "s2_gates.json")
    if os.path.exists(p2):
        g = C.jload(p2)
        res["ops.op_alignment"] = 1.0 if g.get("op_alignment_pass") else 0.0
        res["ops.quant_alignment"] = 1.0 if g.get("quant_alignment_pass") else 0.0
    p6 = os.path.join(C.RESULTS, "s6_perf.json")
    if os.path.exists(p6):
        h = C.jload(p6).get("headline", {})
        res["perf.tpot_ms"] = h.get("tpot_ms")
        res["perf.ttft_ms"] = h.get("ttft_ms")
    return res


# ------------------------------------------------------- 故障注入与定位
def fault_experiments(cfg, st, pk_all, ids, rep):
    """三类典型故障 + 逐层误差曲线, 证明工具链能定位到层/算子。"""
    st_q = S3.variant_state(pk_all, st)
    base = Q.Qwen2Ref(cfg, st_q, act_dtype="fp16")

    def layer_curve(model, ref_caps):
        cap = {"layers": [None] * cfg.layers}
        model.forward(input_ids=ids, capture=cap)
        return [R.cos_sim(ref_caps["layers"][i]["out"].numpy(),
                          cap["layers"][i]["out"].numpy()) for i in range(cfg.layers)]

    ref_cap = {"layers": [None] * cfg.layers}
    base.forward(input_ids=ids, capture=ref_cap)

    experiments = []

    # 用 linear hook 记"当前跑到第几层", 注入只作用于指定层
    class FaultModel(Q.Qwen2Ref):
        """可注入三类故障的参考模型; 注入点只作用于指定层。"""

        def __init__(self, *a, fault=None, layer=8, **kw):
            super().__init__(*a, **kw)
            self.fault, self.layer, self.cur = fault, layer, -1

        def rope_tables(self, head_dim, positions, theta):
            return super().rope_tables(head_dim, positions, theta)

        def rope(self, x, cos, sin):
            if self.fault == "rope_theta" and self.cur == self.layer:
                # 注入: 该层的 RoPE 频率表用错 theta (1e6 -> 1e4)
                cos2, sin2 = self.rope_tables(x.shape[-1], self._positions, 1e4)
                return super().rope(x, cos2, sin2)
            return super().rope(x, cos, sin)

        def attention(self, q, k, v, positions, n_hist=0):
            if self.fault == "attn_scale" and self.cur == self.layer:
                old, self.cfg.scale = self.cfg.scale, 1.0
                try:
                    return super().attention(q, k, v, positions, n_hist)
                finally:
                    self.cfg.scale = old
            return super().attention(q, k, v, positions, n_hist)

        def linear(self, x, w, b=None, name=None):
            if self.fault == "quant_zp" and self.cur == self.layer and \
                    name and name.endswith("q_proj.weight"):
                # 注入: zero-point 读错 1 → 每个 (行, group) 的去量化权重整体偏一个量化步长
                N, K = w.shape
                g = 128
                G = (K + g - 1) // g
                pad = G * g - K
                wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
                wr = wp.reshape(N, G, g)
                step = (wr.amax(-1) - wr.amin(-1)) / ((2 ** 4) - 1)      # [N,G]
                shift = step[:, :, None].expand(N, G, g).reshape(N, G * g)
                w = w + shift[:, :K]
            return super().linear(x, w, b, name)

        def forward(self, *a, **kw):
            """借 linear 的名字推进层号 (每个 linear 名里带 layer 索引)。"""
            self.cur = -1
            orig = super().forward
            old_hook = self.act_hook

            def combined(name, x):
                if name and ".layers." in name:
                    self.cur = int(name.split(".layers.")[1].split(".")[0])
                if old_hook is not None:
                    old_hook(name, x)

            self.act_hook = combined
            try:
                return orig(*a, **kw)
            finally:
                self.act_hook = old_hook

    for fault, layer, desc in [
        ("rope_theta", 8, "RoPE theta 取错 (1e6 → 1e4), 只注入第 8 层"),
        ("attn_scale", 8, "attention scale 丢了 1/√d (scale=1.0), 只注入第 8 层"),
        ("quant_zp", 8, "量化 zero-point 差 1 (权重整体偏移一个量化步长), 只注入第 8 层 q_proj"),
    ]:
        m = FaultModel(cfg, st_q, act_dtype="fp16", fault=fault, layer=layer)
        curve = layer_curve(m, ref_cap)
        # 定位判据 1: 误差(1-cos)在哪一层发生最大跳变 —— 故障注入点
        err = [1.0 - c for c in curve]
        delta = [err[0]] + [err[i] - err[i - 1] for i in range(1, len(err))]
        # 判据 1 (定位用): 第一次出现偏离的层 —— 两侧除注入点外逐 bit 相同, 所以
        # 任何 >1e-9 的误差增量都只能来自注入点。
        first = next((i for i in range(1, len(err)) if delta[i] > 1e-9), None)
        jump = int(np.argmax(delta))          # 判据 2: 误差增量最大的层 (受放大影响, 通常靠后)
        experiments.append({"fault": fault, "desc": desc, "injected_layer": layer,
                            "localized_layer": first if first is not None else jump,
                            # 判定: 定位到注入层或其紧邻后一层 (注入误差在层内可能先低于判据阈值)
                            "localized_correct": (first is not None and abs(first - layer) <= 1),
                            "first_divergence_layer": first,
                            "max_delta_layer": jump,
                            "curve": [round(c, 8) for c in curve],
                            "delta": [round(x, 10) for x in delta],
                            "min_cos": float(min(curve))})
    rep.h("故障注入 → 逐层误差曲线 → 定位", 2)
    rep.p("做法: 用同一份测试算子链路, 每次只在一个层注入一类典型错误, 看**逐层 hidden cosine** 在哪里第一次掉下去。"
          "现场排查时, 这张曲线的第一处跳变就是问题算子所在的层。")
    rep.table(["注入故障", "注入层", "定位层(首次偏离)", "误差增量最大层", "定位正确", "最低 cosine"],
              [[e["desc"], e["injected_layer"], e["localized_layer"],
                e["max_delta_layer"], "是" if e["localized_correct"] else "否",
                f"{e['min_cos']:.6f}"] for e in experiments])
    for e in experiments:
        lo = max(0, e["injected_layer"] - 3)
        hi = min(len(e["curve"]), e["injected_layer"] + 4)
        rep.p(f"**{e['fault']}**: 误差 1-cos 在第 {lo}~{hi - 1} 层 = " +
              ", ".join(f"L{i}={1 - e['curve'][i]:.2e}" for i in range(lo, hi)) +
              f"; 定位到 L{e['localized_layer']} (注入点 L{e['injected_layer']}, "
              f"最低 cosine 出现在 L{int(np.argmin(e['curve']))})。")
    return experiments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a16_rtn_mix8")
    ap.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--allow-fail", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(C.QWEN_DIR, "tokenizer.json"))
    cfg, st = Q.load_state(C.QWEN_DIR, torch.float32)
    ids = tok.encode(args.prompt).ids

    rep = C.Report(f"S7 回归判据与问题定位 ({args.variant})",
                   json_path=os.path.join(C.RESULTS, "s7_regression.json"),
                   md_path=os.path.join(C.RESULTS, "s7_regression.md"))
    results = collect(args.variant)

    # ---- 权重格式 bit 级回归: 用 ISS 自己解包, 和 Python 参考比
    deq = os.path.join(C.BUILD, "deq")
    pk_path = os.path.join(C.BUILD, "quant", f"{args.variant}.pt")
    image0 = os.path.join(C.BUILD, "npu_device_image")
    if os.path.exists(image0 + ".bin") and os.path.exists(os.path.join(C.BUILD, "npu_iss")):
        C.run([os.path.join(C.BUILD, "npu_iss"), "--dump-dequant", deq,
               "--image", image0 + ".bin", "--idx", image0 + ".idx",
               "--hw", os.path.join(C.HW, "mtx_npu_v0.json")])
    if os.path.exists(pk_path) and os.path.exists(deq):
        pk_all = torch.load(pk_path, map_location="cpu", weights_only=False)
        checked, exact = 0, 0
        for name, pk in pk_all.items():
            f = os.path.join(deq, name + ".f32")
            if not os.path.exists(f):
                continue
            ref = S3.dequant_tensor(pk)
            got = np.fromfile(f, dtype=np.float32).reshape(ref.shape)
            checked += 1
            exact += int((ref.view(np.uint32) == got.view(np.uint32)).all())
        results["pack.bit_exact"] = round(exact / max(1, checked) * 100, 3)
        results["pack.tensors"] = checked

    # ---- 端到端回归指标
    p5 = os.path.join(C.RESULTS, "s5_e2e.json")
    step_rows = []
    if os.path.exists(p5):
        e = C.jload(p5)
        results["e2e.logits_cosine_ab"] = e.get("metrics_ab", {}).get("logits_cosine")
        results["e2e.top1_ab"] = e.get("metrics_ab", {}).get("top1_agreement")
        bc = e.get("metrics_bc", {})
        results["e2e.logits_cosine_bc"] = bc.get("logits_cosine_prefill")
        m = e.get("gen_match_bc", "0/0")
        num, den = (m.split("/") + ["1"])[:2]
        results["e2e.gen_match_bc"] = float(num) / max(1, float(den))
        sm = e.get("step_metrics") or []
        dec = [x["cos_b"] for x in sm if x["phase"].startswith("decode")]
        results["e2e.decode_cosine_min"] = min(dec) if dec else results.get("e2e.logits_cosine_bc")
        results["e2e.argmax_all_match"] = all(x["argmax_match"] for x in sm) if sm else None

    # ---- 可重复性: 同一微码跑两次
    prog = os.path.join(C.BUILD, f"npu_program_{args.variant}_kvfp16.mtx")
    image = os.path.join(C.BUILD, "npu_device_image")
    tokfile = os.path.join(C.BUILD, "npu_tokens.bin")
    det = 0.0
    if os.path.exists(prog) and os.path.exists(image + ".bin"):
        outs = []
        for i in range(2):
            o = os.path.join(C.RESULTS, f"npu_profile_repeat{i}.json")
            C.run([os.path.join(C.BUILD, "npu_iss"), "--program", prog,
                   "--image", image + ".bin", "--idx", image + ".idx",
                   "--tokens", tokfile, "--out", o,
                   "--hw", os.path.join(C.HW, "mtx_npu_v0.json")])
            lg = np.fromfile(os.path.join(C.RESULTS, "npu_logits_prefill.bin"), dtype=np.float32)
            outs.append((C.jload(o)["summary"]["cycles"], lg.copy()))
        same = (outs[0][0] == outs[1][0]) and np.array_equal(outs[0][1], outs[1][1])
        det = 1.0 if same else 0.0
        results["repeat.deterministic"] = det
        rep.kv("repeat_cycles", [outs[0][0], outs[1][0]])

    rows, data = run_gates(args.variant, results)
    rep.h("1. 门禁回归结果", 2)
    rep.table(["判据", "说明", "实测值", "判定", "门槛"], rows)
    n_fail = sum(1 for r in rows if r[3] == "FAIL")
    n_skip = sum(1 for r in rows if r[3] == "SKIP")
    rep.p(f"**{len(rows) - n_fail - n_skip} PASS / {n_fail} FAIL / {n_skip} SKIP**。"
          f"任一 FAIL 时脚本返回非 0, 可以直接挂到 CI/流水线上。")
    rep.kv("gates", data)
    rep.kv("n_fail", n_fail)
    rep.kv("n_pass", len(rows) - n_fail - n_skip)

    # ---- 故障注入定位
    faults = []
    if os.path.exists(pk_path):
        pk_all = torch.load(pk_path, map_location="cpu", weights_only=False)
        faults = fault_experiments(cfg, st, pk_all, ids, rep)
    rep.save()
    C.jdump({"experiments": faults}, os.path.join(C.RESULTS, "s7_fault_injection.json"))
    print(f"门禁: {len(rows) - n_fail - n_skip} PASS / {n_fail} FAIL / {n_skip} SKIP")
    for r in rows:
        if r[3] != "PASS":
            print("  ", r[3], r[0], r[2])
    if faults:
        print("故障定位:", ", ".join(f"{f['fault']}→L{f['localized_layer']}"
                                     f"({'对' if f['localized_correct'] else '偏'})" for f in faults))
    if n_fail and not args.allow_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
