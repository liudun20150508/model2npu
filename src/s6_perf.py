#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6 性能分析与自动化: 吞吐 / TTFT / TPOT / 带宽 / 存储 + 瓶颈归因。

对应 能力项 5。数据来源:
  * results/npu_profile_<variant>.json —— ISS 逐算子 cycle/字节/访存 (硬件计数器口径)
  * results/s1_model_analysis.json    —— roofline 与内存规划
  * results/s3_quant.json             —— 各精度档的存储占用

口径说明 (报告里也会写):
  * 功能结果是真算出来的 (ISS 跑真实数值);
  * cycle/带宽是**建模值** (roofline + 效率因子), 不是芯片实测 —— 用来做瓶颈归因和规划。
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C


def phase_stats(prof):
    ph = {}
    for op in prof["ops"]:
        d = ph.setdefault(op["phase"], {"cycles": 0.0, "dram_bytes": 0, "macs": 0,
                                        "cyc_compute": 0.0, "cyc_mem": 0.0, "ops": 0,
                                        "bound_compute": 0.0, "bound_mem": 0.0})
        d["cycles"] += op.get("cycles", 0)
        d["dram_bytes"] += op.get("dram_bytes", 0)
        d["macs"] += op.get("macs", 0)
        d["cyc_compute"] += op.get("cycles_compute", 0)
        d["cyc_mem"] += op.get("cycles_mem", 0)
        if op.get("cycles_compute", 0) >= op.get("cycles_mem", 0):
            d["bound_compute"] += op.get("cycles", 0)
        else:
            d["bound_mem"] += op.get("cycles", 0)
        d["ops"] += 1
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a16_rtn_mix8")
    ap.add_argument("--profile", default="")
    args = ap.parse_args()
    C.ensure_dirs()
    hw = C.hw_cfg()
    freq = hw["freq_mhz"] * 1e6
    bw = hw["dram"]["bandwidth_gbps"] * 1e9
    bpc = bw / freq          # 每 cycle 可搬的字节数
    eff = hw["efficiency"]
    peak_mac = hw["mac_per_cycle"] * freq

    prof_path = args.profile or os.path.join(C.RESULTS, f"npu_profile_{args.variant}.json")
    if not os.path.exists(prof_path):
        cands = sorted(glob.glob(os.path.join(C.RESULTS, "npu_profile_*.json")))
        if not cands:
            raise SystemExit("找不到 ISS profile, 先跑 s5_e2e.py")
        prof_path = cands[0]
    prof = C.jload(prof_path)
    s1 = C.jload(os.path.join(C.RESULTS, "s1_model_analysis.json"))
    s3 = C.jload(os.path.join(C.RESULTS, "s3_quant.json")) if \
        os.path.exists(os.path.join(C.RESULTS, "s3_quant.json")) else None
    e2e = C.jload(os.path.join(C.RESULTS, "s5_e2e.json")) if \
        os.path.exists(os.path.join(C.RESULTS, "s5_e2e.json")) else None

    ph = phase_stats(prof)
    pre = ph.get("prefill", {})
    dec = [v for k, v in ph.items() if k.startswith("decode")]
    ttft = pre.get("cycles", 0) / freq * 1e3
    tpot = sum(d["cycles"] for d in dec) / max(1, len(dec)) / freq * 1e3
    dec_bytes = sum(d["dram_bytes"] for d in dec) / max(1, len(dec))
    dec_macs = sum(d["macs"] for d in dec) / max(1, len(dec))

    rep = C.Report(f"S6 性能分析与瓶颈归因 ({args.variant})",
                   json_path=os.path.join(C.RESULTS, "s6_perf.json"),
                   md_path=os.path.join(C.RESULTS, "s6_perf.md"))
    rep.p(f"profile: `{os.path.relpath(prof_path, C.ROOT)}` ({prof['summary']['n_ops']} 条指令, "
          f"实测墙钟 {prof['summary']['wall_s']:.1f} s —— ISS 在 CPU 上真算, 不是查表)。")
    rep.p(f"目标档位: {hw['mac_per_cycle']} MAC/cycle @ {hw['freq_mhz']} MHz "
          f"({2 * peak_mac / 1e12:.2f} TFLOP/s), DRAM {hw['dram']['bandwidth_gbps']} GB/s, "
          f"SRAM {hw['sram_usable_kib']} KiB 可用。")

    rep.h("1. 端到端指标", 2)
    rows = [
        ["TTFT (首 token)", f"{ttft:.2f} ms", f"{pre.get('cycles', 0) / 1e6:.2f} Mcycle",
         "prefill 阶段全部算子 + 回传"],
        ["TPOT (单 token)", f"{tpot:.2f} ms", f"{sum(d['cycles'] for d in dec) / max(1, len(dec)) / 1e6:.2f} Mcycle",
         f"{len(dec)} 步 decode 的均值"],
        ["吞吐 (单路)", f"{1000 / tpot:.1f} token/s", "-", "batch=1"],
        ["每 token DRAM 访问", C.human_bytes(dec_bytes), f"{dec_macs / 1e6:.1f} M MAC",
         "权重流式读取主导"],
        ["decode MAC 利用率", f"{dec_macs / (sum(d['cycles'] for d in dec) / max(1, len(dec)) * hw['mac_per_cycle']) * 100:.1f}%",
         "-", f"峰值 {hw['mac_per_cycle']} MAC/cycle"],
        ["decode 平均带宽利用率", f"{dec_bytes / (sum(d['cycles'] for d in dec) / max(1, len(dec)) * bpc) * 100:.1f}%",
         "-", f"峰值 {hw['dram']['bandwidth_gbps']} GB/s"],
        ["权重常驻 (设备镜像)", C.human_bytes(prof["summary"].get("image_bytes", 0)), "-",
         "int4 打包 + scale/zp"],
        ["激活/KV 工作区", C.human_bytes(prof["summary"]["dram_used_bytes"]), "-", "DRAM 侧声明式分配"],
        ["SRAM 峰值占用", C.human_bytes(prof["summary"]["sram_peak_bytes"]), "-",
         f"预算 {C.human_bytes(hw['sram_usable_kib'] * 1024)}"],
    ]
    rep.table(["指标", "值", "cycle/字节", "说明"], rows)
    rep.kv("headline", {"ttft_ms": ttft, "tpot_ms": tpot, "tokens_per_s": 1000 / tpot,
                        "decode_dram_bytes_per_token": dec_bytes,
                        "decode_macs_per_token": dec_macs,
                        "cycles_total": prof["summary"]["cycles"],
                        "macs_total": prof["summary"]["macs"],
                        "pcie_bytes": prof["summary"]["pcie_bytes"],
                        "sram_peak_bytes": prof["summary"]["sram_peak_bytes"]})

    rep.h("2. 逐阶段拆解 (prefill vs decode)", 2)
    rows = []
    for k in sorted(ph, key=lambda x: ph[x]["cycles"], reverse=True):
        v = ph[k]
        rows.append([k, f"{v['cycles'] / 1e6:.2f} Mcycle", f"{v['cycles'] / freq * 1e3:.2f} ms",
                     C.human_bytes(v["dram_bytes"]), f"{v['macs'] / 1e6:.1f} M",
                     f"{v['bound_compute'] / v['cycles'] * 100:.0f}/{v['bound_mem'] / v['cycles'] * 100:.0f}",
                     f"{v['dram_bytes'] / (v['cycles'] * bpc) * 100:.1f}%", v["ops"]])
    rep.p("「算/访」列 = 该阶段里按**主要瓶颈**归类的 cycle 占比 (算力主导 / 访存主导, 两者相加 100%)。")
    rep.table(["阶段", "cycle", "时间", "DRAM 访问", "MAC", "算/访 %", "带宽利用率", "算子数"], rows)

    rep.h("3. Top 耗时算子 (瓶颈定位)", 2)
    by_kind = {}
    for op in prof["ops"]:
        d = by_kind.setdefault(op["kind"], {"cycles": 0.0, "dram": 0, "macs": 0,
                                            "cyc_compute": 0.0, "cyc_mem": 0.0, "n": 0})
        d["cycles"] += op["cycles"]
        d["dram"] += op["dram_bytes"]
        d["macs"] += op["macs"]
        d["cyc_compute"] += op["cycles_compute"]
        d["cyc_mem"] += op["cycles_mem"]
        d["n"] += 1
    tot_cyc = prof["summary"]["cycles"]
    rows = []
    for k, d in sorted(by_kind.items(), key=lambda x: -x[1]["cycles"]):
        aic = d["macs"] / d["dram"] if d["dram"] else 0
        ridge = 2 * peak_mac / bw
        rows.append([k, d["n"], f"{d['cycles'] / 1e6:.2f} Mcycle",
                     f"{d['cycles'] / tot_cyc * 100:.1f}%", C.human_bytes(d["dram"]),
                     f"{aic:.1f}", "compute" if aic >= ridge else "memory",
                     f"{d['cyc_compute'] / d['cycles'] * 100:.0f}%"])
    rep.p(f"算术强度脊点 = {2 * peak_mac / bw:.1f} FLOP/byte; AI < 脊点 → 访存瓶颈。")
    rep.table(["算子", "次数", "cycle", "占比", "DRAM", "AI", "瓶颈", "算力受限%"], rows)
    rep.kv("by_kind", by_kind)
    rep.kv("op_count", prof["summary"]["n_ops"])

    rep.h("4. 长上下文: KV 什么时候变成瓶颈", 2)
    c = C.jload(os.path.join(C.MODELS, "qwen2.5-0.5b-instruct", "config.json"))
    L, kvh, D = c["num_hidden_layers"], c["num_key_value_heads"], c["hidden_size"] // c["num_attention_heads"]
    kv_per_tok = 2 * L * kvh * D * 2
    weights_bytes = prof["summary"]["dram_used_bytes"]
    w_only = s3["variants"][args.variant]["bytes"] if (s3 and args.variant in s3.get("variants", {})) else 0
    rows = []
    for ctx in [512, 1024, 2048, 4096, 8192, 32768]:
        kv_fp16 = kv_per_tok * ctx
        kv_i8 = kv_fp16 // 2
        t_kv = kv_i8 / (bw * eff["dram_util"]) * 1e3
        t_w = dec_bytes / (bw * eff["dram_util"]) * 1e3
        rows.append([ctx, C.human_bytes(kv_fp16), C.human_bytes(kv_i8),
                     f"{kv_i8 / max(1, dec_bytes) * 100:.0f}%",
                         f"{t_w:.2f}", f"{t_kv:.2f}", f"{t_w + t_kv:.2f}"])
    rep.p(f"KV 每 token = {kv_per_tok} B (fp16, {L} 层 × {kvh} KV 头 × {D} head_dim × 2); "
          f"int4 权重每 token 读 {C.human_bytes(dec_bytes)}。TPOT 预测 = 权重流式读取 + KV 读取。")
    rep.table(["上下文", "KV fp16", "KV int8", "KV/权重", "权重时间(ms)", "KV 时间(ms)", "TPOT 预测(ms)"], rows)
    cross = dec_bytes / (kv_per_tok / 2)
    rep.p(f"**KV 反超权重的临界点 ≈ {cross / 1000:.1f}K token (int8 KV)** —— "
          f"这就是「端侧要么限上下文, 要么做 KV 量化」的定量依据。")
    rep.kv("kv_crossover_tokens_int8", cross)
    rep.kv("kv_table", rows)

    if s3:
        rep.h("5. 存储占用: 各精度档对比 (含权重打包格式的元数据开销)", 2)
        rows = []
        for v, d in s3["variants"].items():
            rows.append([v, f"{d['bytes_mib']:.1f} MiB", f"{d['compress_ratio']:.2f}x",
                         f"{d['logits_cosine']:.6f}", f"{d['top1_agreement'] * 100:.1f}%",
                         f"{d['gen_token_match'] * 100:.0f}%", f"{d['nll']:.4f}"])
        rep.table(["变体", "设备侧权重", "压缩比", "logits cosine", "top-1", "输出一致", "NLL"], rows)
        rep.kv("storage", s3["variants"])

    if e2e and e2e.get("iss_summary"):
        rep.h("6. 自动化判据 (给回归用)", 2)
        rep.p(f"端到端对齐: prefill {e2e.get('metrics_bc', {}).get('logits_cosine_prefill')} "
              f"(ISS vs 量化模拟); 生成结果一致 {e2e.get('gen_match_bc')}。")
    rep.h("7. 结论 (一句话口径)", 2)
    rep.p(f"- decode 是**纯访存瓶颈**: 每 token 读 {C.human_bytes(dec_bytes)} 权重, "
          f"算力利用率仅 {dec_macs / (sum(d['cycles'] for d in dec) / max(1, len(dec)) * hw['mac_per_cycle']) * 100:.0f}%, "
          f"所以端侧主档必须是 W4A16 而不是 FP16。")
    rep.p(f"- prefill 是**算力瓶颈**: {pre.get('cycles', 0) / 1e6:.2f} Mcycle 里算力受限占大头, "
          f"优化方向是分块/两阶段 GEMM 与 GQA 广播而不是省带宽。")
    rep.p(f"- KV 在 {cross / 1000:.1f}K 上下文以上超过权重, 必须 int8/分页; 长上下文要先算这笔账再改模型。")
    rep.save()
    print(f"TTFT {ttft:.2f} ms, TPOT {tpot:.2f} ms ({1000 / tpot:.1f} tok/s), "
          f"decode {C.human_bytes(dec_bytes)}/token")


if __name__ == "__main__":
    main()
