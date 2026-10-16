#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 模型分析与部署规划。

对应 能力项 1: 分析 Transformer/视觉编码器结构, 梳理 GEMM/Attention/GQA/MLP/
Norm/RoPE/KV Cache, 结合算力、容量、带宽制定算子映射、数据布局和内存规划方案。

产出:
  results/s1_model_analysis.json / .md   结构与部署规划
  results/s1_op_table.csv                逐算子形状/计算量/访存量/瓶颈判定
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_qwen as Q


def tensor_inventory(model_dir):
    """只读 safetensors 头部拿到每个张量的形状/dtype（不加载权重, 秒级）。"""
    import struct
    p = os.path.join(model_dir, "model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    inv = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        inv[k] = {"shape": v["shape"], "dtype": v["dtype"],
                  "numel": int(_prod(v["shape"]))}
    return inv


def _prod(x):
    r = 1
    for i in x:
        r *= i
    return r


def gemm_row(host, op, T, N, K, hw, wbits=4, act_dtype="fp16", note="", extra_bytes=0):
    """一个 GEMM 的 roofline 行。"""
    macs = T * N * K
    flops = 2 * macs
    w_bytes = N * K * (wbits / 8.0)
    a_bytes = T * K * (2 if act_dtype == "fp16" else 4)
    o_bytes = T * N * (2 if act_dtype == "fp16" else 4)
    bytes_moved = w_bytes + a_bytes + o_bytes + extra_bytes
    ai = flops / bytes_moved
    peak_mac = hw["mac_per_cycle"] * hw["freq_mhz"] * 1e6
    eff = hw["efficiency"]
    peak_flops = 2 * peak_mac
    peak_bw = hw["dram"]["bandwidth_gbps"] * 1e9
    ai_ridge = peak_flops / (peak_bw)
    t_compute = macs / (peak_mac * eff["pe_util"])
    t_mem = bytes_moved / (peak_bw * eff["dram_util"])
    t = max(t_compute, t_mem)
    return {
        "host": host, "op": op, "T": T, "N": N, "K": K,
        "macs": macs, "flops": flops,
        "w_bytes": int(w_bytes), "a_bytes": int(a_bytes), "o_bytes": int(o_bytes),
        "bytes": int(bytes_moved),
        "ai": round(ai, 2), "ai_ridge": round(ai_ridge, 1),
        "bound": "compute" if ai >= ai_ridge else "memory",
        "t_compute_us": round(t_compute * 1e6, 2),
        "t_mem_us": round(t_mem * 1e6, 2),
        "t_us": round(t * 1e6, 2),
        "note": note,
    }


def layer_ops(cfg, T, hw, wbits=4):
    """一层 Transformer 的全部 GEMM 访存/计算量（不含 Attention 内部）。"""
    H, I, kv, D = cfg.hidden, cfg.inter, cfg.kv_heads, cfg.head_dim
    rows = []
    rows.append(gemm_row("decoder", "q_proj", T, H, H, hw, wbits, note="含 bias"))
    rows.append(gemm_row("decoder", "k_proj", T, kv * D, H, hw, wbits, note="GQA: N=128 远小于 H"))
    rows.append(gemm_row("decoder", "v_proj", T, kv * D, H, hw, wbits, note="GQA"))
    rows.append(gemm_row("decoder", "o_proj", T, H, H, hw, wbits))
    rows.append(gemm_row("decoder", "gate_proj", T, I, H, hw, wbits, note="SwiGLU"))
    rows.append(gemm_row("decoder", "up_proj", T, I, H, hw, wbits, note="SwiGLU"))
    rows.append(gemm_row("decoder", "down_proj", T, H, I, hw, wbits))
    # Attention 内部: QK^T 与 PV, 访存量按 KV 长度 S 计
    S = T if T > 1 else 1
    return rows


def attn_ops(cfg, T, S, hw, act_dtype="fp16"):
    """Attention 内部两个矩阵乘 (prefill 时 T 个 query)。"""
    Hq, kv, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    rows = []
    # QK^T: [Hq*T, D] x [D, S]
    macs = Hq * T * D * S
    b = (Hq * T * D + kv * S * D) * 2 + Hq * T * S * 2
    rows.append({"host": "attn", "op": "QK^T (GQA 广播)", "T": T, "N": S, "K": D,
                 "macs": macs, "flops": 2 * macs, "w_bytes": 0,
                 "a_bytes": (Hq * T * D + kv * S * D) * 2, "o_bytes": Hq * T * S * 2,
                 "bytes": b, "note": f"Hq={Hq} Hkv={kv} S={S}"})
    macs2 = Hq * T * S * D
    b2 = (Hq * T * S + kv * S * D) * 2 + Hq * T * D * 2
    rows.append({"host": "attn", "op": "PV (softmax 后)", "T": T, "N": D, "K": S,
                 "macs": macs2, "flops": 2 * macs2, "w_bytes": 0,
                 "a_bytes": (Hq * T * S + kv * S * D) * 2, "o_bytes": Hq * T * D * 2,
                 "bytes": b2, "note": "softmax 后重排"})
    for r in rows:
        peak_mac = hw["mac_per_cycle"] * hw["freq_mhz"] * 1e6
        peak_flops = 2 * peak_mac
        peak_bw = hw["dram"]["bandwidth_gbps"] * 1e9
        eff = hw["efficiency"]
        r["ai"] = round(r["flops"] / r["bytes"], 2)
        r["ai_ridge"] = round(peak_flops / peak_bw, 1)
        r["bound"] = "compute" if r["ai"] >= r["ai_ridge"] else "memory"
        r["t_compute_us"] = round(r["macs"] / (peak_mac * eff["pe_util"]) * 1e6, 2)
        r["t_mem_us"] = round(r["bytes"] / (peak_bw * eff["dram_util"]) * 1e6, 2)
        r["t_us"] = round(max(r["t_compute_us"], r["t_mem_us"]), 2)
    return rows


def sram_plan(cfg, hw, T_tile, wbits=4):
    """一层的算子内 SRAM 分块方案（权重 tile + 激活 tile + fp32 输出 tile, 双缓冲）。

    允许同时切 N 与 K: K 向分块时用 fp32 部分和累加 (标准做法)。
    """
    H, I = cfg.hidden, cfg.inter
    budget = hw["sram_usable_kib"] * 1024
    plans = []
    for name, N, K in [("q/k/v/o", H, H), ("gate/up", I, H), ("down", H, I)]:
        best = None
        for N_tile in [N, 512, 256, 128, 64, 32]:
            for K_tile in [K, K // 2, K // 4, K // 8, 256, 128, 64]:
                if K_tile <= 0 or K_tile > K:
                    continue
                w_b = N_tile * K_tile * wbits / 8      # int4 权重 tile
                a_b = T_tile * K_tile * 2              # fp16 激活 tile
                o_b = T_tile * N_tile * 4              # fp32 累加器 (K 分块时跨 tile 保留)
                tot = 2 * (w_b + a_b + o_b)            # 双缓冲
                if tot <= budget:
                    nt, kt = (N + N_tile - 1) // N_tile, (K + K_tile - 1) // K_tile
                    best = {"gemm": name, "N": N, "K": K, "N_tile": N_tile, "K_tile": K_tile,
                            "T_tile": T_tile, "sram_bytes": int(tot),
                            "sram_kib": round(tot / 1024, 1),
                            "util_pct": round(tot / budget * 100, 1),
                            "n_tiles": nt, "k_tiles": kt, "dma_tiles": nt * kt,
                            "w_dram_bytes": int(N * K * wbits / 8),
                            "w_reload_bytes": int(min(nt * kt, 99) * w_b)}
                    break
            if best:
                break
        plans.append(best or {"gemm": name, "N": N, "K": K, "N_tile": None, "K_tile": None,
                              "T_tile": T_tile, "sram_bytes": None, "note": "装不下, 需要更多 SRAM 或更小 T_tile"})
    return {"budget_kib": hw["sram_usable_kib"], "T_tile": T_tile, "plans": plans}


def main():
    C.ensure_dirs()
    hw = C.hw_cfg()
    cfg, _ = Q.load_state(C.QWEN_DIR)
    inv = tensor_inventory(C.QWEN_DIR)
    hw_dt = hw["mac_per_cycle"] * hw["freq_mhz"] * 1e6

    rep = C.Report("S1 模型分析与部署规划 —— Qwen2.5-0.5B-Instruct @ MTX-NPU v0",
                   json_path=os.path.join(C.RESULTS, "s1_model_analysis.json"),
                   md_path=os.path.join(C.RESULTS, "s1_model_analysis.md"))
    rep.h("1. 模型结构与参数量", 2)
    comp = {}
    for k, v in inv.items():
        if "embed_tokens" in k or "lm_head" in k:
            c = "embedding / lm_head (tied)"
        elif "self_attn" in k:
            c = "attention 投影"
        elif "mlp" in k:
            c = "MLP (SwiGLU)"
        elif "norm" in k:
            c = "RMSNorm"
        else:
            c = "其他"
        comp[c] = comp.get(c, 0) + v["numel"]
    rows = []
    tot = sum(comp.values())
    for c, n in sorted(comp.items(), key=lambda x: -x[1]):
        rows.append([c, f"{n / 1e6:.2f} M", f"{n / tot * 100:.1f}%",
                     C.human_bytes(n * 2), C.human_bytes(n), C.human_bytes(n / 2)])
    rows.append(["合计", f"{tot / 1e6:.2f} M", "100%", C.human_bytes(tot * 2),
                 C.human_bytes(tot), C.human_bytes(tot / 2)])
    rep.p("权重构成与不同精度下的存储占用（embedding 与 lm_head 绑定, 只算一份）:")
    rep.table(["模块", "参数量", "占比", "fp16", "int8", "int4"], rows)
    rep.kv("params", {c: n for c, n in comp.items()})

    rep.h("2. 关键计算梳理: 逐算子 roofline", 2)
    op_rows = []
    for tag, T in [("decode (T=1)", 1), ("prefill (T=256)", 256)]:
        for r in layer_ops(cfg, T, hw):
            r["tag"] = tag
            op_rows.append(r)
        for r in attn_ops(cfg, T, T, hw):
            r["tag"] = tag
            op_rows.append(r)
    hdr = ["场景", "算子", "T", "N", "K", "MACs", "访存", "AI", "脊点", "瓶颈", "t_compute", "t_mem", "t"]
    rows = []
    for r in op_rows:
        rows.append([r["tag"], r["op"], r["T"], r["N"], r["K"], f"{r['macs'] / 1e6:.3f}M",
                     C.human_bytes(r["bytes"]), f"{r['ai']:.2f}", f"{r['ai_ridge']:.0f}",
                     r["bound"], f"{r['t_compute_us']:.2f}us", f"{r['t_mem_us']:.2f}us",
                     f"{r['t_us']:.2f}us"])
    rep.p(f"算力脊点: 峰值 {2 * hw_dt / 1e12:.2f} TFLOP/s (int4 PE 利用率 {hw['efficiency']['pe_util']}), "
          f"DRAM 带宽脊点 AI = {2 * hw_dt / (hw['dram']['bandwidth_gbps'] * 1e9):.1f} FLOP/byte。"
          f"**AI < 脊点 = 访存瓶颈**。")
    rep.table(hdr, rows)
    rep.kv("op_rows", op_rows)
    with open(os.path.join(C.RESULTS, "s1_op_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(op_rows[0].keys()))
        w.writeheader()
        w.writerows(op_rows)

    rep.h("3. KV Cache 预算（长上下文的关键约束）", 2)
    kv_rows = []
    L, kvh, D = cfg.layers, cfg.kv_heads, cfg.head_dim
    for ctx in [1024, 4096, 8192, 32768]:
        fp16 = 2 * L * kvh * D * ctx * 2
        int8 = fp16 / 2
        kv_rows.append([f"{ctx}", C.human_bytes(fp16), C.human_bytes(int8),
                        C.human_bytes(fp16 / ctx), C.human_bytes(int8 / ctx),
                        f"{fp16 / (tot * 0.5) * 100:.0f}%", f"{int8 / (tot * 0.5) * 100:.0f}%"])
    rep.p(f"KV 每 token = 2(层) × {L}层 × {kvh} KV头 × {D} head_dim = {2 * L * kvh * D} 元素; "
          f"int4 权重下模型约 {C.human_bytes(tot / 2)}。")
    rep.table(["上下文", "KV fp16", "KV int8", "每 token fp16", "每 token int8",
               "占模型(int4)%", "占模型(int8 KV)%"], kv_rows)
    rep.kv("kv_table", [{"ctx": c, "fp16_bytes": 2 * L * kvh * D * c * 2,
                         "int8_bytes": L * kvh * D * c * 2} for c in [1024, 4096, 8192, 32768]])

    rep.h("4. 片上 SRAM 分块与内存规划", 2)
    plan = sram_plan(cfg, hw, T_tile=32)
    rep.p(f"SRAM 预算 {hw['sram_usable_kib']} KiB（可用, 双缓冲）; T_tile={plan['T_tile']}。"
          f"权重常驻 DRAM 流式读取, 分块只缓存一个 N_tile 的权重。")
    rep.table(["GEMM", "N", "K", "N_tile", "K_tile", "N×K tile 数", "SRAM 占用", "占用率"],
               [[p["gemm"], p["N"], p["K"], p.get("N_tile"), p.get("K_tile"),
                 f"{p.get('n_tiles', '-')}×{p.get('k_tiles', '-')}",
                 C.human_bytes(p["sram_bytes"]) if p["sram_bytes"] else "装不下",
                 f"{p.get('util_pct')}%" if p["sram_bytes"] else "-"] for p in plan["plans"]])
    rep.kv("sram_plan", plan)

    rep.h("5. 算子映射表（模型算子 → NPU ISA / 数据布局 / 精度）", 2)
    mapping = [
        ["Embedding gather", "EMBED", "[T] → [T,H]", "查表, 权重 int8/fp16", "DRAM 随机读 896·2B/token"],
        ["RMSNorm", "RMSNORM", "[T,H] fp16, w[H] fp16", "方差 fp32 累加, eps=1e-6", "向量, 384 cycle 固定开销"],
        ["q/k/v 投影 GEMM", "GEMM4", "W[N,K] int4 g128 打包 / x[T,K] fp16", "y=x·Wᵀ+b, fp32 累加", "prefill 算力瓶颈 / decode 带宽瓶颈"],
        ["RoPE", "ROPE", "[T,Hn,D] 前后半配对", "cos/sin 表 fp16, theta=1e6", "向量, 320 cycle"],
        ["KV 写入", "KVAPPEND", "cache[S,Hkv,D] 追加", "环形/分页地址, 支持 int8 KV", "decode 每步 2·Hkv·D·2B·L"],
        ["Attention (GQA)", "ATTNGQA", "Q[T,Hq,D], KV[S,Hkv,D]", "Hq/Hkv=7 广播, 因果掩码, softmax fp32", "长上下文时 KV 读带宽主导"],
        ["MLP SwiGLU", "SILUMUL", "gate/up[T,I] fp16", "silu(gate)·up", "向量"],
        ["输出投影 GEMM", "GEMM4", "W[H,I] int4 g128", "fp32 累加", "decode 带宽瓶颈"],
        ["LM Head", "GEMM8/GEMM4", "W[V,H] · [T,H]", "与 embedding 绑定; 混合精度可留 int8", "decode 最大单算子 (V=151936)"],
        ["ViT Patch Embed", "IM2COL+GEMM", "img[3,224,224] → [196,192]", "im2col 展开 stride=16", "算力瓶颈"],
        ["ViT Attention", "ATTNGQA(非因果)", "[197,192] + cls token", "无掩码, 全局可见", "197 token 全连接"],
        ["ViT MLP/GELU", "GEMMF + GELU", "[197,768]", "erf 版 GELU", "算力瓶颈"],
    ]
    rep.table(["模型算子", "NPU 指令", "数据形状/布局", "精度与规则", "瓶颈"], mapping)
    rep.kv("op_mapping", [{"op": m[0], "isa": m[1], "layout": m[2], "dtype": m[3], "bottleneck": m[4]}
                          for m in mapping])

    rep.h("6. 结论（规划口径）", 2)
    dec = [r for r in op_rows if r["tag"] == "decode (T=1)"]
    pre = [r for r in op_rows if r["tag"] == "prefill (T=256)"]
    rep.p(f"- decode 单 token: 一层 {'/'.join(sorted(set(r['bound'] for r in dec)))} 瓶颈, "
          f"整模型 24 层 GEMM 访存 {C.human_bytes(sum(r['bytes'] for r in dec) * cfg.layers)}/token。")
    rep.p(f"- prefill (256 token): 一层 GEMM 以 {'/'.join(sorted(set(r['bound'] for r in pre)))} 为主。")
    rep.p("- 结论: **decode 用 W4A16 降权重量, prefill 靠分块把 GEMM 拉回算力瓶颈; "
          "KV Cache 在 8K 以上必须 int8, 否则 KV 体积会超过 int4 权重**。")
    rep.save()
    print(json.dumps({k: rep.data[k] for k in ("params",)}, ensure_ascii=False)[:200])
    print("写入:", os.path.join(C.RESULTS, "s1_model_analysis.md"))


if __name__ == "__main__":
    main()
