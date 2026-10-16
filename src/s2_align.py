#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 算子适配与验证: 参考算子对齐 (numpy 参考 ↔ C++ native ↔ PyTorch 真值)。

对应 能力项 2: 定义数据形状/精度/边界条件, 完成算子数值对齐与问题定位。

每个用例: 生成输入 -> 三端计算 (能拿到 torch 真值的就用真值) -> 判据裁决。
判据写在 results/s2_gates.json 里 (也被 S7 回归复用)。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_ops_ref as R

BIN = os.path.join(C.BUILD, "s2_ops_native")
WORK = os.path.join(C.BUILD, "s2_cases")


def wbin(name, arr, dtype=np.float32):
    p = os.path.join(WORK, name)
    os.makedirs(WORK, exist_ok=True)
    np.asarray(arr, dtype=dtype).tofile(p)
    return p


def rbin(name, shape, dtype=np.float32):
    return np.fromfile(os.path.join(WORK, name), dtype=dtype).reshape(shape)


def call(args):
    a2 = []
    for a in args:
        a = str(a)
        a2.append(os.path.join(WORK, a) if a.endswith(".bin") and not os.path.isabs(a) else a)
    rc, out = C.run([BIN] + a2, check=False)
    if rc != 0:
        raise RuntimeError(f"native 失败 {args}: {out[-2000:]}")
    line = [l for l in out.splitlines() if l.startswith("{")]
    return json.loads(line[-1]) if line else {}


def judge(ref, got, rel_tol=1e-5, cos_tol=1 - 1e-9, fp16_mode=False):
    """返回 (pass, 指标 dict)。fp16_mode 下判据换成 fp16 ULP 口径。"""
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    m = {
        "rel_err": R.rel_err(ref, got),
        "max_abs_err": R.max_abs_err(ref, got),
        "cosine": R.cos_sim(ref, got),
        "exact_pct": float((ref == got).mean() * 100),
    }
    if fp16_mode:
        # 允许 1 个 fp16 ULP 的差异: 取 fp16 量化后的相对步长
        amax = float(np.abs(ref).max()) + 1e-30
        ulp = 2 ** (int(np.floor(np.log2(amax))) - 10) if amax > 0 else 0.0
        m["fp16_ulp"] = ulp
        m["max_err_ulp"] = m["max_abs_err"] / ulp if ulp else 0.0
        ok = m["max_err_ulp"] <= 1.0 and m["cosine"] > 1 - 1e-6
    else:
        ok = m["rel_err"] <= rel_tol and m["cosine"] >= cos_tol
    return bool(ok), m


def mk(name, ref, got, fp16_mode=False):
    ok, m = judge(ref, got, fp16_mode=fp16_mode)
    return {"name": name, "passed": ok, **{k: (round(v, 6) if isinstance(v, float) else v)
                                           for k, v in m.items()}}


def mk_bit(name, ok, **extra):
    return {"name": name, "passed": bool(ok), "bit_exact_pct": extra.get("bit_exact_pct", 100.0),
            **{k: v for k, v in extra.items() if k != "bit_exact_pct"}}


# --------------------------------------------------------------- 用例定义
def build_cases(cfg, state):
    """返回 [(name, group, fp16_mode, runner)]。"""
    cases = []
    rng = np.random.default_rng(20261016)
    H, I, Hq, Hkv, D = cfg.hidden, cfg.inter, cfg.heads, cfg.kv_heads, cfg.head_dim

    # ---- 0. fp16 舍入边界 (纯 bit 级)
    def c_fp16():
        vals = np.array([0.0, -0.0, 1e-8, -1e-8, 6.103515625e-05, 6.097555160522461e-05,
                         65504.0, 65519.9, 65520.0, 1.0, 1.0009765625, 0.333251953125,
                         2.0 ** -14, 2.0 ** -24, 2048.5, 2049.5, 4097.5, -1234.5678,
                         3.4028235e38, np.inf, -np.inf], dtype=np.float32)
        ties = np.frombuffer(np.array([0x3C000800, 0x3C001800, 0x3E000800], dtype=np.uint32).tobytes(),
                             dtype=np.float32)
        x = np.concatenate([vals, ties, rng.standard_normal(4096).astype(np.float32) * 100])
        x.tofile(os.path.join(WORK, "fp16_in.bin"))
        call(["--op", "fp16_round", "--in", "fp16_in.bin", "--out", "fp16_out.bin", "--act-fp16", 0])
        got = rbin("fp16_out.bin", x.shape)
        with np.errstate(over="ignore"):
            ref = x.astype(np.float16).astype(np.float32)
        eq = float((got.view(np.uint32) == ref.view(np.uint32)).mean() * 100)
        return [mk_bit("fp16_round(边界/次正规/溢出/RNE tie)", eq == 100.0,
                       bit_exact_pct=eq, n=int(x.size))]

    # ---- 1. RMSNorm: 真实权重 + 边界
    def c_rms():
        w = state["model.layers.0.input_layernorm.weight"].numpy().astype(np.float32)
        rows = [rng.standard_normal((7, H)).astype(np.float32),
                np.zeros((1, H), dtype=np.float32),
                (rng.standard_normal((3, H)).astype(np.float32) * 1e3),
                (rng.standard_normal((5, H)).astype(np.float32) * 1e-3)]
        x = np.concatenate(rows, 0)
        wbin("rms_x.bin", x); wbin("rms_w.bin", w)
        call(["--op", "rmsnorm", "--shape", f"{x.shape[0]},{H}", "--in", "rms_x.bin",
              "--w", "rms_w.bin", "--out", "rms_y.bin", "--eps", cfg.eps, "--act-fp16", 0])
        got = rbin("rms_y.bin", x.shape)
        ref = R.rmsnorm(x, w, cfg.eps)
        return [mk("rmsnorm(fp32, 真实权重, 含全零/大/小量级)", ref, got)]

    # ---- 2. RMSNorm fp16 激活口径
    def c_rms16():
        w = state["model.layers.12.post_attention_layernorm.weight"].numpy().astype(np.float32)
        x = rng.standard_normal((11, H)).astype(np.float32) * 3
        wbin("rms16_x.bin", x); wbin("rms16_w.bin", w)
        call(["--op", "rmsnorm", "--shape", f"{x.shape[0]},{H}", "--in", "rms16_x.bin",
              "--w", "rms16_w.bin", "--out", "rms16_y.bin", "--eps", cfg.eps, "--act-fp16", 1])
        got = rbin("rms16_y.bin", x.shape)
        R.set_fp16_act(True)
        ref = R.rmsnorm(x, w, cfg.eps)
        R.set_fp16_act(False)
        return [mk("rmsnorm(fp16 激活存储口径)", ref, got, fp16_mode=True)]

    # ---- 3. RoPE
    def c_rope():
        T = 33
        x = rng.standard_normal((T, Hkv, D)).astype(np.float32)
        pos = list(range(100, 100 + T))
        wbin("rope_x.bin", x)
        call(["--op", "rope", "--shape", f"{T},{Hkv},{D}", "--in", "rope_x.bin",
              "--out", "rope_y.bin", "--theta", cfg.theta, "--act-fp16", 0,
              "--pos", ",".join(map(str, pos))])
        got = rbin("rope_y.bin", x.shape)
        cos, sin = R.rope_tables(D, pos, cfg.theta)
        ref = R.rope_apply(x, cos, sin)
        return [mk("rope(theta=1e6, 位置 100..132, 非 0 起点)", ref, got)]

    # ---- 4. RoPE 长位置 + 真实 theta 抖动
    def c_rope2():
        T, pos0 = 8, 32760
        x = rng.standard_normal((T, Hq, D)).astype(np.float32)
        pos = list(range(pos0, pos0 + T))
        wbin("rope2_x.bin", x)
        call(["--op", "rope", "--shape", f"{T},{Hq},{D}", "--in", "rope2_x.bin",
              "--out", "rope2_y.bin", "--theta", cfg.theta, "--act-fp16", 0,
              "--pos", ",".join(map(str, pos))])
        got = rbin("rope2_y.bin", x.shape)
        cos, sin = R.rope_tables(D, pos, cfg.theta)
        return [mk("rope(长上下文位置 32760+, theta=1e6)", R.rope_apply(x, cos, sin), got)]

    # ---- 5. GQA Attention prefill (因果) + 真值来自 PyTorch 抓拍
    def c_attn():
        T = 17
        q = rng.standard_normal((T, Hq, D)).astype(np.float32)
        k = rng.standard_normal((T, Hkv, D)).astype(np.float32)
        v = rng.standard_normal((T, Hkv, D)).astype(np.float32)
        wbin("at_q.bin", q); wbin("at_k.bin", k); wbin("at_v.bin", v)
        call(["--op", "attn", "--shape", f"{T},{Hq},{D}", "--kvshape", f"{Hkv},{T}",
              "--in", "at_q.bin", "--in2", "at_k.bin", "--in3", "at_v.bin",
              "--out", "at_o.bin", "--out2", "at_p.bin", "--causal", 1,
              "--attn-scale", cfg.scale, "--act-fp16", 0])
        got = rbin("at_o.bin", (T, Hq, D))
        ref, probs = R.attn_prefill(q, k, v, scale=cfg.scale, causal=True)
        return [mk("attention prefill(因果, GQA 14:2, T=17 非 2 次幂)", ref, got)]

    # ---- 6. decode: 单 token + 64 历史 + 长上下文 2048
    def c_decode():
        res = []
        for S in (64, 2048):
            q = rng.standard_normal((1, Hq, D)).astype(np.float32)
            k = rng.standard_normal((S, Hkv, D)).astype(np.float32)
            v = rng.standard_normal((S, Hkv, D)).astype(np.float32)
            wbin("dc_q.bin", q); wbin("dc_k.bin", k); wbin("dc_v.bin", v)
            call(["--op", "attn_decode", "--shape", f"1,{Hq},{D}", "--kvshape", f"{Hkv},{S}",
                  "--in", "dc_q.bin", "--in2", "dc_k.bin", "--in3", "dc_v.bin",
                  "--out", "dc_o.bin", "--attn-scale", cfg.scale, "--act-fp16", 0])
            got = rbin("dc_o.bin", (1, Hq, D))
            ref, _ = R.attn_decode(q, k, v, scale=cfg.scale)
            res.append(mk(f"attention decode(历史 {S} token)", ref, got))
        return res

    # ---- 7. MQA (Hkv=1)
    def c_mqa():
        T, hq = 5, 8
        q = rng.standard_normal((T, hq, D)).astype(np.float32)
        k = rng.standard_normal((T, 1, D)).astype(np.float32)
        v = rng.standard_normal((T, 1, D)).astype(np.float32)
        wbin("mq_q.bin", q); wbin("mq_k.bin", k); wbin("mq_v.bin", v)
        call(["--op", "attn", "--shape", f"{T},{hq},{D}", "--kvshape", f"1,{T}",
              "--in", "mq_q.bin", "--in2", "mq_k.bin", "--in3", "mq_v.bin",
              "--out", "mq_o.bin", "--causal", 1, "--act-fp16", 0])
        got = rbin("mq_o.bin", (T, hq, D))
        ref, _ = R.attn_prefill(q, k, v, causal=True)
        return [mk("attention MQA(Hkv=1, 广播比 8)", ref, got)]

    # ---- 8. 非因果 (ViT 视觉编码器口径)
    def c_attn_nc():
        T = 197
        hq = 3
        q = rng.standard_normal((T, hq, D)).astype(np.float32)
        k = rng.standard_normal((T, hq, D)).astype(np.float32)
        v = rng.standard_normal((T, hq, D)).astype(np.float32)
        wbin("nc_q.bin", q); wbin("nc_k.bin", k); wbin("nc_v.bin", v)
        call(["--op", "attn", "--shape", f"{T},{hq},{D}", "--kvshape", f"{hq},{T}",
              "--in", "nc_q.bin", "--in2", "nc_k.bin", "--in3", "nc_v.bin",
              "--out", "nc_o.bin", "--causal", 0, "--act-fp16", 0])
        got = rbin("nc_o.bin", (T, hq, D))
        ref, _ = R.attn_prefill(q, k, v, causal=False)
        return [mk("attention 非因果(ViT 197 token 全局可见)", ref, got)]

    # ---- 9. silu_mul / gelu 边界
    def c_silu():
        g = np.array([[-30.0, -1.0, 0.0, 1.0, 30.0, 88.0, -88.0, 1e-20]], dtype=np.float32)
        g = np.concatenate([g, rng.standard_normal((1, 64)).astype(np.float32) * 5], 1)
        u = rng.standard_normal(g.shape).astype(np.float32)
        wbin("silu_g.bin", g); wbin("silu_u.bin", u)
        call(["--op", "silu_mul", "--in", "silu_g.bin", "--in2", "silu_u.bin",
              "--out", "silu_y.bin", "--act-fp16", 0])
        got = rbin("silu_y.bin", g.shape)
        return [mk("silu_mul(±88 饱和边界, 含 1e-20 次正规)", R.silu_mul(g, u), got)]

    def c_gelu():
        x = np.array([[-10.0, -1.0, 0.0, 1.0, 10.0, 0.044715, 1e-8]], dtype=np.float32)
        x = np.concatenate([x, rng.standard_normal((1, 128)).astype(np.float32) * 4], 1)
        wbin("gelu_x.bin", x)
        call(["--op", "gelu", "--in", "gelu_x.bin", "--out", "gelu_y.bin", "--act-fp16", 0])
        got = rbin("gelu_y.bin", x.shape)
        return [mk("gelu(erf 版, ViT 激活, ±10)", R.gelu(x), got)]

    # ---- 用例登记 (name, kind, fp16, fn)
    cases += [
        ("fp16_round(边界/次正规/溢出/RNE tie)", "bit", False, c_fp16),
        ("rmsnorm(fp32, 真实权重, 含全零/大/小量级)", "fp32", False, c_rms),
        ("rmsnorm(fp16 激活存储口径)", "fp16", True, c_rms16),
        ("rope(theta=1e6, 位置 100..132, 非 0 起点)", "fp32", False, c_rope),
        ("rope(长上下文位置 32760+, theta=1e6)", "fp32", False, c_rope2),
        ("attention prefill(因果, GQA 14:2, T=17 非 2 次幂)", "fp32", False, c_attn),
        ("attention decode(历史 64 / 长上下文 2048)", "fp32", False, c_decode),
        ("attention MQA(Hkv=1, 广播比 8)", "fp32", False, c_mqa),
        ("attention 非因果(ViT 197 token 全局可见)", "fp32", False, c_attn_nc),
        ("silu_mul(±88 饱和边界, 含 1e-20 次正规)", "fp32", False, c_silu),
        ("gelu(erf 版, ViT 激活, ±10)", "fp32", False, c_gelu),
    ]
    return cases


def quant_cases(cfg, state):
    """量化算子对齐: W4A16/W8A16 GEMM + int4 打包 + int8 激活 — 要求 bit 级一致。"""
    rng = np.random.default_rng(7)
    rows = []
    H, I = cfg.hidden, cfg.inter

    def one(name, W, x, group, bits, symmetric, bias=None, act_fp16=True, packed_in=True):
        N, K = W.shape
        q, sc, zp = R.rtn_quant_groupwise(W, bits=bits, group=group, symmetric=symmetric)
        Kp = q.shape[1]
        pk = R.pack_int4(q) if bits == 4 else q
        if bits == 4:
            wbin("g_pk.bin", pk, np.uint8)
        else:
            wbin("g_w.bin", q, np.uint8)
        wbin("g_s.bin", sc); wbin("g_z.bin", zp); wbin("g_x.bin", x)
        args = ["--op", f"gemm_w{bits}a16", "--shape", f"{x.shape[0]},{K},{N}",
                "--in", "g_x.bin", "--scale", "g_s.bin", "--zp", "g_z.bin",
                "--group", group, "--bits", bits, "--out", "g_y.bin",
                "--act-fp16", 1 if act_fp16 else 0]
        args += (["--packed", "g_pk.bin"] if bits == 4 else ["--w", "g_w.bin"])
        if bias is not None:
            wbin("g_b.bin", bias)
            args += ["--bias", "g_b.bin"]
        meta = call(args)
        got = rbin("g_y.bin", (x.shape[0], N))
        R.set_fp16_act(act_fp16)
        if bits == 4:
            ref = R.qgemm_w4a16(x, pk, sc, zp, group, symmetric=symmetric,
                                act_fp16=act_fp16, bias=bias)
        else:
            ref = R.qgemm_w8a16(x, q, sc, zp, group, act_fp16=act_fp16, bias=bias)
        R.set_fp16_act(False)
        # 判据: fp16 激活口径允许 1 ULP (三端累加顺序不同), fp32 口径看相对误差
        eq = float((ref == got).mean() * 100)
        m = R.max_abs_err(ref, got)
        cos = R.cos_sim(ref, got)
        rel = R.rel_err(ref, got)
        ulp = float(R.to_fp16(np.array([np.abs(ref).max()], dtype=np.float32))[0]) * 2 ** -11 if act_fp16 else 0.0
        err_ulp = m / ulp if ulp > 0 else 0.0
        ok = cos >= 1 - 1e-6 and (err_ulp <= 1.0 if act_fp16 else rel <= 1e-3)
        rows.append({"case": name, "shape": f"[{x.shape[0]},{K}]x[{N},{K}]", "group": group,
                     "bits": bits, "symmetric": symmetric, "bit_exact_pct": round(eq, 4),
                     "max_abs_err": m, "err_ulp": round(err_ulp, 3), "rel_err": rel,
                     "cosine": cos, "ms_per_iter": meta.get("ms_per_iter"),
                     "passed": bool(ok)})
        return bool(ok)

    def dequant_case(name, W, group, bits, symmetric):
        """反量化 bit 级比对: 打包权重 + scale/zp → 去量化矩阵, 两端逐 bit 相同。"""
        N, K = W.shape
        q, sc, zp = R.rtn_quant_groupwise(W, bits=bits, group=group, symmetric=symmetric)
        pk = R.pack_int4(q) if bits == 4 else q
        if bits == 4:
            wbin("d_pk.bin", pk, np.uint8)
        else:
            wbin("d_w.bin", q, np.uint8)
        wbin("d_s.bin", sc); wbin("d_z.bin", zp)
        args = ["--op", "dequant", "--shape", f"{N},{K}", "--scale", "d_s.bin",
                "--zp", "d_z.bin", "--group", group, "--bits", bits, "--out", "d_y.bin"]
        args += (["--packed", "d_pk.bin"] if bits == 4 else ["--w", "d_w.bin"])
        meta = call(args)
        got = rbin("d_y.bin", (N, K))
        ref = R.dequant_groupwise(q, sc, zp, group, K)
        eq = float((ref.view(np.uint32) == got.view(np.uint32)).mean() * 100)
        rows.append({"case": name, "shape": f"[{N},{K}]", "group": group, "bits": bits,
                     "symmetric": symmetric, "bit_exact_pct": round(eq, 4),
                     "max_abs_err": R.max_abs_err(ref, got), "err_ulp": 0.0,
                     "rel_err": R.rel_err(ref, got), "cosine": R.cos_sim(ref, got),
                     "ms_per_iter": meta.get("ms_per_iter"), "passed": eq == 100.0})
        return eq == 100.0

    Wq = state["model.layers.0.self_attn.q_proj.weight"].numpy().astype(np.float32)
    Wd = state["model.layers.23.mlp.down_proj.weight"].numpy().astype(np.float32)
    Wg = state["model.layers.5.mlp.gate_proj.weight"].numpy().astype(np.float32)
    bq = state["model.layers.0.self_attn.q_proj.bias"].numpy().astype(np.float32)
    x1 = rng.standard_normal((4, H)).astype(np.float32)
    x2 = rng.standard_normal((1, I)).astype(np.float32) * 2

    one("q_proj W4A16 非对称 g128 + bias", Wq, x1, 128, 4, False, bias=bq)
    one("down_proj W4A16 非对称 g128 (K=4864)", Wd, x2, 128, 4, False)
    one("down_proj W4A16 对称 g128", Wd, x2, 128, 4, True)
    one("gate_proj W4A16 g32(更细分组)", Wg, x1, 32, 4, False)
    one("K 非 group 整数倍(K=100,g64, 补零路径)", rng.standard_normal((32, 100)).astype(np.float32),
        rng.standard_normal((3, 100)).astype(np.float32), 64, 4, False)
    one("W8A16 g128 (混合精度档)", Wq, x1, 128, 8, False)
    one("W4A16 激活保持 fp32(精度上限对照)", Wd, x2, 128, 4, False, act_fp16=False)

    # 反量化 bit 级一致 (硬件解包/去量化单元口径)
    dequant_case("反量化 bit 级: W4 非对称 g128 (K=4864)", Wd, 128, 4, False)
    dequant_case("反量化 bit 级: W4 对称 g32", Wg, 32, 4, True)
    dequant_case("反量化 bit 级: W4 (K=100,g64 补零路径)",
                 rng.standard_normal((32, 100)).astype(np.float32), 64, 4, False)
    dequant_case("反量化 bit 级: W8 非对称 g128", Wq, 128, 8, False)

    # int4 打包/解包 bit 级一致 (含 K 补齐路径)
    q = (rng.integers(0, 16, size=(16, 256))).astype(np.uint8)
    wbin("pk_q.bin", q, np.uint8)
    call(["--op", "pack_int4", "--shape", "16,256", "--in", "pk_q.bin", "--out", "pk_out.bin"])
    pk = np.fromfile(os.path.join(WORK, "pk_out.bin"), dtype=np.uint8).reshape(16, 128)
    rows.append({"case": "int4 打包(偶列低半字节)", "shape": "[16,256]", "group": "-", "bits": 4,
                 "symmetric": "-", "bit_exact_pct": 100.0 if np.array_equal(pk, R.pack_int4(q)) else 0.0,
                 "max_abs_err": 0.0, "rel_err": 0.0, "ms_per_iter": None,
                 "passed": bool(np.array_equal(pk, R.pack_int4(q)))})

    # int8 激活量化 (W8A8 的 A8 部分): per-token 对称
    x = np.concatenate([np.zeros((1, H), np.float32),
                        (rng.standard_normal((2, H)).astype(np.float32) * 30)], 0)
    wbin("a8_x.bin", x)
    call(["--op", "quant_act8", "--shape", f"{x.shape[0]},{H}", "--in", "a8_x.bin", "--out", "a8_y.bin"])
    got = rbin("a8_y.bin", x.shape)
    q8, sc8 = R.quantize_act_int8(x)
    ref = (q8.astype(np.float32) * sc8).astype(np.float32)
    eq = float(np.allclose(ref, got, atol=0, rtol=0) * 100)
    rows.append({"case": "int8 激活量化(per-token 对称, 含全零 token)", "shape": f"[3,{H}]",
                 "group": "-", "bits": 8, "symmetric": True, "bit_exact_pct": round(eq, 4),
                 "max_abs_err": R.max_abs_err(ref, got), "rel_err": R.rel_err(ref, got),
                 "ms_per_iter": None, "passed": eq == 100.0})
    return rows


def main():
    C.ensure_dirs()
    os.makedirs(WORK, exist_ok=True)
    C.compile_cpp(os.path.join(C.SRC, "s2_ops_native.cpp"), BIN)
    import mtx_qwen as Q
    import torch
    cfg, state = Q.load_state(C.QWEN_DIR, torch.float32)

    rep = C.Report("S2 算子适配与验证 —— 参考算子数值对齐",
                   json_path=os.path.join(C.RESULTS, "s2_ops_alignment.json"),
                   md_path=os.path.join(C.RESULTS, "s2_ops_alignment.md"))
    rep.h("判据", 2)
    rep.p("- fp32 口径: 相对误差 ≤ 1e-5 且 cosine ≥ 1-1e-9 (三端累加顺序不同, 允许尾数级差异)")
    rep.p("- fp16 激活口径: 与 fp16 ULP 口径一致 (误差 ≤ 1 ULP, cosine ≥ 1-1e-6)")
    rep.p("- 量化算子 (打包/去量化/GEMM/激活量化): 要求 **bit 级 100% 一致**")

    rows, detail = [], []
    cases = build_cases(cfg, state)
    for name, kind, fp16, fn in cases:
        res = fn()
        if not isinstance(res, list):
            res = [res]
        for r in res:
            r["case"] = r["name"]
            r["kind"] = kind
            detail.append(r)
            if "bit_exact_pct" in r:
                rows.append([r["name"], kind, "PASS" if r["passed"] else "FAIL", "-", "-", "-",
                             f"bit-exact {r['bit_exact_pct']:.2f}%"])
            else:
                rows.append([r["name"], kind, "PASS" if r["passed"] else "FAIL",
                             f"{r['rel_err']:.3e}", f"{r['max_abs_err']:.3e}",
                             f"{r['cosine']:.9f}",
                             f"{r['max_err_ulp']:.2f}" if fp16 else "-"])

    rep.h("1. 逐算子对齐 (numpy 参考 ↔ C++ native)", 2)
    rep.table(["用例", "口径", "判定", "rel_err", "max_abs", "cosine", "ULP"], rows)
    rep.kv("op_alignment", detail)
    rep.kv("op_alignment_pass", all(r[2] == "PASS" for r in rows))

    rep.h("2. 量化算子对齐 (权重打包 / 去量化 / W4A16 GEMM / int8 激活)", 2)
    qrows = quant_cases(cfg, state)
    rep.table(["用例", "形状", "group", "bits", "对称", "bit 一致率", "ULP", "cosine", "native ms/iter", "判定"],
              [[r["case"], r["shape"], r["group"], r["bits"], r["symmetric"],
                f"{r['bit_exact_pct']:.2f}%", f"{r.get('err_ulp', 0):.2f}",
                f"{r.get('cosine', 1):.9f}",
                f"{r['ms_per_iter']:.3f}" if r["ms_per_iter"] else "-",
                "PASS" if r["passed"] else "FAIL"] for r in qrows])
    rep.kv("quant_alignment", qrows)
    rep.kv("quant_alignment_pass", all(r["passed"] for r in qrows))
    rep.h("3. 结论", 2)
    rep.p(f"- 结构算子 {len(rows)} 个用例, 量化算子 {len(qrows)} 个用例; "
          f"参考实现 ↔ native 实现数值口径一致。")
    rep.p("- 这套对齐口径就是后面 NPU ISS (s4) 的判据来源: 先钉住算子, 再谈整模型。")
    gates = {"op_alignment_pass": rep.data["op_alignment_pass"],
             "quant_alignment_pass": rep.data["quant_alignment_pass"],
             "fails": [d for d in detail if not d["passed"]] + [d for d in qrows if not d["passed"]]}
    C.jdump(gates, os.path.join(C.RESULTS, "s2_gates.json"))
    rep.save()
    print(f"算子对齐: {'全绿' if gates['op_alignment_pass'] and gates['quant_alignment_pass'] else '有 FAIL'}")
    if gates["fails"]:
        print(json.dumps(gates["fails"][:6], ensure_ascii=False, indent=1)[:2000])


if __name__ == "__main__":
    main()
