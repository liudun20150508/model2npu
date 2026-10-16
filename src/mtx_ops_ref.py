#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考算子库 (numpy 实现, 零框架依赖).

这里是"参考实现"的一端: 对照 PyTorch 官方实现写出来的算子, 用于
  1) 和 C++ 原生实现 (s2_ops_native.cpp) 做数值对齐;
  2) 和 NPU ISS (s4_iss.cpp) 做数值对齐。
数据排布/精度/边界条件在本文件里是唯一约定, 三端必须一致。
"""
import numpy as np

F16 = np.float16

# 是否模拟"激活以 fp16 存储"的硬件行为 (对应 s2_ops_native.cpp 的 --act-fp16)
_R = {"on": False}


def set_fp16_act(on):
    _R["on"] = bool(on)


def _r(x):
    return to_fp16(x) if _R["on"] else np.asarray(x, dtype=np.float32)


# ------------------------------------------------------------ 精度/误差工具
def to_fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def to_bf16(x):
    """bf16 舍入 = round-to-nearest-even 到 8 位尾数。"""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    return (rounded & 0xFFFF0000).astype(np.uint32).view(np.float32)


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rel_err(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) + eps
    return float(np.linalg.norm(a - b) / denom)


def max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def ulp_stats(a, b):
    """按 fp32 整数表示统计 ULP 差 (只对同号有限值有意义)."""
    a = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    d = np.abs(a - b)
    return {"max_ulp": int(d.max()), "mean_ulp": float(d.mean()), "exact_pct": float((d == 0).mean() * 100)}


def kl_div(logits_p, logits_q):
    """KL(p||q), 输入是未归一化 logits。"""
    p = logits_p.astype(np.float64)
    q = logits_q.astype(np.float64)
    p = p - p.max(axis=-1, keepdims=True)
    q = q - q.max(axis=-1, keepdims=True)
    lp = p - np.log(np.exp(p).sum(axis=-1, keepdims=True))
    lq = q - np.log(np.exp(q).sum(axis=-1, keepdims=True))
    return float((np.exp(lp) * (lp - lq)).sum(axis=-1).mean())


def top1_agreement(logits_a, logits_b):
    a = np.asarray(logits_a).argmax(axis=-1)
    b = np.asarray(logits_b).argmax(axis=-1)
    return float((a == b).mean())


# ---------------------------------------------------------------- 基础算子
def rmsnorm(x, w, eps=1e-6):
    """x:[T,H] fp32, w:[H] -> fp32 (方差用 fp32 累加, 与 RTL 定点估计一致)."""
    x = np.asarray(x, dtype=np.float32)
    v = np.mean(np.square(x), axis=-1, keepdims=True)
    return _r((x / np.sqrt(v + eps)) * w)


def layernorm(x, w, b, eps=1e-5):
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return _r((x - m) / np.sqrt(v + eps) * w + b)


def rope_tables(head_dim, positions, theta=1e6):
    """返回 cos/sin [T, head_dim] (rotate_half 约定: 前半/后半配对)."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    p = np.asarray(positions, dtype=np.float64)[:, None]
    f = p * inv[None, :]
    emb = np.concatenate([f, f], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rope_apply(x, cos, sin):
    """x:[T,Hn,D] -> 应用 RoPE; cos/sin:[T,D]."""
    x = np.asarray(x, dtype=np.float32)
    if _R["on"]:
        cos, sin = to_fp16(cos), to_fp16(sin)
    d = x.shape[-1]
    c = cos[:, None, :]
    s = sin[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return _r(x * c + rot * s)


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def gqa_repeat(kv, n_q, n_kv):
    """GQA: [T,Hkv,D] -> [T,Hq,D], 与 HF repeat_kv 一致 (每头重复 n_q/n_kv 次)."""
    kv = np.asarray(kv, dtype=np.float32)
    rep = n_q // n_kv
    return np.repeat(kv[:, :, None, :], rep, axis=2).reshape(kv.shape[0], n_q, kv.shape[2])


def attn_prefill(q, k, v, scale=None, causal=True, positions=None):
    """q:[T,Hq,D] k/v:[T,Hkv,D] -> o:[T,Hq,D]。KV 缓存场景下 k/v 含历史 token。"""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    scale = scale if scale is not None else 1.0 / np.sqrt(D)
    kk = gqa_repeat(k, Hq, Hkv)   # [T,Hq,D]
    vv = gqa_repeat(v, Hq, Hkv)
    if positions is None:
        positions = np.arange(k.shape[0])
    scores = _r(np.einsum("thd,shd->hts", q, kk) * scale)   # [Hq,T,S]
    if causal:
        m = positions[None, :] > positions[:, None]     # 查询 q[i] 只能看 pos<=p_i
        scores = np.where(m[None, :, :], -np.inf, scores)
    p = _r(softmax(scores.astype(np.float32), axis=-1))
    out = np.einsum("hts,shd->thd", p, vv)
    return _r(out), p


def attn_decode(q, k_cache, v_cache, scale=None):
    """单 token decode: q:[1,Hq,D], k/v_cache:[S,Hkv,D]."""
    return attn_prefill(q, k_cache, v_cache, scale=scale, causal=False)


def silu(x):
    x = np.asarray(x, dtype=np.float32)
    return _r(x / (1.0 + np.exp(-x)))


def silu_mul(gate, up):
    return _r(silu(gate) * up)


def gelu(x):
    """erf 版 GELU (ViT/DeiT hidden_act=gelu), 用 math.erf 逐元素算, 与 libm 参考一致。"""
    import math
    x = np.asarray(x, dtype=np.float32)
    y = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x.ravel()],
                 dtype=np.float32).reshape(x.shape)
    return _r(y)


def gelu_tanh(x):
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)


# ---------------------------------------------------------------- 量化原语
def qmax(bits):
    return (1 << bits) - 1


def rtn_quant_groupwise(w, bits=4, group=128, symmetric=False, axis=-1):
    """分组对称/非对称 round-to-nearest 量化。

    返回 (q_int [N,K] uint8, scale [N,G] fp32, zp [N,G] fp32)
    - 非对称: scale=(max-min)/(2^b-1), zp=round(-min/scale), q=clip(round(w/scale)+zp)
    - 对称  : scale=max|w|/(2^(b-1)-1),  zp=2^(b-1), q=clip(round(w/scale)+zp)
    去量化: w_hat = (q - zp) * scale
    """
    w = np.asarray(w, dtype=np.float32)
    N, K = w.shape
    G = (K + group - 1) // group
    pad = G * group - K
    wp = np.pad(w, ((0, 0), (0, pad)), mode="constant") if pad else w
    wg = wp.reshape(N, G, group)
    if symmetric:
        lim = (1 << (bits - 1)) - 1
        amax = np.abs(wg).max(axis=-1, keepdims=True)
        scale = np.where(amax == 0, 1.0, amax / lim).astype(np.float32)
        zpf = np.full_like(scale, 1 << (bits - 1), dtype=np.float32)
        q = np.clip(np.round(wg / scale) + (1 << (bits - 1)), 0, qmax(bits))
    else:
        wmin = wg.min(axis=-1, keepdims=True)
        wmax = wg.max(axis=-1, keepdims=True)
        scale = np.where(wmax == wmin, 1.0, (wmax - wmin) / qmax(bits)).astype(np.float32)
        zpf = np.round(-wmin / scale)
        q = np.clip(np.round(wg / scale) + zpf, 0, qmax(bits))
    return (q.reshape(N, G * group).astype(np.uint8),
            scale.reshape(N, G).astype(np.float32),
            zpf.reshape(N, G).astype(np.float32))


def dequant_groupwise(q, scale, zp, group, K):
    """与硬件反量化位精确一致的参考实现 (q:[N,Kp] uint8, scale/zp:[N,G])."""
    q = np.asarray(q)
    N, Kp = q.shape
    G = scale.shape[1]
    w = (q.astype(np.float32).reshape(N, G, group) - zp[:, :, None]) * scale[:, :, None]
    return w.reshape(N, Kp)[:, :K].astype(np.float32)


def pack_int4(q):
    """把 [N,Kp] uint8 (0..15) 打包成 [N,Kp/2] uint8: 低半字节=偶数列, 高半字节=奇数列。"""
    q = np.asarray(q, dtype=np.uint8)
    N, K = q.shape
    assert K % 2 == 0, "K 必须偶数 (不足时先 pad)"
    lo = q[:, 0::2]
    hi = q[:, 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(p):
    p = np.asarray(p, dtype=np.uint8)
    N, K2 = p.shape
    out = np.empty((N, K2 * 2), dtype=np.uint8)
    out[:, 0::2] = p & 0x0F
    out[:, 1::2] = (p >> 4) & 0x0F
    return out


def qgemm_w4a16(x, w_packed, scale, zp, group, symmetric=False, act_fp16=True, bias=None):
    """W4A16 模拟 GEMM: 去量化权重 * fp16 激活, fp32 累加。x:[T,K] -> [T,N]。

    这是"硬件行为"的 numpy 参考: 反量化 -> fp32 MAC, 激活按 fp16 存储。
    """
    x = np.asarray(x, dtype=np.float32)
    if act_fp16:
        x = to_fp16(x)
    q = unpack_int4(w_packed)
    w = dequant_groupwise(q, scale, zp, group, x.shape[-1])
    acc = np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)
    # 分组累加: 让 fp32 累加顺序与分组/分块数据流一致
    for g0 in range(0, x.shape[-1], group):
        acc += x[:, g0:g0 + group] @ w[:, g0:g0 + group].T
    if bias is not None:
        acc = acc + bias[None, :]
    return _r(acc)


def qgemm_w8a16(x, w8, scale, zp, group, act_fp16=True, bias=None):
    x = np.asarray(x, dtype=np.float32)
    if act_fp16:
        x = to_fp16(x)
    N, K = w8.shape
    G = scale.shape[1]
    w = ((w8.astype(np.float32).reshape(N, G, group) - zp[:, :, None]) * scale[:, :, None]).reshape(N, K)
    acc = np.zeros((x.shape[0], N), dtype=np.float32)
    for g0 in range(0, x.shape[-1], group):
        acc += x[:, g0:g0 + group] @ w[:, g0:g0 + group].T
    if bias is not None:
        acc = acc + bias[None, :]
    return _r(acc)


def quantize_act_int8(x, symmetric=True):
    """per-token 动态 int8 激活量化 (W8A8 方案里的 A8)。返回 (q, scale) — scale 为 [T,1]。"""
    x = np.asarray(x, dtype=np.float32)
    amax = np.abs(x).max(axis=-1, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale
