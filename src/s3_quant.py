#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 量化与精度优化: RTN / AWQ / GPTQ, 混合精度, 硬件权重格式导出。

对应 能力项 3:
  * 评估 FP16/BF16、INT8、INT4 方案;
  * 处理 AWQ、GPTQ 等量化方式; 混合精度 (lm_head 留 8bit) 与硬件权重格式导出;
  * 用最终 logits 和生成结果定位精度损失。

产出:
  build/quant/<variant>.pt          紧凑量化包 (打包权重 + scale/zp + AWQ 反缩放)
  results/s3_quant.json / .md       变体精度对比 (权重误差 -> 层误差 -> logits/生成结果)
  results/s3_weight_errors.csv      逐 tensor 量化误差
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_ops_ref as R
import mtx_qwen as Q

QUANT_DIR = os.path.join(C.BUILD, "quant")
PROJS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


# ------------------------------------------------------------------ 工具
def quant_tensor(W, bits=4, group=128, symmetric=False, scale_fp16=True):
    """分组量化一个 [N,K] 权重 -> 紧凑包 (与硬件反量化口径一致)。"""
    Wn = np.asarray(W, dtype=np.float32)
    q, sc, zp = R.rtn_quant_groupwise(Wn, bits=bits, group=group, symmetric=symmetric)
    if scale_fp16:                       # 硬件把 scale / zero-point 存成 fp16
        sc = np.asarray(sc, dtype=np.float16).astype(np.float32)
        zp = np.asarray(np.round(zp), dtype=np.float16).astype(np.float32)
    Kp = q.shape[1]
    packed = R.pack_int4(q) if bits == 4 else q
    return {"N": Wn.shape[0], "K": Wn.shape[1], "Kp": Kp, "bits": bits, "group": group,
            "symmetric": symmetric, "packed": packed,
            "scale": sc.astype(np.float16), "zp": zp.astype(np.float32)}


def dequant_tensor(pk):
    """与硬件一致的反量化: 解包 -> (q - zp) * scale。"""
    q = R.unpack_int4(pk["packed"]) if pk["bits"] == 4 else pk["packed"]
    return R.dequant_groupwise(q, pk["scale"].astype(np.float32), pk["zp"], pk["group"], pk["K"])


def pack_size(pk):
    return pk["packed"].nbytes + pk["scale"].nbytes + pk["zp"].nbytes


# ------------------------------------------------------- 标定: 激活统计
def collect_calib(model, tok, texts, hess_names, sample_rows=48):
    """跑标定集, 统计每个 linear 的输入:
      * 每通道 |x| 均值 (AWQ 要用)
      * X^T X (GPTQ 要用, 只对大矩阵目标累积, 省内存)
      * 若干行样本激活 (评估量化后层输出误差)
    """
    stats = {}

    def hook(name, x):
        x = x.detach().float()
        st = stats.setdefault(name, {"abs_sum": None, "rows": 0, "H": None,
                                     "samples": [], "K": x.shape[-1]})
        a = x.abs().sum(0)
        st["abs_sum"] = a if st["abs_sum"] is None else st["abs_sum"] + a
        st["rows"] += x.shape[0]
        if name in hess_names:
            if st["H"] is None:
                st["H"] = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float32)
            st["H"] += x.t() @ x
        if sum(s.shape[0] for s in st["samples"]) < sample_rows:
            st["samples"].append(x[: sample_rows])

    model.act_hook = hook
    for t in texts:
        ids = tok.encode(t).ids[:256]
        if len(ids) < 2:
            continue
        model.forward(input_ids=ids)
    model.act_hook = None
    for name, st in stats.items():
        st["abs_mean"] = (st["abs_sum"] / max(1, st["rows"])).numpy().astype(np.float32)
        st["samples"] = torch.cat([s for s in st["samples"] if s.numel()], 0)[:sample_rows].contiguous()
        del st["abs_sum"]
    return stats


# ------------------------------------------------------------ AWQ 缩放搜索
def awq_scale(W, X, alpha, group=128, bits=4, symmetric=False):
    """AWQ 等变缩放: s = (act_scale)^a / (w_scale)^(1-a), 几何均值归一。

    返回 (每通道反缩放 inv=1/s, 缩放后权重的量化包, s)
    """
    act = np.abs(np.asarray(X, dtype=np.float32)).mean(0) + 1e-6
    wei = np.abs(np.asarray(W, dtype=np.float32)).mean(0) + 1e-6
    s = (act ** alpha) / (wei ** (1 - alpha))
    s = s / np.exp(np.mean(np.log(s)))              # 几何均值归一, 保持整体尺度
    Ws = (np.asarray(W, dtype=np.float32) * s[None, :]).astype(np.float32)
    pk = quant_tensor(Ws, bits=bits, group=group, symmetric=symmetric)
    return (1.0 / s).astype(np.float32), pk, s.astype(np.float32)


def layer_out_err(W, Wq, X, inv=None):
    """层输出相对误差 ||X(W-Wq)^T|| / ||XW^T|| — GPTQ/AWQ 的目标函数。"""
    X = np.asarray(X, dtype=np.float32)
    if inv is not None:
        X = X * inv[None, :]
    return R.rel_err(X @ W.astype(np.float32).T, X @ Wq.T)


# ------------------------------------------------------------------ GPTQ
def gptq_quantize(W, H, bits=4, group=128, damp=0.01, symmetric=False):
    """按列量化 + 误差补偿 (H = X^T X 的二阶信息)。

    与文献实现同构: e = (w - q)/Hinv[j,j], W[:, j+1:] -= e (x) Hinv[j, j+1:]
    """
    W = np.array(W, dtype=np.float32, copy=True)
    N, K = W.shape
    H = np.array(H, dtype=np.float32, copy=True)
    dg = np.diag(H).copy()
    H[np.diag_indices(K)] += damp * (dg.mean() if dg.mean() > 0 else 1.0)
    L = np.linalg.cholesky(H)
    Hinv = np.linalg.inv(L).T @ np.linalg.inv(L)                       # = H^-1
    Q = np.zeros((N, K), dtype=np.uint8)
    G = (K + group - 1) // group
    scales = np.zeros((N, G), dtype=np.float32)
    zps = np.zeros((N, G), dtype=np.float32)
    qmaxv = (1 << bits) - 1
    for g in range(G):
        g0, g1 = g * group, min(K, (g + 1) * group)
        blk = W[:, g0:g1]
        if symmetric:
            lim = (1 << (bits - 1)) - 1
            sc = np.abs(blk).max(1, keepdims=True)
            sc = np.where(sc == 0, 1.0, sc / lim).astype(np.float32)
            zp = np.full_like(sc, 1 << (bits - 1))
        else:
            mn, mx = blk.min(1, keepdims=True), blk.max(1, keepdims=True)
            sc = np.where(mx == mn, 1.0, (mx - mn) / qmaxv).astype(np.float32)
            zp = np.round(-mn / sc)
        for j in range(g0, g1):
            w = W[:, j]
            q = np.clip(np.round(w / sc[:, 0]) + zp[:, 0], 0, qmaxv)
            deq = (q - zp[:, 0]) * sc[:, 0]
            Q[:, j] = q.astype(np.uint8)
            if j + 1 < K:
                e = (w - deq) / Hinv[j, j]
                W[:, j + 1:] -= np.outer(e, Hinv[j, j + 1:])
        scales[:, g] = sc[:, 0]
        zps[:, g] = zp[:, 0]
    return {"N": N, "K": K, "Kp": G * group, "bits": bits, "group": group,
            "symmetric": symmetric, "packed": R.pack_int4(Q),
            "scale": scales.astype(np.float16), "zp": zps.astype(np.float32)}


# ---------------------------------------------------------------- 变体构建
def tensor_names(cfg, include_embed=True):
    names = [f"model.layers.{i}.{p}.weight" for i in range(cfg.layers) for p in PROJS]
    if include_embed:
        names.append("model.embed_tokens.weight")
    return names


def build_variant(cfg, state, mode, calib=None, gptq_names=(), rep_rows=None):
    """按 mode 逐 tensor 量化, 返回 {tensor: 紧凑包}。"""
    pk_all, info = {}, {}
    for key in tensor_names(cfg):
        W = state[key].numpy().astype(np.float32)
        is_embed = key == "model.embed_tokens.weight"
        if mode == "fp16" or (is_embed and not mode.endswith("mix8") and mode != "w8"):
            pk_all[key] = {"fp16": True, "W16": W.astype(np.float16)}
            continue
        bits = 8 if (mode == "w8" or is_embed) else 4
        X = calib[key]["samples"].numpy() if (calib and key in calib) else None
        if mode.startswith("gptq") and not is_embed and calib is not None and \
                calib.get(key, {}).get("H") is not None and key in set(gptq_names):
            pk = gptq_quantize(W, calib[key]["H"].numpy(), bits=bits, group=128)
            method = "gptq"
        elif mode.startswith("awq") and not is_embed and X is not None:
            best = None
            for a in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75):
                inv, pk, s = awq_scale(W, X, a, bits=bits)
                err = layer_out_err(W, dequant_tensor(pk) * s[None, :], X)
                if best is None or err < best[0]:
                    best = (err, inv, pk, s)
            err, inv, pk, s = best
            pk["inv_scale"] = inv.astype(np.float16)
            pk["awq_err"] = float(err)
            method = "awq"
        else:
            pk = quant_tensor(W, bits=bits, group=128)
            method = "w8" if bits == 8 else "rtn"
        pk_all[key] = pk
        if rep_rows is not None:
            Wq = dequant_tensor(pk)
            if "inv_scale" in pk:
                Wq_eff = Wq * pk["inv_scale"].astype(np.float32)[None, :]
                werr = R.rel_err(W, Wq_eff)
            else:
                Wq_eff, werr = Wq, R.rel_err(W, Wq)
            row = {"tensor": key, "method": method, "group": pk["group"], "bits": pk["bits"],
                   "weight_rel_err": werr}
            if X is not None and not is_embed:
                row["layer_rel_err"] = layer_out_err(W, Wq_eff, X)
            rep_rows.append(row)
    return pk_all, info


def variant_state(pk_all, state_fp32):
    """紧凑包 -> fp32 state dict (模拟硬件去量化后的权重)。"""
    out = {}
    for k, pk in pk_all.items():
        if pk.get("fp16"):
            out[k] = torch.from_numpy(pk["W16"].astype(np.float32))
        else:
            Wq = dequant_tensor(pk)
            if "inv_scale" in pk:
                Wq = Wq * pk["inv_scale"].astype(np.float32)[None, :]
            out[k] = torch.from_numpy(Wq)
    for k, v in state_fp32.items():
        if k not in out:
            out[k] = v
    return out


def total_bytes(pk_all):
    tot = 0
    for pk in pk_all.values():
        tot += pk["W16"].nbytes if pk.get("fp16") else pack_size(pk)
        if "inv_scale" in pk:
            tot += pk["inv_scale"].nbytes
    return tot


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gptq", default="sample", choices=["sample", "all", "none"])
    ap.add_argument("--fast", action="store_true", help="跳过 AWQ/GPTQ (只跑 RTN)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true",
                    help="复用 build/quant 里已有的量化包, 只重跑精度评估 (秒级)")
    args = ap.parse_args()
    C.ensure_dirs()
    os.makedirs(QUANT_DIR, exist_ok=True)

    from tokenizers import Tokenizer
    data = C.jload(os.path.join(C.SRC, "data", "calib.json"))
    tok = Tokenizer.from_file(os.path.join(C.QWEN_DIR, "tokenizer.json"))
    cfg, st = Q.load_state(C.QWEN_DIR, torch.float32)
    golden = Q.Qwen2Ref(cfg, st, act_dtype="fp32", name="golden")

    rep = C.Report("S3 量化与精度优化 —— RTN / AWQ / GPTQ + 混合精度 + 硬件权重格式",
                   json_path=os.path.join(C.RESULTS, "s3_quant.json"),
                   md_path=os.path.join(C.RESULTS, "s3_quant.md"))
    rep.h("1. 标定集与激活统计", 2)
    # GPTQ 的覆盖范围:
    #   none   -> 不跑 GPTQ (只 RTN/AWQ)
    #   sample -> 全部 attention 投影 + 首尾层的 MLP (110 个矩阵, 分钟级)
    #   all    -> 全部 168 个矩阵 (全量, 半小时级)
    attn = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    mlp = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    gptq_names = set()
    if args.gptq == "all":
        gptq_names = {f"model.layers.{i}.{p}.weight" for i in range(cfg.layers)
                      for p in (attn + mlp)}
    elif args.gptq == "sample" and not args.fast:
        gptq_names = {f"model.layers.{i}.{p}.weight" for i in range(cfg.layers) for p in attn}
        gptq_names |= {f"model.layers.{i}.{p}.weight" for i in (0, cfg.layers - 1) for p in mlp}
    hess = set(gptq_names)
    calib = None
    if args.eval_only:
        gptq_names = set()
        # 复用上一次构建留下的逐 tensor 统计 (GPTQ/AWQ 的收益结论), 避免重跑 10 分钟
        for prev in ("s3_quant.json", "s3_quant_gptqall.json"):
            p = os.path.join(C.RESULTS, prev)
            if not os.path.exists(p):
                continue
            old = C.jload(p)
            for k in ("gptq_gain", "awq_gain", "awq_rows", "per_tensor", "calib"):
                if k in old and k not in rep.data:
                    rep.data[k] = old[k]
    else:
        t0 = time.time()
        calib = collect_calib(golden, tok, data["calib"], hess)
        rep.p(f"标定样本 {len(data['calib'])} 条 (每条 <=256 token), 覆盖 {len(calib)} 个 linear 的输入激活; "
              f"其中 {len(hess)} 个矩阵累积了 Hessian H=X^T X。耗时 {time.time() - t0:.1f} s。")
        rep.kv("calib", {"texts": len(data["calib"]), "linears": len(calib), "hessians": len(hess),
                         "sample_rows": int(next(iter(calib.values()))["samples"].shape[0])})

    n_gptq = len(gptq_names)
    variants = [("fp16", "fp16", "全 fp16 权重 + fp16 激活 (硬件 fp16 档)"),
                ("w8a16_g128", "w8", "W8A16 g128 (精度上限对照)"),
                ("w4a16_rtn_g128", "rtn", "W4A16 RTN 非对称 g128, lm_head 留 fp16"),
                ("w4a16_rtn_mix8", "rtn_mix8", "W4A16 + lm_head int8 (混合精度)"),
                ("w4a16_awq_mix8", "awq_mix8", "W4A16 AWQ 缩放 + lm_head int8"),
                ("w4a16_gptq_mix8", "gptq_mix8",
                 f"W4A16 GPTQ ({n_gptq}/168 个矩阵) + lm_head int8")]
    if args.fast:
        variants = [v for v in variants if "awq" not in v[0] and "gptq" not in v[0]]

    rows_csv, packs, meta = [], {}, {}
    for vname, mode, desc in variants:
        p = os.path.join(QUANT_DIR, f"{vname}.pt")
        t0 = time.time()
        if args.eval_only:
            if not os.path.exists(p):
                continue
            pk_all = torch.load(p, map_location="cpu", weights_only=False)
            rep_rows = None
        else:
            pk_all, info = build_variant(cfg, st, mode, calib=calib,
                                         gptq_names=gptq_names, rep_rows=rows_csv)
            torch.save(pk_all, p)
        packs[vname] = pk_all
        meta[vname] = {"desc": desc, "mode": mode, "bytes": total_bytes(pk_all),
                       "build_s": round(time.time() - t0, 1)}

    if not args.eval_only:
      rep.h("2. 逐 tensor 量化误差 (权重 / 层输出)", 2)
      methods = {}
      for r in rows_csv:
        methods.setdefault(r["method"], []).append(r)
      rows = []
      for m, rs in methods.items():
        le = [x["layer_rel_err"] for x in rs if "layer_rel_err" in x]
        rows.append([m, len(rs), f"{np.median([x['weight_rel_err'] for x in rs]) * 100:.3f}%",
                     f"{np.median(le) * 100:.3f}%" if le else "-",
                     f"{min(le) * 100:.2f}% / {max(le) * 100:.2f}%" if le else "-"])
      rep.p("目标函数是层输出相对误差 ||X(W-Wq)^T|| / ||XW^T|| (正是 GPTQ 的优化目标):")
      rep.table(["方法", "tensor 数", "权重相对误差(中位)", "层输出相对误差(中位)", "最好/最差"], rows)
      rep.kv("per_tensor", rows_csv)
      with open(os.path.join(C.RESULTS, "s3_weight_errors.csv"), "w", encoding="utf-8") as f:
        keys = ["tensor", "method", "group", "bits", "weight_rel_err", "layer_rel_err"]
        f.write(",".join(keys) + "\n")
        for r in rows_csv:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

      gq = {r["tensor"]: r for r in rows_csv if r["method"] == "gptq"}
      rtn = {r["tensor"]: r for r in rows_csv if r["method"] == "rtn"}
      cmp_rows = []
      for k in gq:
        if k in rtn and "layer_rel_err" in gq[k] and "layer_rel_err" in rtn[k]:
            cmp_rows.append([k.replace("model.layers.", "L").replace(".weight", ""),
                             f"{rtn[k]['layer_rel_err'] * 100:.3f}%",
                             f"{gq[k]['layer_rel_err'] * 100:.3f}%",
                             f"{gq[k]['layer_rel_err'] / rtn[k]['layer_rel_err']:.3f}"])
      awq_rows = []
      for r in rows_csv:
        if r["method"] == "awq" and r["tensor"] in rtn and "layer_rel_err" in r:
            awq_rows.append([r["tensor"].replace("model.layers.", "L").replace(".weight", ""),
                             f"{rtn[r['tensor']]['layer_rel_err'] * 100:.3f}%",
                             f"{r['layer_rel_err'] * 100:.3f}%",
                             f"{r['layer_rel_err'] / rtn[r['tensor']]['layer_rel_err']:.3f}"])
      if cmp_rows:
        rep.p("GPTQ 相对同 group_size RTN 的收益 (目标函数 = 层输出相对误差):")
        rep.table(["tensor", "RTN g128", "GPTQ g128", "误差比"], cmp_rows)
        ratios = [float(r[3]) for r in cmp_rows]
        rep.kv("gptq_gain", {"n": len(ratios), "median_ratio": float(np.median(ratios)),
                             "best": float(min(ratios)), "worst": float(max(ratios)),
                             "all_better": bool(all(x < 1 for x in ratios))})
      if awq_rows:
        ar = [float(r[3]) for r in awq_rows]
        rep.kv("awq_gain", {"n": len(ar), "median_ratio": float(np.median(ar)),
                            "best": float(min(ar)), "worst": float(max(ar)),
                            "better_pct": float(np.mean([x < 1 for x in ar]) * 100)})
        rep.p(f"AWQ 缩放相对 RTN 的收益 (缩放系数按激活幅度搜索出, 随权重一起导出): "
              f"{len(ar)} 个 tensor, 中位误差比 {np.median(ar):.3f}, "
              f"{(np.array(ar) < 1).mean() * 100:.0f}% 的 tensor 更优。")
        show = sorted(awq_rows, key=lambda r: float(r[3]))[:6] + \
            sorted(awq_rows, key=lambda r: float(r[3]))[-3:]
        rep.table(["tensor", "RTN g128", "AWQ g128", "误差比"], show)
        rep.kv("awq_rows", awq_rows)
    else:
        rep.h("2. 逐 tensor 量化误差 (复用上一次构建的量化包与统计)", 2)
        g = rep.data.get("gptq_gain")
        if g:
            rep.p(f"GPTQ: {g['n']} 个矩阵, 层输出误差中位比 **{g['median_ratio']:.3f}** "
                  f"(最好 {g['best']:.3f}, 最差 {g['worst']:.3f}, 全部优于 RTN: {g['all_better']})")
        a = rep.data.get("awq_gain")
        if a:
            rep.p(f"AWQ: {a['n']} 个 tensor, 中位误差比 **{a['median_ratio']:.3f}**, "
                  f"更优占比 {a['better_pct']:.1f}%")
        if not (g or a):
            rep.p("(本次为 `--eval-only`, 且没有找到上一次的统计, 只重跑了第 3 节) ")

    rep.h("3. 端到端精度: logits 与生成结果对比", 2)
    prompt = data["prompts"][0]
    ids = tok.encode(prompt).ids
    g_logits, _, _ = golden.forward(input_ids=ids)
    golden_tokens, _, _ = golden.generate(ids, max_new_tokens=args.max_new)
    eval_ids = tok.encode(data["eval"][0]).ids[:192]
    eval_ids2 = tok.encode(data["eval"][1]).ids[:192] if len(data["eval"]) > 1 else eval_ids
    g_eval = golden.forward(input_ids=eval_ids)[0]
    g_eval2 = golden.forward(input_ids=eval_ids2)[0]

    def nll(logits, ids_):
        lg = torch.log_softmax(logits[:-1], -1)
        return float(-lg[range(len(ids_) - 1), torch.tensor(ids_[1:])].mean())

    g_nll = nll(g_eval, eval_ids)
    rows, detail = [], {}
    for vname, mode, desc in variants:
        st_v = variant_state(packs[vname], st)
        m = Q.Qwen2Ref(cfg, st_v, act_dtype="fp16")
        lg, _, _ = m.forward(input_ids=ids)
        toks, _, _ = m.generate(ids, max_new_tokens=args.max_new)
        ev = m.forward(input_ids=eval_ids)[0]
        ev2 = m.forward(input_ids=eval_ids2)[0]
        same = sum(1 for a, b in zip(golden_tokens, toks) if a == b) / len(golden_tokens)
        d = {"variant": vname, "desc": desc, "bytes": meta[vname]["bytes"],
             "bytes_mib": round(meta[vname]["bytes"] / 2 ** 20, 1),
             "compress_ratio": round((sum(v.numel() for v in st.values()) * 2) / meta[vname]["bytes"], 2),
             "logits_cosine": round(R.cos_sim(g_logits[-1].numpy(), lg[-1].numpy()), 6),
             "logits_kl": round(R.kl_div(g_logits[-1].numpy(), lg[-1].numpy()), 6),
             "top1_agreement": round(R.top1_agreement(g_logits.numpy(), lg.numpy()), 4),
             "gen_token_match": round(same, 4), "gen_first_token_match": int(toks[0] == golden_tokens[0]),
             "nll": round(nll(ev, eval_ids), 4), "nll_delta": round(nll(ev, eval_ids) - g_nll, 5),
             "text": tok.decode(toks)}
        # 语料级判据 (192×2 个 token 位置, 比 prompt 内 7 个位置稳得多)
        d["corpus_top1_agreement"] = round(
            (R.top1_agreement(g_eval.numpy(), ev.numpy()) + R.top1_agreement(g_eval2.numpy(), ev2.numpy())) / 2, 4)
        d["corpus_logits_cosine"] = round(
            (R.cos_sim(g_eval.numpy(), ev.numpy()) + R.cos_sim(g_eval2.numpy(), ev2.numpy())) / 2, 6)
        detail[vname] = d
        rows.append([vname, f"{d['bytes_mib']:.1f} MiB", f"{d['compress_ratio']:.2f}x",
                     f"{d['logits_cosine']:.6f}", f"{d['corpus_logits_cosine']:.6f}",
                     f"{d['top1_agreement'] * 100:.0f}% / {d['corpus_top1_agreement'] * 100:.1f}%",
                     f"{d['gen_token_match'] * 100:.0f}%",
                     f"{d['nll']:.4f}", f"{d['nll_delta']:+.4f}"])
    rep.p(f"golden = fp32 参考 (参考 NLL {g_nll:.4f}); 判据是最终 logits 与生成结果, 不是单层误差。"
          f"「top-1 一致率」两栏分别是: prompt 内 7 个位置 / 语料 384 个位置 —— "
          f"位置太少时这个指标会被单个 token 放大, 所以**看语料级那一栏**。")
    rep.table(["变体", "设备侧权重", "压缩比", "prompt logits cos", "语料 logits cos",
               "top-1 (prompt/语料)", "16 token 输出一致率", "NLL", "NLL 差"], rows)
    rep.kv("variants", detail)
    rep.kv("golden_nll", round(g_nll, 4))
    rep.kv("golden_text", tok.decode(golden_tokens))

    rep.h("4. 生成的文本 (直观对照)", 2)
    rep.p(f"prompt: `{prompt}`")
    rep.table(["变体", "生成结果"],
              [["golden(fp32)", tok.decode(golden_tokens)]] + [[v, detail[v]["text"]] for v in detail])
    rep.h("5. 结论", 2)
    rep.p("- FP16 是精度基准; INT8 权重留作 lm_head 的混合精度档; INT4 是端侧主档。")
    if "gptq_gain" in rep.data:
        g = rep.data["gptq_gain"]
        rep.p(f"- GPTQ 在同 group 下把层输出误差降到 RTN 的 {g['median_ratio']:.3f} 倍 "
              f"({g['n']} 个矩阵全部更优: {g['all_better']})。")
    if "awq_gain" in rep.data:
        g = rep.data["awq_gain"]
        rep.p(f"- AWQ 缩放同样有效, 中位误差比 {g['median_ratio']:.3f}; "
              f"缩放系数作为逐通道反缩放导出到硬件权重格式, 由设备侧 VSCALE 施加。")
    rep.save()
    print("变体:", ", ".join(f"{k}({v['bytes_mib']}MiB,cos={v['logits_cosine']})"
                             for k, v in detail.items()))


if __name__ == "__main__":
    main()
