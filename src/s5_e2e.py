#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5 端到端: 参考输出 -> 量化权重 -> NPU 运行结果, 三端对比。

对应 能力项 4 (FPGA/ASIC 端到端集成) 与工程目标:
  * 设备镜像加载 (Host->Device, PCIe 计数);
  * prefill + 连续 decode, KV Cache 在设备内存里增长;
  * 逐层 hidden state 与每一步 logits 回传, 和 PyTorch 参考逐层对齐;
  * 生成结果对比 (贪心解码是否完全一致)。

产出: results/s5_e2e.json / .md, results/npu_logits_*.bin, build/npu_dumps/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_ops_ref as R
import mtx_qwen as Q
import s3_quant as S3


def read_f32(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a16_rtn_mix8")
    ap.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--kv-dtype", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--skip-iss", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()
    dumps = os.path.join(C.BUILD, "npu_dumps")
    os.makedirs(dumps, exist_ok=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(C.QWEN_DIR, "tokenizer.json"))
    cfg, st = Q.load_state(C.QWEN_DIR, torch.float32)
    ids = tok.encode(args.prompt).ids
    P = len(ids)
    n_layers = cfg.layers if args.layers == "all" else len(args.layers.split(","))

    rep = C.Report(f"S5 端到端集成与数值对齐 ({args.variant}, prefill={P}, decode={args.decode})",
                   json_path=os.path.join(C.RESULTS, "s5_e2e.json"),
                   md_path=os.path.join(C.RESULTS, "s5_e2e.md"))
    rep.h("1. 三条链路", 2)
    rep.table(["链路", "权重", "激活", "执行位置", "用途"],
              [["A 参考 (golden)", "fp32", "fp32", "PyTorch CPU", "精度基准"],
               ["B 量化模拟", f"{args.variant} (去量化后 fp32)", "fp16", "PyTorch CPU", "量化误差上界"],
               ["C NPU ISS", f"{args.variant} 设备镜像 (int4/int8 + scale/zp)", "fp16", "MTX-NPU 指令级仿真",
                "硬件运行结果 + 性能计数"]])
    rep.p(f"prompt: `{args.prompt}` -> {P} token; 生成 {args.decode} 个 token (贪心)。")
    if args.layers != "all":
        rep.p(f"> 注意: 本次只跑了 layers={args.layers} (`--layers`), 所以**第 3~5 节的"
              f"整模型 logits / 生成结果对比不代表整模型**; 逐层对齐 (第 3 节) 仍然有效。")

    # ---------------- 链路 A: fp32 参考
    t0 = time.time()
    golden = Q.Qwen2Ref(cfg, st, act_dtype="fp32", name="golden")
    cap = {"layers": [None] * cfg.layers}
    g_logits, _, cap = golden.forward(input_ids=ids, capture=cap)
    g_tokens, _, _ = golden.generate(ids, max_new_tokens=args.decode)
    t_golden = time.time() - t0

    # ---------------- 链路 B: 量化模拟 (去量化权重 + fp16 激活)
    pk_path = os.path.join(C.BUILD, "quant", f"{args.variant}.pt")
    if not os.path.exists(pk_path):
        raise SystemExit(f"缺 {pk_path}, 先跑 s3_quant.py")
    pk_all = torch.load(pk_path, map_location="cpu", weights_only=False)
    st_q = S3.variant_state(pk_all, st)
    t0 = time.time()
    qmodel = Q.Qwen2Ref(cfg, st_q, act_dtype="fp16", name="quant_sim")
    qcap = {"layers": [None] * cfg.layers}
    q_logits, _, qcap = qmodel.forward(input_ids=ids, capture=qcap)
    # 逐步 decode: 收每一步的 logits 与第一步的逐层 capture (与 ISS 逐步对齐用)
    q_step_logits, q_step_caps, q_tokens = [q_logits[-1].numpy()], [], []
    lg, past, _ = qmodel.forward(input_ids=ids)
    for i in range(args.decode):
        nxt = int(lg[-1].argmax())
        q_tokens.append(nxt)
        cap_i = {"layers": [None] * cfg.layers}
        lg, past, cap_i = qmodel.forward(input_ids=[nxt], past=past, positions=[P + i],
                                         capture=cap_i)
        q_step_logits.append(lg[-1].numpy())
        q_step_caps.append(cap_i)
    t_quant = time.time() - t0

    # 参考链路 A 的逐步 logits (同一口径, 用于逐 token 对比)
    g_step_logits = [g_logits[-1].numpy()]
    lgg, pastg, _ = golden.forward(input_ids=ids)
    for i in range(args.decode):
        nt = int(lgg[-1].argmax())
        lgg, pastg, _ = golden.forward(input_ids=[nt], past=pastg, positions=[P + i])
        g_step_logits.append(lgg[-1].numpy())

    # ---------------- 链路 C: NPU ISS
    iss_ok, npu_detail = True, {}
    if not args.skip_iss:
        image = os.path.join(C.BUILD, "npu_device_image")
        tag = f"{args.variant}"
        img_bin = os.path.join(C.BUILD, f"image_{tag}.bin")
        if not os.path.exists(image + ".idx") or args.variant not in C.jload(image + ".json")["variant"]:
            C.run([sys.executable, os.path.join(C.SRC, "s3_pack.py"), "--variant", args.variant,
                   "--out", image])
        tokfile = os.path.join(C.BUILD, "npu_tokens.bin")
        np.asarray(ids, dtype=np.int32).tofile(tokfile)
        prog = os.path.join(C.BUILD, f"npu_program_{tag}_kv{args.kv_dtype}.mtx")
        C.run([sys.executable, os.path.join(C.SRC, "s4_gen_program.py"), "--variant", args.variant,
               "--prompt-len", str(P), "--decode", str(args.decode), "--layers", args.layers,
               "--kv-dtype", args.kv_dtype, "--dump-layers", dumps, "--out", prog])
        prof_path = os.path.join(C.RESULTS, f"npu_profile_{tag}.json")
        rc, out = C.run([os.path.join(C.BUILD, "npu_iss"), "--program", prog,
                         "--image", image + ".bin", "--idx", image + ".idx",
                         "--tokens", tokfile, "--out", prof_path,
                         "--hw", os.path.join(C.HW, "mtx_npu_v0.json")], check=False)
        if rc != 0:
            iss_ok = False
            rep.p(f"**ISS 执行失败**: `{out[-500:]}`")
        else:
            prof = C.jload(prof_path)
            npu_detail["profile"] = prof
            # 逐层 hidden state 对齐
            lay_rows = []
            for i in range(n_layers):
                f = os.path.join(dumps, f"L{i}_prefill.bin")
                if not os.path.exists(f):
                    continue
                n = read_f32(f, (P, cfg.hidden))
                ref_q = qcap["layers"][i]["out"].numpy()
                ref_g = cap["layers"][i]["out"].numpy()
                lay_rows.append([f"L{i}", f"{R.cos_sim(ref_g, n):.6f}",
                                 f"{R.rel_err(ref_g, n) * 100:.3f}%",
                                 f"{R.cos_sim(ref_q, n):.6f}", f"{R.rel_err(ref_q, n) * 100:.3f}%"])
            # 每步 logits
            step_rows, npu_tokens = [], []
            for s in range(args.decode + 1):
                f = os.path.join(C.RESULTS, "npu_logits_prefill.bin" if s == 0
                                 else f"npu_logits_step{s}.bin")
                if not os.path.exists(f):
                    continue
                lg = read_f32(f, (P if s == 0 else 1, cfg.vocab))
                ref_q, ref_g = q_step_logits[s], g_step_logits[s]
                npu_tokens.append(int(lg[-1].argmax()))
                npu_detail.setdefault("step_metrics", []).append(
                    {"phase": "prefill" if s == 0 else f"decode{s}",
                     "cos_b": R.cos_sim(lg[-1], ref_q), "cos_a": R.cos_sim(lg[-1], ref_g),
                     "max_abs_b": R.max_abs_err(lg[-1], ref_q),
                     "argmax_match": bool(int(lg[-1].argmax()) == int(ref_q.argmax()))})
                step_rows.append([f"{'prefill' if s == 0 else 'decode ' + str(s)}",
                                  f"{R.cos_sim(lg[-1], ref_q):.6f}",
                                  f"{R.cos_sim(lg[-1], ref_g):.6f}",
                                  f"{R.max_abs_err(lg[-1], ref_q):.4f}",
                                  "是" if int(lg[-1].argmax()) == int(ref_q.argmax()) else "否"])
            npu_detail["layer_rows"] = lay_rows
            # decode 第一步的逐层对齐 (定位 decode 与 prefill 的差异来源)
            dec_rows = []
            if os.path.exists(os.path.join(dumps, "L0_decode1.bin")):
                cap1 = q_step_caps[0]
                for i in range(n_layers):
                    f = os.path.join(dumps, f"L{i}_decode1.bin")
                    if not os.path.exists(f):
                        continue
                    v = read_f32(f, (1, cfg.hidden))
                    r = cap1["layers"][i]["out"].numpy()
                    dec_rows.append([f"L{i}", f"{R.cos_sim(r, v):.6f}", f"{R.rel_err(r, v) * 100:.3f}%"])
                npu_detail["decode_layer_rows"] = dec_rows
            npu_detail["step_rows"] = step_rows
            npu_detail["npu_tokens"] = npu_tokens

    # ---------------- 对比与判据
    rep.h("2. 链路 A↔B: 量化带来的精度损失 (同一 tokenizer, 同一 prompt)", 2)
    rep.table(["指标", "A 参考 fp32", "B 量化模拟", "差距"],
              [["logits cosine", "1.000000", f"{R.cos_sim(g_logits[-1].numpy(), q_logits[-1].numpy()):.6f}",
                f"{1 - R.cos_sim(g_logits[-1].numpy(), q_logits[-1].numpy()):.2e}"],
               ["KL(fp||q)", "0", f"{R.kl_div(g_logits[-1].numpy(), q_logits[-1].numpy()):.5f}", "-"],
               ["top-1 一致率", "100%", f"{R.top1_agreement(g_logits.numpy(), q_logits.numpy()) * 100:.1f}%", "-"],
               ["生成 token 一致", f"{len(g_tokens)}", f"{sum(1 for a, b in zip(g_tokens, q_tokens) if a == b)}/{len(g_tokens)}", "-"]])
    rep.p(f"生成文本 A: `{tok.decode(g_tokens)}`")
    rep.p(f"生成文本 B: `{tok.decode(q_tokens)}`")

    if iss_ok and npu_detail.get("layer_rows"):
        rep.h("3. 链路 B↔C: 逐层 hidden state 对齐 (prefill 最后一次 token)", 2)
        rep.p("这张表就是「算子/层/系统链路定位问题」的依据: 哪一层开始发散, 一眼能看出来。")
        rep.table(["层", "cos vs A(参考)", "rel_err vs A", "cos vs B(量化模拟)", "rel_err vs B"],
                  npu_detail["layer_rows"])
        if npu_detail.get("decode_layer_rows"):
            rep.h("3b. 同一张表, 换到 decode 第一步 (每层 1 个 token)", 3)
            rep.table(["层", "cos vs B(量化模拟)", "rel_err vs B"], npu_detail["decode_layer_rows"])
        rep.h("4. 链路 A/B↔C: 每一步 logits 对齐 (prefill + 连续 decode)", 2)
        rep.table(["阶段", "cos vs B", "cos vs A", "max_abs vs B", "argmax 一致"], npu_detail["step_rows"])
        rep.h("5. 生成结果三端对比", 2)
        npu_text = tok.decode(npu_detail["npu_tokens"])
        rep.table(["链路", "生成 token", "文本"],
                  [["A 参考 fp32", str(g_tokens), tok.decode(g_tokens)],
                   ["B 量化模拟", str(q_tokens), tok.decode(q_tokens)],
                   ["C NPU ISS", str(npu_detail["npu_tokens"]), npu_text]])
        same_bc = sum(1 for a, b in zip(q_tokens, npu_detail["npu_tokens"]) if a == b)
        rep.kv("gen_match_bc", f"{same_bc}/{len(q_tokens)}")
        rep.kv("npu_text", npu_text)
        rep.p(f"B 与 C 的生成 token 一致: {same_bc}/{len(q_tokens)} "
              f"(不一致的只可能是数值差把 logits 挤过了 argmax 边界, 报告里保留原文对照)")
        if iss_ok:
            prof = npu_detail["profile"]
            rep.kv("iss_summary", prof["summary"])
            rep.p(f"ISS 计数: {prof['summary']['cycles'] / 1e6:.2f} Mcycle "
                  f"({prof['summary']['ms']:.2f} ms @1GHz), DRAM {C.human_bytes(prof['summary']['dram_bytes'])}, "
                  f"PCIe {C.human_bytes(prof['summary']['pcie_bytes'])}, "
                  f"{prof['summary']['n_ops']} 条指令。")
    rep.kv("variant", args.variant)
    rep.kv("step_metrics", npu_detail.get("step_metrics", []))
    rep.kv("prompt", args.prompt)
    rep.kv("prompt_ids", ids)
    rep.kv("golden_tokens", g_tokens)
    rep.kv("quant_tokens", q_tokens)
    rep.kv("timings_s", {"golden": round(t_golden, 2), "quant_sim": round(t_quant, 2)})
    rep.kv("metrics_ab", {
        "logits_cosine": R.cos_sim(g_logits[-1].numpy(), q_logits[-1].numpy()),
        "logits_kl": R.kl_div(g_logits[-1].numpy(), q_logits[-1].numpy()),
        "top1_agreement": R.top1_agreement(g_logits.numpy(), q_logits.numpy()),
        "gen_token_match": sum(1 for a, b in zip(g_tokens, q_tokens) if a == b) / len(g_tokens)})
    if npu_detail.get("step_rows"):
        rep.kv("metrics_bc", {
            "logits_cosine_prefill": float(npu_detail["step_rows"][0][1]),
            "gen_match": f"{same_bc}/{len(q_tokens)}"})
    rep.save()
    print(f"ISS: {'OK' if iss_ok else 'FAIL'}; "
          f"A↔B cosine={R.cos_sim(g_logits[-1].numpy(), q_logits[-1].numpy()):.6f}")


if __name__ == "__main__":
    main()
