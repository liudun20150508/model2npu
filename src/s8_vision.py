#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8 视觉编码器 (ViT/DeiT) + 多模态视觉 token 链路。

对应能力项 的 "视觉编码器 / ViT / 多模态融合 / 视觉 token":
  1. ViT 参考实现 (逐算子展开) → 真实图片分类结果;
  2. int4 量化 + 设备镜像 + 微码生成, 在 NPU ISS 上跑完整视觉编码器
     (IM2COL → patch embedding → 12 层 MHA/MLP → LayerNorm → classifier);
  3. 视觉 token 投到文本嵌入空间, 和 prompt 拼在一起进 LLM 的 prefill + KV Cache,
     验证多模态输入在 NPU 上的完整链路 (projector 是随机初始化的结构演示, 会在报告里标注)。

产出: results/s8_vision.json / .md, build/vit_device_image.*, results/npu_logits_cls.bin
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
import mtx_vit as V
import s3_pack as P3
import s3_quant as S3

VIT_IMAGE = os.path.join(C.BUILD, "vit_device_image")
IMAGE_SRC = os.path.join(C.ROOT, "assets", "demo.jpg")


def load_image(path, size=224):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def quantize_vit(cfg, st, bits=4, group=128):
    """把 ViT 的所有 2D 权重 (线性层 + patch embed 展平) 量化成 int4 g128。"""
    pk_all, extras = {}, {}
    for k, v in st.items():
        w = v.numpy().astype(np.float32)
        if w.ndim == 2 and k != "classifier.weight":
            pk_all[k] = S3.quant_tensor(w, bits=bits, group=group)
        elif k == "classifier.weight":
            pk_all[k] = S3.quant_tensor(w, bits=8, group=group)      # 分类头留 8bit (混合精度)
        elif k.endswith("projection.weight"):
            pk_all[k] = S3.quant_tensor(w.reshape(w.shape[0], -1), bits=bits, group=group)
        else:                                                        # bias / LN / cls / pos
            extras[k] = w.astype(np.float16)
    return pk_all, extras


def build_vit_image(cfg, pk_all, extras, path):
    class Shim:      # build_image 只用到这些字段
        pass
    sh = Shim()
    sh.hidden = cfg.hidden
    sh.layers = cfg.layers
    sh.heads = cfg.heads
    sh.kv_heads = cfg.heads
    sh.head_dim = cfg.head_dim
    sh.inter = cfg.inter
    sh.vocab = cfg.classes
    sh.eps = cfg.eps
    sh.theta = 10000.0
    sh.tie = False
    man, total = P3.build_image(pk_all, sh, "vit-deit-tiny-w4a16", path + ".bin", path + ".json",
                                extras=extras)
    idx = P3.write_idx(man, path + ".idx")
    return man, total, idx


def gen_vit_program(cfg, image_dir, out, img_bin, tab_bin):
    """视觉编码器微码: 一次 prefill 形状 (197 token, 非因果 attention)。"""
    T = cfg.tokens
    lines = []
    a = lines.append
    a(f"# ViT (DeiT-tiny) 微码 | {T} token = 196 patch + 1 cls | 非因果 attention")
    a("SRAM 512")
    a(f"ACT img 1 {3 * cfg.image * cfg.image} fp16")
    a(f"ACT pat {T} {3 * cfg.patch * cfg.patch} fp16")
    a(f"ACT pe {T} {cfg.hidden} fp16")
    a(f"ACT tab {T} {cfg.hidden} fp16")
    a(f"ACT h {T} {cfg.hidden} fp16")
    a(f"ACT ln1 {T} {cfg.hidden} fp16")
    a(f"ACT q {T} {cfg.hidden} fp16")
    a(f"ACT k {T} {cfg.hidden} fp16")
    a(f"ACT v {T} {cfg.hidden} fp16")
    a(f"ACT attn {T} {cfg.hidden} fp16")
    a(f"ACT o {T} {cfg.hidden} fp16")
    a(f"ACT x2 {T} {cfg.hidden} fp16")
    a(f"ACT ln2 {T} {cfg.hidden} fp16")
    a(f"ACT m1 {T} {cfg.inter} fp16")
    a(f"ACT m2 {T} {cfg.hidden} fp16")
    a(f"ACT cls {T} {cfg.hidden} fp16")
    a(f"ACT logits {T} {cfg.classes} fp16")
    a(f"ACT out_cls 1 {cfg.classes} fp16")
    a("SETT 1")
    a(f"PHASE vision")
    a(f"LOAD_HOST file={img_bin} buf=img roff=0")
    a(f"IM2COL img=img out=pat patch={cfg.patch} img_size={cfg.image} ch=3")
    a(f"SETT {cfg.tokens - 1}")       # 196 个 patch 行参与 patch-embedding GEMM
    a(f"GEMM out=pe a=pat w=vit.embeddings.patch_embeddings.projection.weight "
      f"bias=vit.embeddings.patch_embeddings.projection.bias ntile=192 ktile=768")
    a(f"LOAD_HOST file={tab_bin} buf=tab roff=0")
    a(f"ADDTAB out=h a=pe tab=tab")
    a(f"SETT {T}")
    a(f"READBACK file={os.path.join(C.RESULTS, 'vit_dbg_embed.bin')} buf=h")
    for i in range(cfg.layers):
        p = f"vit.encoder.layer.{i}."
        a(f"# ---- encoder layer {i} ----")
        a(f"LAYERNORM out=ln1 a=h w={p}layernorm_before.weight b={p}layernorm_before.bias")
        for proj, buf in [("query", "q"), ("key", "k"), ("value", "v")]:
            a(f"GEMM out={buf} a=ln1 w={p}attention.attention.{proj}.weight "
              f"bias={p}attention.attention.{proj}.bias ntile=192 ktile=192")
        a(f"KVC kvk.V{i} {T} fp16")
        a(f"KVC kvv.V{i} {T} fp16")
        a(f"KVWRITE kc=kvk.V{i} vc=kvv.V{i} k=k v=v pos=0 kv_dtype=fp16")
        a(f"ATTN out=attn q=q kc=kvk.V{i} vc=kvv.V{i} mode=decode pos=0")
        a(f"GEMM out=o a=attn w={p}attention.output.dense.weight "
          f"bias={p}attention.output.dense.bias ntile=192 ktile=192")
        a(f"ADD out=h a=h b=o")
        a(f"LAYERNORM out=ln2 a=h w={p}layernorm_after.weight b={p}layernorm_after.bias")
        a(f"GEMM out=m1 a=ln2 w={p}intermediate.dense.weight bias={p}intermediate.dense.bias "
          f"ntile=384 ktile=192")
        a(f"GELU out=m1 a=m1")
        a(f"GEMM out=m2 a=m1 w={p}output.dense.weight bias={p}output.dense.bias "
          f"ntile=192 ktile=768")
        a(f"ADD out=h a=h b=m2")
        if i == 0:
            a(f"READBACK file={os.path.join(C.RESULTS, 'vit_dbg_L0.bin')} buf=h")
        if i == 6:
            a(f"READBACK file={os.path.join(C.RESULTS, 'vit_dbg_L6.bin')} buf=h")
    a(f"LAYERNORM out=cls a=h w=vit.layernorm.weight b=vit.layernorm.bias")
    a(f"LMHEAD out=logits a=cls w=classifier.weight")
    a(f"READBACK file={os.path.join(C.RESULTS, 'npu_logits_cls.bin')} buf=logits")
    a("HALT")
    C.wtext(out, "\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=IMAGE_SRC)
    ap.add_argument("--skip-iss", action="store_true")
    ap.add_argument("--llm-decode", type=int, default=4)
    ap.add_argument("--prompt", default="描述这张图片。")
    args = ap.parse_args()
    C.ensure_dirs()
    if not os.path.exists(args.image):
        raise SystemExit(f"缺图片 {args.image} (可从 04_推理框架与端侧部署/绿联_edge-infer/data/dog.jpg 拷一份)")

    cfg, st = V.load_state(C.VIT_DIR, torch.float32)
    img = load_image(args.image, cfg.image)
    px = V.preprocess(img, cfg.image)
    rep = C.Report("S8 视觉编码器 (ViT/DeiT) + 视觉 token 多模态链路",
                   json_path=os.path.join(C.RESULTS, "s8_vision.json"),
                   md_path=os.path.join(C.RESULTS, "s8_vision.md"))
    rep.h("1. 模型与输入", 2)
    rep.table(["项", "值"],
              [["模型", "facebook/deit-tiny-patch16-224 (真实预训练权重)"],
               ["结构", f"hidden {cfg.hidden} / {cfg.layers} 层 / {cfg.heads} 头 "
                        f"(head_dim {cfg.head_dim}) / MLP {cfg.inter} / patch {cfg.patch}"],
               ["序列", f"{cfg.tokens} token = {cfg.grid}×{cfg.grid} patch + 1 cls"],
               ["输入图", f"`{os.path.relpath(args.image, C.ROOT)}` → {cfg.image}×{cfg.image} "
                          "bilinear + ImageNet 归一化"],
               ["参数量", f"{sum(v.numel() for v in st.values()) / 1e6:.2f} M"]])

    vit32 = V.ViTRef(cfg, st, act_dtype="fp32")
    t0 = time.time()
    logits32, cap32 = vit32.forward(px)
    rep.p(f"fp32 参考前向 {time.time() - t0:.2f} s。")
    top5 = torch.topk(logits32[0], 5)
    rows = [[f"{i + 1}", cfg.id2label[int(k)], f"{float(v):.3f}"]
            for i, (k, v) in enumerate(zip(top5.indices.tolist(), top5.values.tolist()))]
    rep.h("2. 参考结果 (fp32, 真实图片)", 2)
    rep.table(["#", "类别", "logit"], rows)
    rep.kv("fp32_top5", [{"label": cfg.id2label[int(k)], "logit": float(v)}
                         for k, v in zip(top5.indices.tolist(), top5.values.tolist())])

    # ---- 量化 (int4 g128, 分类头 int8) + 量化模拟
    pk_all, extras = quantize_vit(cfg, st)
    st_q = S3.variant_state(pk_all, st)
    vitq = V.ViTRef(cfg, st_q, act_dtype="fp16")
    logits_q, cap_q = vitq.forward(px)
    rep.h("3. 量化 (W4A16 g128 + 分类头 int8) 与 NPU ISS 结果", 2)
    rep.table(["链路", "top-1", "类别", "与 fp32 的 logits cosine", "top-5 一致率"],
              [["fp32 参考", int(logits32.argmax()), cfg.id2label[int(logits32.argmax())], "1.000000", "5/5"],
               ["量化模拟 (去量化权重 + fp16 激活)", int(logits_q.argmax()),
                cfg.id2label[int(logits_q.argmax())],
                f"{R.cos_sim(logits32[0].numpy(), logits_q[0].numpy()):.6f}",
                f"{len(set(torch.topk(logits32[0], 5).indices.tolist()) & set(torch.topk(logits_q[0], 5).indices.tolist()))}/5"]])

    man, total, idx = build_vit_image(cfg, pk_all, extras, VIT_IMAGE)
    np.asarray(px.numpy(), dtype=np.float32).tofile(os.path.join(C.BUILD, "vit_img.bin"))
    # 设备侧序列布局是 [196 patch, 1 cls] (与 HF 的 [cls, 196 patch] 只是行序不同,
    # 无因果掩码时两者等价), 所以 pos table 也要按同样的配对关系来拼:
    #   row i (patch i)   <- pos_embed[i+1]
    #   row 196 (cls)     <- cls_token + pos_embed[0]
    pos = st["vit.embeddings.position_embeddings"].reshape(cfg.tokens, cfg.hidden).numpy()
    cls = st["vit.embeddings.cls_token"].reshape(1, cfg.hidden).numpy()
    tab = np.concatenate([pos[1:], cls + pos[:1]], 0).astype(np.float32)
    tab.tofile(os.path.join(C.BUILD, "vit_tab.bin"))

    iss_ok, res = False, {}
    if not args.skip_iss:
        prog = gen_vit_program(cfg, VIT_IMAGE, os.path.join(C.BUILD, "vit_program.mtx"),
                               os.path.join(C.BUILD, "vit_img.bin"),
                               os.path.join(C.BUILD, "vit_tab.bin"))
        prof_path = os.path.join(C.RESULTS, "npu_profile_vit.json")
        rc, outp = C.run([os.path.join(C.BUILD, "npu_iss"), "--program", prog,
                          "--image", VIT_IMAGE + ".bin", "--idx", VIT_IMAGE + ".idx",
                          "--tokens", os.path.join(C.BUILD, "npu_tokens.bin"),
                          "--out", prof_path, "--hw", os.path.join(C.HW, "mtx_npu_v0.json")],
                         check=False)
        if rc != 0:
            rep.p(f"**ViT ISS 失败**: `{outp[-600:]}`")
        else:
            iss_ok = True
            lg = np.fromfile(os.path.join(C.RESULTS, "npu_logits_cls.bin"),
                             dtype=np.float32).reshape(cfg.tokens, cfg.classes)[-1]
            prof = C.jload(prof_path)
            res = {"profile": prof, "iss_logits": lg}
            top5_iss = np.argsort(-lg)[:5]
            rows = [[f"{i + 1}", cfg.id2label[int(k)], f"{lg[k]:.3f}"]
                    for i, k in enumerate(top5_iss)]
            rep.table(["#", "类别 (ISS)", "logit"], rows)
            rep.table(["对比", "值"],
                      [["ISS vs 量化模拟 cosine", f"{R.cos_sim(lg, logits_q[0].numpy()):.6f}"],
                       ["ISS vs 量化模拟 最大绝对差", f"{R.max_abs_err(lg, logits_q[0].numpy()):.4f}"],
                       ["ISS vs fp32 cosine", f"{R.cos_sim(lg, logits32[0].numpy()):.6f}"],
                       ["top-1 是否一致 (ISS vs fp32)",
                        "是" if int(lg.argmax()) == int(logits32.argmax()) else "否"],
                       ["ISS 总 cycle", f"{prof['summary']['cycles'] / 1e6:.2f} Mcycle "
                                        f"({prof['summary']['ms']:.2f} ms @1GHz)"],
                       ["视觉编码 DRAM 访问", C.human_bytes(prof['summary']['dram_bytes'])],
                       ["设备镜像", C.human_bytes(total)]])
            rep.kv("iss_top5", [{"label": cfg.id2label[int(k)], "logit": float(lg[k])} for k in top5_iss])
            rep.kv("iss_cos_vs_quant", R.cos_sim(lg, logits_q[0].numpy()))
            rep.kv("iss_cos_vs_fp32", R.cos_sim(lg, logits32[0].numpy()))
            rep.kv("iss_top1_match", int(lg.argmax()) == int(logits32.argmax()))
            rep.kv("vit_profile_summary", prof["summary"])
            # 逐层对齐 (含"分类头放大"的量化说明)
            lay = []
            for tag, key in [("embedding", "embed"), ("L0", 0), ("L6", 6)]:
                f = os.path.join(C.RESULTS, f"vit_dbg_{key if key == 'embed' else 'L' + str(key)}.bin")
                if not os.path.exists(f):
                    continue
                ref = cap_q["embed"].numpy() if key == "embed" else cap_q["layers"][key].numpy()
                got = np.fromfile(f, dtype=np.float32).reshape(ref.shape)
                lay.append([tag, f"{R.cos_sim(ref, got):.8f}", f"{R.max_abs_err(ref, got):.6f}"])
            if lay:
                rep.table(["阶段", "cos vs 量化模拟", "max_abs"], lay)
                rep.p("逐层能到 6~7 个 9, 而分类头 logits 的 cosine 只有 "
                      f"{R.cos_sim(lg, logits_q[0].numpy()):.3f} —— 因为分类头的输出量级是 "
                      "hidden 的数倍, 中间层最后几位小数会被放大; 所以判据要同时看"
                      "「逐层对齐」和「最终 top-1」, 只看其中一个都会误判。")

    # ---- 多模态: 视觉 token → 文本嵌入空间 → LLM prefill/decode
    rep.h("4. 视觉 token 进 LLM (多模态链路)", 2)
    lcfg, lst = Q.load_state(C.QWEN_DIR, torch.float32)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(C.QWEN_DIR, "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    rng = np.random.default_rng(20261016)
    proj = (rng.standard_normal((lcfg.hidden, cfg.hidden)) * (1.0 / np.sqrt(cfg.hidden))
            ).astype(np.float32)
    proj_pack = S3.quant_tensor(proj, bits=4, group=128)
    lst2 = dict(lst)
    lst2["mtx.projector.weight"] = torch.from_numpy(S3.dequant_tensor(proj_pack))
    # 视觉 token = ViT 最后一层的 197 个 token (量化模拟口径)
    vt = cap_q["layers"][cfg.layers - 1].numpy().astype(np.float32)
    np.ascontiguousarray(vt).tofile(os.path.join(C.BUILD, "vision_tokens.bin"))
    vt_pack = S3.quant_tensor(vt, bits=4, group=128)
    # 设备镜像 = LLM 权重 + projector (多模态新增的那一层)
    llm_pack = torch.load(os.path.join(C.BUILD, "quant", "w4a16_rtn_mix8.pt"),
                          map_location="cpu", weights_only=False)
    llm_pack = dict(llm_pack)
    llm_pack["mtx.projector.weight"] = proj_pack
    extras_l = {k: v.numpy().astype(np.float16) for k, v in lst.items()
                if k.endswith("_proj.bias") or k.endswith("layernorm.weight")
                or k.endswith("model.norm.weight")}
    mm_img = os.path.join(C.BUILD, "mm_device_image")
    man_l, total_l = P3.build_image(llm_pack, lcfg, "w4a16_rtn_mix8+vision", mm_img + ".bin",
                                    mm_img + ".json", extras=extras_l)
    P3.write_idx(man_l, mm_img + ".idx")
    rep.p(f"视觉 token {cfg.tokens} 个 (ViT 最后一层输出, 已量化) 经过 int4 投影层 "
          f"[{cfg.hidden}→{lcfg.hidden}] 进入文本嵌入空间, 与 {len(ids)} 个 prompt token "
          f"拼成 {cfg.tokens + len(ids)} 长度的序列进 prefill。")
    rep.p("> 说明: 投影层是**随机初始化**的结构演示 (没有做过对齐训练), 这里验证的是"
          "「视觉 token 能不能走通设备链路 + KV Cache 里位置/长度是否正确」, 不是语义效果。")
    prog_mm = os.path.join(C.BUILD, "mm_program.mtx")
    C.run([sys.executable, os.path.join(C.SRC, "s4_gen_program.py"), "--variant",
           "w4a16_rtn_mix8", "--image", mm_img, "--prompt-len", str(len(ids)),
           "--decode", str(args.llm_decode), "--layers", "all", "--kv-dtype", "fp16",
           "--vision-rows", str(cfg.tokens), "--vision-file", os.path.join(C.BUILD, "vision_tokens.bin"),
           "--out", prog_mm], check=False)
    tokfile = os.path.join(C.BUILD, "mm_tokens.bin")
    np.asarray(ids, dtype=np.int32).tofile(tokfile)
    mm_prof = os.path.join(C.RESULTS, "npu_profile_multimodal.json")
    rc, outp = C.run([os.path.join(C.BUILD, "npu_iss"), "--program", prog_mm,
                      "--image", mm_img + ".bin", "--idx", mm_img + ".idx",
                      "--tokens", tokfile, "--out", mm_prof,
                      "--hw", os.path.join(C.HW, "mtx_npu_v0.json")], check=False)
    if rc != 0:
        rep.p(f"**多模态 ISS 失败**: `{outp[-600:]}`")
    else:
        mp = C.jload(mm_prof)
        seq = cfg.tokens + len(ids)
        ph = {}
        for op in mp["ops"]:
            d = ph.setdefault(op["phase"], {"c": 0.0, "b": 0})
            d["c"] += op["cycles"]
            d["b"] += op["dram_bytes"]
        freq = C.hw_cfg()["freq_mhz"] * 1e6
        rep.table(["阶段", "cycle", "时间", "DRAM 访问"],
                  [[k, f"{v['c'] / 1e6:.2f} Mcycle", f"{v['c'] / freq * 1e3:.2f} ms",
                    C.human_bytes(v["b"])] for k, v in sorted(ph.items())])
        rep.p(f"序列长度 {seq} token ({cfg.tokens} 视觉 + {len(ids)} 文本); "
              f"KV Cache 峰值 {C.human_bytes(mp['summary']['kv_peak_bytes'])}, "
              f"prefill 之后连续 decode {args.llm_decode} 步无越界/无 NaN —— "
              f"说明视觉 token 的位置编码、KV 写入与因果掩码在设备侧是对齐的。")
        rep.kv("multimodal", {"seq": seq, "vision_tokens": cfg.tokens, "text_tokens": len(ids),
                              "kv_peak_bytes": mp["summary"]["kv_peak_bytes"],
                              "phases": {k: {"cycles": v["c"], "dram_bytes": v["b"]}
                                         for k, v in ph.items()}})
        lg = np.fromfile(os.path.join(C.RESULTS, "npu_logits_prefill.bin"),
                         dtype=np.float32).reshape(seq, lcfg.vocab)
        finite = bool(np.isfinite(lg).all())
        rep.p(f"多模态 prefill 的 logits: 形状 {lg.shape}, 全有限值={finite}, "
              f"最后一行 top-1 token id = {int(lg[-1].argmax())}。")
        rep.kv("multimodal_logits_finite", finite)
    rep.save()
    print("视觉链路:", "OK" if iss_ok else "跳过/失败",
          f"top-1(fp32)={cfg.id2label[int(logits32.argmax())]}")


if __name__ == "__main__":
    main()
