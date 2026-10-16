#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4b 微码生成: 把模型 + 量化权重 + 内存规划翻译成 NPU 微码。

对应 能力项 2/4 的 "算子映射 / 数据布局 / 静态内存规划":
  * 每个模型算子映射到一条 ISA 指令 (GEMM4/GEMM8/RMSNORM/ROPE/KVAPPEND/ATTNGQA/...);
  * 分块参数 (ntile/ktile) 直接取自 S1 的 SRAM 规划, 装不下就换更小的块;
  * KV Cache / 激活张量在 DRAM 的驻留量有上界 (声明式分配);
  * prefill / 连续 decode / 长上下文三种形态共用同一份微码模板。

产出: build/npu_program_<variant>.mtx + .json (元信息)
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C


def load_idx(idx_path):
    """读设备镜像的 flat idx: 返回 {tensor_name: {dtype, bits, group, has_inv, ...}}。"""
    info = {}
    for line in open(idx_path, encoding="utf-8"):
        if line.startswith("tensor "):
            parts = line.split()
            name, dtype = parts[1], parts[3]
            kv = dict(p.split("=") for p in parts[4:] if "=" in p)
            info[name] = {"dtype": dtype, "bits": int(dtype[3:]) if dtype.startswith("int") else 16,
                          "group": int(kv.get("group", 0)),
                          "inv": int(kv.get("inv_bytes", 0)) > 0,
                          "N": int(kv.get("N", 0)), "K": int(kv.get("K", 0))}
    return info


def tile_for(manifest, name_pat, k, default_n=256, default_k=0):
    """从 S1 的 SRAM 规划里取分块参数 (没有就用默认)。"""
    for p in manifest.get("sram_plan", {}).get("plans", []):
        if p.get("gemm") == name_pat:
            return p.get("N_tile", default_n), p.get("K_tile", default_k)
    return default_n, default_k


def gen(cfg, tensors, prompt_len, decode_steps, layers, kv_dtype, dump_layers_dir,
        max_ctx, s1=None, variant="w4a16_rtn_mix8", vision_rows=0, vision_file=None):
    """vision_rows>0 时走多模态路径: 先把 197 个视觉 token 投到文本嵌入空间,
    再拼接 prompt token 的 embedding, 一起进 prefill + decode。"""
    H, I, V, L = cfg["hidden"], cfg["inter"], cfg["vocab"], cfg["layers"]
    pre_len = vision_rows + prompt_len
    maxT = pre_len + decode_steps
    out = []
    a = out.append
    a(f"# MTX-NPU 微码 | variant={variant} | prefill={prompt_len} tokens + decode {decode_steps} steps")
    a(f"# 激活/KV 驻留 DRAM; 权重从设备镜像流式读取 (int4 打包 + fp16 scale + int8 zp)")
    a("SRAM 512")
    # ---- 激活张量声明 (静态内存规划: 每类的上界都可算)
    decls = [("x", H), ("x2", H), ("ln1", H), ("ln2", H), ("awqt", H),
             ("q", H), ("k", cfg["kv_heads"] * cfg["head_dim"]),
                       ("v", cfg["kv_heads"] * cfg["head_dim"]),
             ("attn", H), ("o", H), ("gate", I), ("up", I), ("act", I), ("logits", V)]
    if vision_rows:
        decls.append(("vt", cfg.get("vision_hidden", 192)))
    for name, cols in decls:
        a(f"ACT {name} {maxT} {cols} fp16")
    for i in range(L):
        a(f"KVC kvk.L{i} {max_ctx} {kv_dtype}")
        a(f"KVC kvv.L{i} {max_ctx} {kv_dtype}")
    a(f"# 静态内存规划: 激活 {maxT} 行 x (H={H}, I={I}, V={V}) + KV {L} 层 x {max_ctx} x {kv_dtype}")

    def layer_block(i, t_off, tlen, phase, causal):
        p = f"model.layers.{i}."
        a(f"# ---- layer {i} ({phase}, T={tlen}, pos={t_off}) ----")
        a(f"RMSNORM out=ln1 a=x w={p}input_layernorm.weight")
        for proj, buf, ncols in [("q_proj", "q", H), ("k_proj", "k", cfg["kv_heads"] * cfg["head_dim"]),
                                 ("v_proj", "v", cfg["kv_heads"] * cfg["head_dim"])]:
            w = f"{p}self_attn.{proj}.weight"
            n_t, k_t = tile_for(s1 or {}, "q/k/v/o" if ncols == H else "q/k/v/o", cfg["hidden"])
            src = "ln1"
            if tensors.get(w, {}).get("inv"):
                a(f"VSCALE out=awqt a=ln1 s={w}")
                src = "awqt"
            a(f"GEMM out={buf} a={src} w={w} bias={p}self_attn.{proj}.bias ntile={n_t} ktile={k_t}")
        a(f"ROPE q=q k=k pos={t_off}")
        a(f"KVWRITE kc=kvk.L{i} vc=kvv.L{i} k=k v=v pos={t_off} kv_dtype={kv_dtype}")
        a(f"ATTN out=attn q=q kc=kvk.L{i} vc=kvv.L{i} mode={causal} pos={t_off}")
        w = f"{p}self_attn.o_proj.weight"
        n_t, k_t = tile_for(s1 or {}, "q/k/v/o", H)
        src = "attn"
        if tensors.get(w, {}).get("inv"):
            a(f"VSCALE out=awqt a=attn s={w}")
            src = "awqt"
        a(f"GEMM out=o a={src} w={w} ntile={n_t} ktile={k_t}")
        a(f"ADD out=x2 a=x b=o")
        a(f"RMSNORM out=ln2 a=x2 w={p}post_attention_layernorm.weight")
        for proj in ("gate_proj", "up_proj"):
            w = f"{p}mlp.{proj}.weight"
            n_t, k_t = tile_for(s1 or {}, "gate/up", H)
            src = "ln2"
            if tensors.get(w, {}).get("inv"):
                a(f"VSCALE out=awqt a=ln2 s={w}")
                src = "awqt"
            a(f"GEMM out={'gate' if 'gate' in proj else 'up'} a={src} w={w} ntile={n_t} ktile={k_t}")
        a("SILUMUL out=act gate=gate up=up")
        w = f"{p}mlp.down_proj.weight"
        n_t, k_t = tile_for(s1 or {}, "down", I)
        src = "act"
        if tensors.get(w, {}).get("inv"):
            a(f"VSCALE out=awqt a=act s={w}")
            src = "awqt"
        a(f"GEMM out=o a={src} w={w} ntile={n_t} ktile={k_t}")
        a("ADD out=x a=x2 b=o")
        if dump_layers_dir and (phase == "prefill" or phase == "decode1"):
            a(f"READBACK file={dump_layers_dir}/L{i}_{phase}.bin buf=x")

    # ---- prefill
    a("PHASE prefill")
    if vision_rows:
        a("SETT 1")
        a(f"LOAD_HOST file={vision_file} buf=vt roff=0")
        a(f"GEMM out=x a=vt w=mtx.projector.weight ntile=256 ktile=192")
    a(f"SETT {prompt_len}")
    a(f"EMBED out=x table=model.embed_tokens.weight tok=0 roff={vision_rows}")
    a(f"SETT {pre_len}")
    for i in layers:
        layer_block(i, 0, pre_len, "prefill", "prefill")
    a(f"RMSNORM out=ln2 a=x w=model.norm.weight")
    a("LMHEAD out=logits a=ln2 w=model.embed_tokens.weight")
    a("READBACK file=@OUT@/npu_logits_prefill.bin buf=logits")
    a("SAMPLE buf=logits")
    # ---- 连续 decode
    for s in range(decode_steps):
        pos = pre_len + s
        a(f"# ==== decode step {s + 1}/{decode_steps} (pos={pos}) ====")
        a(f"PHASE decode{s + 1}")
        a("SETT 1")
        a(f"EMBED out=x table=model.embed_tokens.weight tok={prompt_len + s} roff={vision_rows}")
        for i in layers:
            layer_block(i, pos, 1, f"decode{s + 1}", "decode")
        a("RMSNORM out=ln2 a=x w=model.norm.weight")
        a("LMHEAD out=logits a=ln2 w=model.embed_tokens.weight")
        a(f"READBACK file=@OUT@/npu_logits_step{s + 1}.bin buf=logits")
        a("SAMPLE buf=logits")
    a("HALT")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a16_rtn_mix8")
    ap.add_argument("--image", default=os.path.join(C.BUILD, "npu_device_image"))
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--kv-dtype", default="fp16", choices=["fp16", "int8"])
    ap.add_argument("--max-ctx", type=int, default=0)
    ap.add_argument("--dump-layers", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--vision-rows", type=int, default=0,
                    help=">0 时走多模态: 视觉 token 数 (DeiT-tiny 是 197)")
    ap.add_argument("--vision-file", default="")
    args = ap.parse_args()
    C.ensure_dirs()
    idx = args.image + ".idx"
    man = C.jload(args.image + ".json")
    tensors = load_idx(idx)
    cfg = man["config"]
    layers = list(range(cfg["layers"])) if args.layers == "all" else \
        [int(x) for x in args.layers.split(",")]
    max_ctx = args.max_ctx or (args.vision_rows + args.prompt_len + args.decode + 8)
    s1_path = os.path.join(C.RESULTS, "s1_model_analysis.json")
    s1 = C.jload(s1_path) if os.path.exists(s1_path) else None
    out_path = args.out or os.path.join(C.BUILD, f"npu_program_{args.variant}.mtx")
    txt = gen(cfg, tensors, args.prompt_len, args.decode, layers, args.kv_dtype,
              args.dump_layers, max_ctx, s1, args.variant,
              vision_rows=args.vision_rows, vision_file=args.vision_file or None)
    txt = txt.replace("@OUT@", C.RESULTS)
    C.wtext(out_path, txt)
    meta = {"program": os.path.relpath(out_path, C.ROOT), "variant": args.variant,
            "prompt_len": args.prompt_len, "decode_steps": args.decode,
            "layers": layers, "kv_dtype": args.kv_dtype, "max_ctx": max_ctx,
            "instructions": len([l for l in txt.splitlines() if l and not l.startswith("#")]),
            "image": os.path.relpath(args.image, C.ROOT)}
    C.jdump(meta, out_path.replace(".mtx", ".json"))
    print(f"微码: {meta['program']}  {meta['instructions']} 条指令, layers={len(layers)}, "
          f"kv={args.kv_dtype}, prompt={args.prompt_len}, decode={args.decode}")


if __name__ == "__main__":
    main()
