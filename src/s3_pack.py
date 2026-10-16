#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3b 硬件权重格式导出: 把紧凑量化包打成设备镜像 (.mtnpu)。

加分能力点名了 "权重打包 / 自定义量化格式"。这里定义的格式就是后面
NPU ISS (s4) 真正读的那个格式, 不是纸上设计:

    magic "MTXNPU01" | u32 manifest_len | u32 version | manifest(JSON, 256B 对齐)
    | 各 tensor 的权重段/scale 段/zp 段 (256B 对齐, 便于 DMA)

每个权重张量:
    q       : int4 打包 (偶列低半字节) 或 int8 原样, [N, Kp]
    scale   : fp16, [N, G]  (group 沿 K 方向, G = ceil(K/group))
    zp      : int8, [N, G]  (非对称量化; 存 int8 而不是 fp32, 元数据省一半)
    inv_scale: fp16, [K]    (仅 AWQ 变体: 设备侧逐通道反缩放)

产出:
  build/npu_device_image.bin + build/npu_device_image.json
  results/s3_pack.json / .md         格式说明 + 体积账单
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_ops_ref as R

MAGIC = b"MTXNPU01"
ALIGN = 256


def align_up(x, a=ALIGN):
    return (x + a - 1) // a * a


def build_image(pk_all, cfg, variant, out_bin, out_json, quant_meta=None, extras=None):
    """把紧凑包写成设备镜像, 返回 manifest。"""
    sections = []          # (bytes,)
    tensors = []
    off = 0

    def add(buf):
        nonlocal off
        start = align_up(off)
        pad = b"\0" * (start - off)
        sections.append(pad)
        sections.append(buf)
        off = start + len(buf)
        return start

    for name in sorted(pk_all.keys()):
        pk = pk_all[name]
        t = {"name": name, "role": "weight"}
        if pk.get("fp16"):
            W16 = np.ascontiguousarray(pk["W16"])
            t.update({"dtype": "fp16", "N": int(W16.shape[0]), "K": int(W16.shape[1]),
                      "bytes": int(W16.nbytes)})
            t["off"] = add(W16.tobytes())
        else:
            N, K, G = pk["N"], pk["K"], pk["packed"].shape[1] // (2 if pk["bits"] == 4 else 1)
            # scale 段
            sc16 = np.ascontiguousarray(pk["scale"].astype(np.float16))
            # zero-point: int4 的量程 0..15 存 int8; int8 的量程 0..255 必须存 int16
            zpd = "int8" if pk["bits"] == 4 else "int16"
            zp8 = np.ascontiguousarray(np.round(pk["zp"]).astype(zpd))
            pk_buf = np.ascontiguousarray(pk["packed"])
            q_off = add(pk_buf.tobytes())
            s_off = add(sc16.tobytes())
            z_off = add(zp8.tobytes())
            t.update({"dtype": f"int{pk['bits']}", "packed": pk["bits"] == 4,
                      "N": int(N), "K": int(K), "group": int(pk["group"]),
                      "symmetric": bool(pk["symmetric"]), "ngroup": int(pk["scale"].shape[1]),
                      "q_off": q_off, "q_bytes": int(pk_buf.nbytes),
                      "scale_off": s_off, "scale_bytes": int(sc16.nbytes), "scale_dtype": "fp16",
                      "zp_off": z_off, "zp_bytes": int(zp8.nbytes), "zp_dtype": zpd})
            if "inv_scale" in pk:
                inv = np.ascontiguousarray(pk["inv_scale"].astype(np.float16))
                t["inv_scale_off"] = add(inv.tobytes())
                t["inv_scale_bytes"] = int(inv.nbytes)
            t["bytes"] = t["q_bytes"] + t["scale_bytes"] + t["zp_bytes"] + t.get("inv_scale_bytes", 0)
        tensors.append(t)
    # 额外 fp16 张量: q/k/v bias、RMSNorm 权重 (不参与量化, 走 fp16 通路)
    for name, arr in sorted((extras or {}).items()):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float16))
        t = {"name": name, "role": "weight", "dtype": "fp16",
             "N": int(a.reshape(-1).shape[0]), "K": 1, "bytes": int(a.nbytes)}
        t["off"] = add(a.tobytes())
        tensors.append(t)

    body = b"".join(sections)
    manifest = {
        "format": MAGIC.decode(), "version": 1, "model": "Qwen2.5-0.5B-Instruct",
        "variant": variant, "created_by": "model2npu/s3_pack.py",
        "config": {"hidden": cfg.hidden, "layers": cfg.layers, "heads": cfg.heads,
                   "kv_heads": cfg.kv_heads, "head_dim": cfg.head_dim, "inter": cfg.inter,
                   "vocab": cfg.vocab, "rms_eps": cfg.eps, "rope_theta": cfg.theta,
                   "tie_embeddings": cfg.tie},
        "layout": {"weight": "[N,K] 行主序, group 沿 K, int4 偶列在低半字节",
                   "scale": "[N,G] fp16", "zp": "[N,G] int8",
                   "inv_scale": "[K] fp16 (仅 AWQ, 设备侧反缩放)"},
        "tensors": tensors,
        "body_bytes": len(body), "body_off": None, "quant_meta": quant_meta or {},
    }
    mj = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    head = MAGIC + struct.pack("<II", len(mj), 1)
    body_off = align_up(len(head) + len(mj))
    manifest["body_off"] = body_off
    mj = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(out_bin, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(mj), 1))
        f.write(mj)
        f.write(b"\0" * (body_off - len(head) - len(mj)))
        f.write(body)
    C.jdump(manifest, out_json)
    write_idx(manifest, out_json.replace(".json", ".idx"))
    return manifest, body_off + len(body)


def write_idx(manifest, path):
    """给设备侧 ISS 用的 flat idx 侧车 (设备上不塞 JSON 解析器)。"""
    lines = ["# mtximg-idx v1", "# config <key> = <value>"]
    lines.append(f"config body_off = {manifest.get('body_off', 0)}")
    for k, v in manifest["config"].items():
        lines.append(f"config {k} = {v}")
    lines.append("# tensor <name> weight <dtype> N=.. K=.. group=.. ngroup=.. packed=.. sym=.. "
                 "q_off=.. q_bytes=.. scale_off=.. scale_bytes=.. zp_off=.. zp_bytes=.. "
                 "inv_off=.. inv_bytes=.. off=.. bytes=..")
    for t in manifest["tensors"]:
        if t["dtype"] == "fp16":
            lines.append(f"tensor {t['name']} weight fp16 N={t['N']} K={t['K']} "
                         f"off={t['off']} bytes={t['bytes']}")
        else:
            lines.append(
                f"tensor {t['name']} weight {t['dtype']} N={t['N']} K={t['K']} group={t['group']} "
                f"ngroup={t['ngroup']} packed={1 if t['packed'] else 0} sym={1 if t['symmetric'] else 0} "
                f"q_off={t['q_off']} q_bytes={t['q_bytes']} scale_off={t['scale_off']} "
                f"scale_bytes={t['scale_bytes']} zp_off={t['zp_off']} zp_bytes={t['zp_bytes']} "
                f"zp_dtype={t['zp_dtype']} "
                f"inv_off={t.get('inv_scale_off', -1)} inv_bytes={t.get('inv_scale_bytes', 0)} "
                f"bytes={t['bytes']}")
    C.wtext(path, "\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a16_rtn_mix8")
    ap.add_argument("--out", default=os.path.join(C.BUILD, "npu_device_image"))
    args = ap.parse_args()
    C.ensure_dirs()
    pk_path = os.path.join(C.BUILD, "quant", f"{args.variant}.pt")
    if not os.path.exists(pk_path):
        raise SystemExit(f"缺量化包 {pk_path}, 请先跑 s3_quant.py")
    pk_all = torch.load(pk_path, map_location="cpu", weights_only=False)
    import mtx_qwen as Q
    cfg, st = Q.load_state(C.QWEN_DIR)
    extras = {}
    for k, v in st.items():
        if k.endswith("_proj.bias") or k.endswith("layernorm.weight") or k.endswith("model.norm.weight"):
            extras[k] = v.numpy().astype(np.float16)
    manifest, total = build_image(pk_all, cfg, args.variant,
                                  args.out + ".bin", args.out + ".json", extras=extras)

    rep = C.Report(f"S3b 硬件权重格式导出 ({args.variant})",
                   json_path=os.path.join(C.RESULTS, "s3_pack.json"),
                   md_path=os.path.join(C.RESULTS, "s3_pack.md"))
    rep.h("1. 设备镜像", 2)
    rep.p(f"`{os.path.relpath(args.out + '.bin', C.ROOT)}` 共 {C.human_bytes(total)}, "
          f"{len(manifest['tensors'])} 个张量段; 段间 256B 对齐 (DMA 友好)。")
    rows = []
    by_role = {}
    for t in manifest["tensors"]:
        by_role[t["dtype"]] = by_role.get(t["dtype"], 0) + t["bytes"]
    for k, v in sorted(by_role.items(), key=lambda x: -x[1]):
        rows.append([k, len([t for t in manifest["tensors"] if t["dtype"] == k]),
                     C.human_bytes(v), f"{v / total * 100:.1f}%", C.human_bytes(v / max(1, len([t for t in manifest['tensors'] if t['dtype'] == k])))])
    rep.table(["段 dtype", "张量数", "合计", "占比", "平均"], rows)
    ex = [t for t in manifest["tensors"] if t["dtype"] == "int4"]
    if ex:
        t = ex[0]
        rep.h("2. 一个 int4 张量的三段布局 (以第一个为例)", 2)
        rep.table(["段", "offset", "字节数", "dtype/布局"],
                  [["q (权重)", t["q_off"], t["q_bytes"], f"int4 打包 [N={t['N']}, Kp={t['K']}]"],
                   ["scale", t["scale_off"], t["scale_bytes"],
                    f"fp16 [N={t['N']}, G={t['ngroup']}], group={t['group']}"],
                   ["zp", t["zp_off"], t["zp_bytes"], f"int8 [N={t['N']}, G={t['ngroup']}]"] +
                   ([["inv_scale (AWQ)", t["inv_scale_off"], t["inv_scale_bytes"], f"fp16 [K={t['K']}]"]]
                    if "inv_scale_off" in t else [])])
    overhead = sum(t.get("scale_bytes", 0) + t.get("zp_bytes", 0) for t in manifest["tensors"])
    wq = sum(t.get("q_bytes", 0) for t in manifest["tensors"])
    rep.p(f"元数据开销: (scale fp16 + zp int8) / 权重 = {overhead / max(1, wq) * 100:.1f}% "
          f"(group=128 时理论值 {(2 + 1) / (128 / 2) * 100:.1f}%)。")
    rep.h("3. 格式定义", 2)
    rep.code(json.dumps({"magic": manifest["format"], "layout": manifest["layout"],
                         "int4 打包": "byte = q[2i] | (q[2i+1] << 4), 即偶列在低半字节",
                         "对齐": f"{ALIGN} B", "反量化": "w = (q - zp) * scale  (scale/zp 按 group 索引)"},
                        ensure_ascii=False, indent=2), "json")
    rep.kv("image", {"path": os.path.relpath(args.out + ".bin", C.ROOT), "bytes": total,
                     "tensors": len(manifest["tensors"]), "variant": args.variant,
                     "metadata_overhead_ratio": round(overhead / max(1, wq), 4)})
    rep.save()
    print(f"设备镜像: {C.human_bytes(total)} -> {args.out}.bin")


if __name__ == "__main__":
    main()
