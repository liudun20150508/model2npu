#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ViT 设备链路逐段对齐诊断: 把 ISS 的 READBACK dump 和参考实现逐段对比。
# 用法: python src/dbg_vit_align.py
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C
import mtx_ops_ref as R
import mtx_vit as V
import s3_quant as S3
import s8_vision as S8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=S8.IMAGE_SRC)
    args = ap.parse_args()

    cfg, st = V.load_state(C.VIT_DIR, torch.float32)
    img = S8.load_image(args.image, cfg.image)
    px = V.preprocess(img, cfg.image)

    pk, extras = S8.quantize_vit(cfg, st)
    stq = S3.variant_state(pk, st)
    mq = V.ViTRef(cfg, stq, act_dtype="fp16")
    lgq, capq = mq.forward(px)

    cols = mq.patch_embed_im2col(px)
    print("im2col shape", tuple(cols.shape))
    print("im2col row0[:6]", np.round(cols[0, :6].numpy(), 4))
    print("im2col row1[:6]", np.round(cols[1, :6].numpy(), 4))
    print("im2col row196 is patch row of last block [:6]",
          np.round(cols[-1, :6].numpy(), 4))
    dev_img = np.fromfile(os.path.join(C.BUILD, "vit_img.bin"), dtype=np.float32)
    print("device img floats", dev_img.size, "expected", 3 * cfg.image * cfg.image)
    print("device img[:6]", np.round(dev_img[:6], 4), "max", float(dev_img.max()))
    print("device img dtype f32 range", float(dev_img.min()), float(dev_img.max()))
    print("ref px[:6]", np.round(px.reshape(-1)[:6].numpy(), 4),
          "max", float(px.max()))

    for tag, key in [("embed", "embed"), ("L0", 0), ("L6", 6)]:
        f = os.path.join(C.RESULTS, "vit_dbg_%s.bin" % tag)
        if not os.path.exists(f):
            print(tag, "missing dump")
            continue
        ref = (capq["embed"] if key == "embed" else capq["layers"][key]).numpy().astype(np.float32)
        got = np.fromfile(f, dtype=np.float32).reshape(ref.shape)
        print("%s: cos=%.6f rel=%.6f maxabs=%.5f refmax=%.3f gotmax=%.3f" % (
            tag, R.cos_sim(ref, got), R.rel_err(ref, got), R.max_abs_err(ref, got),
            float(np.abs(ref).max()), float(np.abs(got).max())))
        per = [round(float(R.cos_sim(ref[i], got[i])), 4) for i in (0, 1, 100, 195, 196)]
        print("   per-row cos [0,1,100,195,196]", per)
        print("   ref row0[:4]", np.round(ref[0][:4], 4), "got row0[:4]", np.round(got[0][:4], 4))
        print("   ref row196[:4]", np.round(ref[196][:4], 4),
              "got row196[:4]", np.round(got[196][:4], 4))

    lgq_last = lgq[0].numpy()
    lg_dev = np.fromfile(os.path.join(C.RESULTS, "npu_logits_cls.bin"),
                         dtype=np.float32).reshape(cfg.tokens, cfg.classes)[-1]
    print("quant-sim top5", np.argsort(-lgq_last)[:5], np.round(np.sort(lgq_last)[-5:][::-1], 3))
    print("device    top5", np.argsort(-lg_dev)[:5], np.round(np.sort(lg_dev)[-5:][::-1], 3))
    print("device logits range", float(lg_dev.min()), float(lg_dev.max()))
    print("quant-sim logits range", float(lgq_last.min()), float(lgq_last.max()))


if __name__ == "__main__":
    main()
