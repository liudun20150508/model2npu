#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ViT (DeiT-tiny) 参考实现 —— 视觉编码器侧的算子与数据布局。

和 mtx_qwen.py 一样的思路: 按 config 把模型逐算子展开, 每个算子的形状/精度/边界
都由我们自己定义, 便于和 C++ / NPU ISS 三端对齐。

DeiT-tiny patch16-224: hidden 192 / 12 层 / 3 头 (head_dim 64) / MLP 768 /
patch 16 → 196 patch + 1 cls = 197 token / GELU(erf) / LayerNorm(eps=1e-12)
"""
import json
import os

import numpy as np
import torch

torch.set_grad_enabled(False)


class ViTConfig:
    def __init__(self, cfg):
        self.hidden = cfg["hidden_size"]
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.inter = cfg["intermediate_size"]
        self.patch = cfg["patch_size"]
        self.image = cfg["image_size"]
        self.classes = cfg["num_labels"] if "num_labels" in cfg else len(cfg["id2label"])
        self.eps = cfg.get("layer_norm_eps", 1e-12)
        self.head_dim = self.hidden // self.heads
        self.grid = self.image // self.patch
        self.tokens = self.grid * self.grid + 1
        self.scale = self.head_dim ** -0.5
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}

    def as_dict(self):
        d = dict(self.__dict__)
        d.pop("id2label", None)
        return d


def load_state(model_dir, dtype=torch.float32):
    cfg = ViTConfig(json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8")))
    st = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu",
                    weights_only=True)
    return cfg, {k: v.to(dtype).contiguous() for k, v in st.items()}


class ViTRef:
    """逐算子展开的 ViT 前向。act_dtype 与 Qwen2Ref 同口径。"""

    def __init__(self, cfg: ViTConfig, state, act_dtype="fp32", name="vit"):
        self.cfg = cfg
        self.st = state
        self.act_dtype = act_dtype
        self.name = name
        self.act_hook = None

    def retype(self, x):
        if self.act_dtype == "fp16":
            return x.half().float()
        if self.act_dtype == "bf16":
            return x.bfloat16().float()
        return x

    def layernorm(self, x, w, b):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, unbiased=False, keepdim=True)
        return self.retype((x - m) / torch.sqrt(v + self.cfg.eps) * w + b)

    def gelu(self, x):
        return self.retype(torch.nn.functional.gelu(x, approximate="none"))

    def linear(self, x, w, b=None, name=None):
        if self.act_hook is not None and name is not None:
            self.act_hook(name, x)
        y = x.to(torch.float32) @ w.to(torch.float32).t()
        if b is not None:
            y = y + b.to(torch.float32)
        return self.retype(y)

    def patch_embed_im2col(self, img):
        """img:[C,H,W] fp32 -> [P, C*patch*patch] (与设备侧 IM2COL 同布局)。"""
        cfg = self.cfg
        p, g = cfg.patch, cfg.grid
        x = img.reshape(cfg.hidden and img.shape[0], cfg.image, cfg.image)
        cols = []
        for gy in range(g):
            for gx in range(g):
                blk = x[:, gy * p:(gy + 1) * p, gx * p:(gx + 1) * p]      # [C,p,p]
                cols.append(blk.reshape(-1))
        return torch.stack(cols, 0)                                       # [P, C*p*p]

    def forward(self, pixel_values, capture=None):
        cfg, st = self.cfg, self.st
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        img = pixel_values.to(torch.float32)
        cap = capture if capture is not None else {}
        cols = self.patch_embed_im2col(img)                               # [196, 768]
        Wp = st["vit.embeddings.patch_embeddings.projection.weight"].reshape(cfg.hidden, -1)
        bp = st["vit.embeddings.patch_embeddings.projection.bias"]
        pe = self.linear(cols, Wp, bp, name="vit.embeddings.patch_embeddings.projection.weight")
        cap["im2col"] = cols.clone()
        cap["patch_embed"] = pe.clone()
        cls = st["vit.embeddings.cls_token"].reshape(1, -1)
        pos = st["vit.embeddings.position_embeddings"].reshape(cfg.tokens, cfg.hidden)
        # 设备侧序列布局 [196 patch, 1 cls]: patch i 配 pos_embed[i+1], cls 配 pos_embed[0]
        # (无因果掩码时与 HF 的 [cls, 196 patch] 布局等价)
        h = torch.cat([pe, torch.zeros(1, cfg.hidden)], 0)                # 最后一行为 cls 占位
        tab = torch.cat([pos[1:], cls + pos[:1]], 0)
        self.tab = tab.clone()
        h = self.retype(h + tab)
        cap["embed"] = h.clone()
        for i in range(cfg.layers):
            p = f"vit.encoder.layer.{i}."
            res = h
            x = self.layernorm(h, st[p + "layernorm_before.weight"], st[p + "layernorm_before.bias"])
            T = x.shape[0]
            q = self.linear(x, st[p + "attention.attention.query.weight"],
                            st[p + "attention.attention.query.bias"],
                            name=p + "attention.attention.query.weight")
            k = self.linear(x, st[p + "attention.attention.key.weight"],
                            st[p + "attention.attention.key.bias"],
                            name=p + "attention.attention.key.weight")
            v = self.linear(x, st[p + "attention.attention.value.weight"],
                            st[p + "attention.attention.value.bias"],
                            name=p + "attention.attention.value.weight")
            q = self.retype(q.view(T, cfg.heads, cfg.head_dim).permute(1, 0, 2))
            k = self.retype(k.view(T, cfg.heads, cfg.head_dim).permute(1, 0, 2))
            v = self.retype(v.view(T, cfg.heads, cfg.head_dim).permute(1, 0, 2))
            scores = torch.einsum("htd,hsd->hts", q.to(torch.float32), k.to(torch.float32)) * cfg.scale
            attn = self.retype(torch.softmax(self.retype(scores), dim=-1))
            ctx = torch.einsum("hts,hsd->htd", attn, v.to(torch.float32))
            ctx = ctx.permute(1, 0, 2).reshape(T, cfg.hidden)
            o = self.linear(self.retype(ctx), st[p + "attention.output.dense.weight"],
                            st[p + "attention.output.dense.bias"],
                            name=p + "attention.output.dense.weight")
            h = self.retype(res + o)
            res = h
            x = self.layernorm(h, st[p + "layernorm_after.weight"], st[p + "layernorm_after.bias"])
            m1 = self.linear(x, st[p + "intermediate.dense.weight"], st[p + "intermediate.dense.bias"],
                             name=p + "intermediate.dense.weight")
            m1 = self.gelu(m1)
            m2 = self.linear(m1, st[p + "output.dense.weight"], st[p + "output.dense.bias"],
                             name=p + "output.dense.weight")
            h = self.retype(res + m2)
            cap.setdefault("layers", [None] * cfg.layers)[i] = h.clone()
        # 设备/参考布局里 cls token 是**最后一行** (无因果掩码时与 HF 的 [cls, patches] 等价)
        cls_out = self.layernorm(h[cfg.tokens - 1:], st["vit.layernorm.weight"],
                                 st["vit.layernorm.bias"])
        logits = self.linear(cls_out, st["classifier.weight"], st["classifier.bias"],
                             name="classifier.weight")
        cap["cls"] = cls_out.clone()
        cap["logits"] = logits.clone()
        return logits, cap


# ------------------------------------------------------------ 图像预处理
def preprocess(img_hwc_uint8, size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """[H,W,3] uint8 -> [3,size,size] fp32 (bilinear resize + ImageNet 归一化)。"""
    x = torch.from_numpy(np.asarray(img_hwc_uint8)).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1)[None]                                    # [1,3,H,W]
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                        align_corners=False)
    x = x[0]
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x - m) / s
