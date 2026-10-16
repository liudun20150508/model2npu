#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen2 参考实现 (纯 PyTorch, 逐算子显式展开)。

为什么不用 transformers 的 Qwen2ForCausalLM:
  1) 本机 transformers 会去 import 坏掉的 torch_npu, 直接 import 失败;
  2) 更重要的是, 需求 要的就是"能读模型实现 + 写参考算子 + 逐算子对齐",
     参考模型按 config 显式展开后, 每个边界 (RoPE 配对、GQA 广播、
     q/k/v bias、KV 布局) 都是我们自己定义的, 能和 C++/ISS 端严格一致。

数据排布约定 (三端一致):
  激活      x:[T,H]  fp32 (可切换成 fp16 存储)
  线性权重  W:[N,K]  (与 safetensors 一致),  y = x @ W.T + b
  q/k/v     [T, Hq|Hkv, D]  D=H/Hq
  KV cache  [T, Hkv, D]  (按层分开)
"""
import json
import os

import numpy as np
import torch

torch.set_grad_enabled(False)


class Qwen2Config:
    def __init__(self, cfg):
        self.hidden = cfg["hidden_size"]
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.inter = cfg["intermediate_size"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["rope_theta"])
        self.max_pos = cfg["max_position_embeddings"]
        self.tie = cfg.get("tie_word_embeddings", True)
        self.head_dim = self.hidden // self.heads
        self.scale = self.head_dim ** -0.5

    def as_dict(self):
        return self.__dict__.copy()


def load_state(model_dir, dtype=torch.float32):
    """读 safetensors / pytorch_model.bin, 统一成 fp32 的 dict。"""
    cfg = Qwen2Config(json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8")))
    saf = os.path.join(model_dir, "model.safetensors")
    st = {}
    if os.path.exists(saf) and os.path.getsize(saf) > 1024:
        from safetensors.torch import load_file
        st = load_file(saf)
    else:
        st = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu",
                        weights_only=True)
    out = {k: v.to(dtype).contiguous() for k, v in st.items()}
    return cfg, out


class Qwen2Ref:
    """逐算子展开的 Qwen2 前向。act_dtype: 'fp32' | 'fp16' | 'bf16'。"""

    def __init__(self, cfg: Qwen2Config, state, act_dtype="fp32", name="qwen2"):
        self.cfg = cfg
        self.st = state
        self.act_dtype = act_dtype
        self.name = name
        self.act_hook = None      # 标定用: hook(name, x) 拿到每个 linear 的输入激活

    # ------------------------------------------------------------ 精度辅助
    def retype(self, x):
        if self.act_dtype == "fp32":
            return x
        if self.act_dtype == "fp16":
            return x.half().float()
        if self.act_dtype == "bf16":
            return x.bfloat16().float()
        raise ValueError(self.act_dtype)

    # ------------------------------------------------------------ 基础算子
    def rmsnorm(self, x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return self.retype((x * torch.rsqrt(v + self.cfg.eps)) * w)

    def linear(self, x, w, b=None, name=None):
        """x:[T,K] @ W.T -> [T,N]; fp32 累加, 输出按 act_dtype 存回。"""
        if self.act_hook is not None and name is not None:
            self.act_hook(name, x)
        y = x.to(torch.float32) @ w.to(torch.float32).t()
        if b is not None:
            y = y + b.to(torch.float32)
        return self.retype(y)

    def rope(self, x, cos, sin):
        x = x.to(torch.float32)
        if x.dim() == 3:            # [T,Hn,D] —— 广播到 head 维
            cos, sin = cos[:, None, :], sin[:, None, :]
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return self.retype(x * cos + rot * sin)

    @staticmethod
    def rope_tables(head_dim, positions, theta):
        half = head_dim // 2
        inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) / half))
        f = torch.tensor(positions, dtype=torch.float64)[:, None] * inv[None, :]
        emb = torch.cat([f, f], dim=-1).to(torch.float32)
        return emb.cos(), emb.sin()

    def attention(self, q, k, v, positions, n_hist=0):
        """q:[T,Hq,D] k/v:[S,Hkv,D] S=T+n_hist。causal: 只允许看 pos<=当前。"""
        c = self.cfg
        T = q.shape[0]
        rep = c.heads // c.kv_heads
        kk = k[:, :, None, :].repeat(1, 1, rep, 1).reshape(k.shape[0], c.heads, c.head_dim)
        vv = v[:, :, None, :].repeat(1, 1, rep, 1).reshape(v.shape[0], c.heads, c.head_dim)
        scores = torch.einsum("thd,shd->hts", q.to(torch.float32), kk.to(torch.float32)) * c.scale
        S = kk.shape[0]
        keys = torch.arange(S)
        qpos = torch.tensor(positions)
        mask = keys[None, :] > qpos[:, None]                     # [T,S]
        if n_hist and T == 1:
            mask = torch.zeros_like(mask)
        scores = scores.masked_fill(mask[None, :, :], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("hts,shd->thd", p, vv.to(torch.float32))
        return self.retype(o), p

    def silu_mul(self, gate, up):
        return self.retype(torch.nn.functional.silu(gate) * up)

    # ------------------------------------------------------------ 前向
    def forward(self, input_ids=None, inputs_embeds=None, past=None, positions=None,
                capture=None, want_probs=False):
        cfg = self.cfg
        st = self.st
        if inputs_embeds is not None:
            h = inputs_embeds.to(torch.float32)
        else:
            idx = torch.as_tensor(input_ids, dtype=torch.long)
            h = st["model.embed_tokens.weight"].to(torch.float32)[idx]
        h = self.retype(h)
        T = h.shape[0]
        if positions is None:
            positions = list(range(T))
        self._positions = positions          # 供故障注入/调试使用
        cos, sin = self.rope_tables(cfg.head_dim, positions, cfg.theta)
        capp = capture if capture is not None else {}
        capp["embed"] = h.clone()
        capp.setdefault("layers", [None] * cfg.layers)

        new_past = []
        for i in range(cfg.layers):
            p = f"model.layers.{i}."
            rec = {}
            res = h
            x = self.rmsnorm(h, st[p + "input_layernorm.weight"])
            rec["ln1"] = x.clone()
            q = self.linear(x, st[p + "self_attn.q_proj.weight"], st[p + "self_attn.q_proj.bias"],
                            name=p + "self_attn.q_proj.weight")
            k = self.linear(x, st[p + "self_attn.k_proj.weight"], st[p + "self_attn.k_proj.bias"],
                            name=p + "self_attn.k_proj.weight")
            v = self.linear(x, st[p + "self_attn.v_proj.weight"], st[p + "self_attn.v_proj.bias"],
                            name=p + "self_attn.v_proj.weight")
            q = self.rope(q.view(T, cfg.heads, cfg.head_dim), cos, sin)
            k = self.rope(k.view(T, cfg.kv_heads, cfg.head_dim), cos, sin)
            v = self.retype(v.view(T, cfg.kv_heads, cfg.head_dim))
            n_hist = 0
            if past is not None and past[i] is not None:
                kp, vp = past[i]
                n_hist = kp.shape[0]
                k = torch.cat([kp, k], dim=0)
                v = torch.cat([vp, v], dim=0)
            new_past.append((k.clone(), v.clone()))
            o, probs = self.attention(q, k, v, positions, n_hist=n_hist)
            rec["q"], rec["k"], rec["v"] = q.clone(), k.clone(), v.clone()
            if want_probs:
                rec["attn_prob"] = probs
            if not want_probs and T <= 8:
                rec["attn_prob"] = probs
            attn = self.linear(o.reshape(T, cfg.hidden), st[p + "self_attn.o_proj.weight"],
                               name=p + "self_attn.o_proj.weight")
            rec["attn_out"] = attn.clone()
            h = self.retype(res + attn)
            res = h
            x = self.rmsnorm(h, st[p + "post_attention_layernorm.weight"])
            gate = self.linear(x, st[p + "mlp.gate_proj.weight"], name=p + "mlp.gate_proj.weight")
            up = self.linear(x, st[p + "mlp.up_proj.weight"], name=p + "mlp.up_proj.weight")
            act = self.silu_mul(gate, up)
            down = self.linear(act, st[p + "mlp.down_proj.weight"], name=p + "mlp.down_proj.weight")
            rec["mlp_out"] = down.clone()
            h = self.retype(res + down)
            rec["out"] = h.clone()
            capp["layers"][i] = rec
        hn = self.rmsnorm(h, st["model.norm.weight"])
        capp["final_norm"] = hn.clone()
        logits = hn.to(torch.float32) @ st["model.embed_tokens.weight"].to(torch.float32).t()
        capp["logits"] = logits.clone()
        return logits, new_past, capp

    # ------------------------------------------------------------ 生成
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=16, capture_first=False):
        """贪心解码; 返回 (生成的 token 列表, 首步 capture, 逐 token 的 top1 logits)。"""
        logits, past, cap = self.forward(input_ids=input_ids, capture={} if capture_first else None)
        out, steps = [], []
        pos = len(input_ids)
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        steps.append({"logits_top5": [(int(v), float(logits[-1][v])) for v in logits[-1].topk(5).indices],
                      "entropy": float(-(torch.softmax(logits[-1], -1) * torch.log_softmax(logits[-1], -1)).sum())})
        for _ in range(max_new_tokens - 1):
            logits, past, _ = self.forward(input_ids=[nxt], past=past, positions=[pos])
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            steps.append({"logits_top5": [(int(v), float(logits[-1][v])) for v in logits[-1].topk(5).indices]})
            pos += 1
        return out, cap, steps
