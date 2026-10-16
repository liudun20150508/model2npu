# 多模态 / 长上下文：先查布局，再查精度

## 症状

- prefill 正常，连续 decode 崩或 NaN；
- 视觉 token 拼进文本序列后结果异常；
- 长上下文（>2K）精度明显下降，短序列看不出来。

## 结论

这三类问题的根因**通常不是数值精度，而是序列布局**：

1. **位置 id**：设备侧序列布局与 HF 不同（本仓库设备布局是 `[196 patch, cls]`，
   HF 是 `[cls, 196 patch]`），因此位置表要按同样的配对关系拼：
   `row i (patch i) ← pos_embed[i+1]`、`row 196 (cls) ← cls_token + pos_embed[0]`；
2. **KV 写入位置**：视觉 token 与文本 token 在 KV Cache 里的位置必须连续且与
   attention mask 一致；写错位置在短序列上常常「看起来还行」；
3. **因果掩码**：视觉编码器是**非因果** attention，文本侧是因果 —— 两者混在同一份微码里
   时必须显式区分。

## 验证方式（本仓库口径）

- `results/s8_vision.md`：视觉编码器整模型上设备后，top-1 与 fp32 参考一致；
- 多模态链路：197 视觉 token + 4 文本 token = 201 序列，跑 prefill + 连续 4 步 decode，
  检查 **KV Cache 峰值**、**logits 是否全有限**、末行 top-1 token id 是否稳定；
- 长上下文：`src/s6_perf.py` 给出 KV 读取反超权重的临界 token 数，作为「是否需要 KV 压缩」
  的量化依据。

## 注意

多模态演示里的 projector 通常是**随机初始化**的（本仓库就是），它验证的是**链路与布局正确性**，
不是语义效果。对外汇报时必须写清楚，否则会被当成「模型效果」。

## 相关文件

- `src/s8_vision.py`、`src/mtx_vit.py`：视觉编码器与视觉 token 投影链路
- `results/s8_vision.md`：201 序列的 prefill/decode 实测
- `knowledge/kb_align_debug.md`：布局类问题的逐层定位方法
