# S8 视觉编码器 (ViT/DeiT) + 视觉 token 多模态链路

## 1. 模型与输入

| 项   | 值                                                          |
|-----|------------------------------------------------------------|
| 模型  | facebook/deit-tiny-patch16-224 (真实预训练权重)                   |
| 结构  | hidden 192 / 12 层 / 3 头 (head_dim 64) / MLP 768 / patch 16 |
| 序列  | 197 token = 14×14 patch + 1 cls                            |
| 输入图 | `assets/demo.jpg` → 224×224 bilinear + ImageNet 归一化        |
| 参数量 | 5.72 M                                                     |

fp32 参考前向 19.59 s。

## 2. 参考结果 (fp32, 真实图片)

| # | 类别                          | logit |
|---|-----------------------------|-------|
| 1 | Samoyed, Samoyede           | 9.194 |
| 2 | Pomeranian                  | 6.932 |
| 3 | West Highland white terrier | 4.891 |
| 4 | keeshond                    | 4.734 |
| 5 | papillon                    | 4.712 |

## 3. 量化 (W4A16 g128 + 分类头 int8) 与 NPU ISS 结果

| 链路                     | top-1 | 类别                | 与 fp32 的 logits cosine | top-5 一致率 |
|------------------------|-------|-------------------|------------------------|-----------|
| fp32 参考                | 258   | Samoyed, Samoyede | 1.000000               | 5/5       |
| 量化模拟 (去量化权重 + fp16 激活) | 258   | Samoyed, Samoyede | 0.826483               | 4/5       |

| # | 类别 (ISS)                    | logit |
|---|-----------------------------|-------|
| 1 | Samoyed, Samoyede           | 8.008 |
| 2 | Pomeranian                  | 6.617 |
| 3 | West Highland white terrier | 5.527 |
| 4 | papillon                    | 4.762 |
| 5 | Angora, Angora rabbit       | 4.527 |

| 对比                       | 值                           |
|--------------------------|-----------------------------|
| ISS vs 量化模拟 cosine       | 0.982297                    |
| ISS vs 量化模拟 最大绝对差        | 0.6594                      |
| ISS vs fp32 cosine       | 0.801098                    |
| top-1 是否一致 (ISS vs fp32) | 是                           |
| ISS 总 cycle              | 2.62 Mcycle (2.62 ms @1GHz) |
| 视觉编码 DRAM 访问             | 43.86 MiB                   |
| 设备镜像                     | 3.76 MiB                    |

| 阶段        | cos vs 量化模拟 | max_abs  |
|-----------|-------------|----------|
| embedding | 0.99999989  | 0.003906 |
| L0        | 0.99999927  | 0.015625 |
| L6        | 0.99999096  | 0.093750 |

逐层能到 6~7 个 9, 而分类头 logits 的 cosine 只有 0.982 —— 因为分类头的输出量级是 hidden 的数倍, 中间层最后几位小数会被放大; 所以判据要同时看「逐层对齐」和「最终 top-1」, 只看其中一个都会误判。

## 4. 视觉 token 进 LLM (多模态链路)

视觉 token 197 个 (ViT 最后一层输出, 已量化) 经过 int4 投影层 [192→896] 进入文本嵌入空间, 与 4 个 prompt token 拼成 201 长度的序列进 prefill。

> 说明: 投影层是**随机初始化**的结构演示 (没有做过对齐训练), 这里验证的是「视觉 token 能不能走通设备链路 + KV Cache 里位置/长度是否正确」, 不是语义效果。

| 阶段      | cycle         | 时间        | DRAM 访问    |
|---------|---------------|-----------|------------|
| decode1 | 9.07 Mcycle   | 9.07 ms   | 372.11 MiB |
| decode2 | 9.07 Mcycle   | 9.07 ms   | 372.13 MiB |
| decode3 | 9.07 Mcycle   | 9.07 ms   | 372.14 MiB |
| decode4 | 9.07 Mcycle   | 9.07 ms   | 372.15 MiB |
| init    | 3.24 Mcycle   | 3.24 ms   | 118.82 MiB |
| prefill | 165.39 Mcycle | 165.39 ms | 2.35 GiB   |

序列长度 201 token (197 视觉 + 4 文本); KV Cache 峰值 106.50 KiB, prefill 之后连续 decode 4 步无越界/无 NaN —— 说明视觉 token 的位置编码、KV 写入与因果掩码在设备侧是对齐的。

多模态 prefill 的 logits: 形状 (201, 151936), 全有限值=True, 最后一行 top-1 token id = 13。
