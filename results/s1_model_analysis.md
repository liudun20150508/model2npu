# S1 模型分析与部署规划 —— Qwen2.5-0.5B-Instruct @ MTX-NPU v0

## 1. 模型结构与参数量

权重构成与不同精度下的存储占用（embedding 与 lm_head 绑定, 只算一份）:

| 模块                         | 参数量      | 占比    | fp16       | int8       | int4       |
|----------------------------|----------|-------|------------|------------|------------|
| MLP (SwiGLU)               | 313.79 M | 63.5% | 598.50 MiB | 299.25 MiB | 149.62 MiB |
| embedding / lm_head (tied) | 136.13 M | 27.6% | 259.66 MiB | 129.83 MiB | 64.91 MiB  |
| attention 投影               | 44.07 M  | 8.9%  | 84.05 MiB  | 42.03 MiB  | 21.01 MiB  |
| RMSNorm                    | 0.04 M   | 0.0%  | 85.75 KiB  | 42.88 KiB  | 21.44 KiB  |
| 合计                         | 494.03 M | 100%  | 942.29 MiB | 471.15 MiB | 235.57 MiB |

## 2. 关键计算梳理: 逐算子 roofline

算力脊点: 峰值 2.05 TFLOP/s (int4 PE 利用率 0.62), DRAM 带宽脊点 AI = 40.0 FLOP/byte。**AI < 脊点 = 访存瓶颈**。

| 场景              | 算子             | T   | N    | K    | MACs      | 访存         | AI     | 脊点 | 瓶颈      | t_compute | t_mem    | t         |
|-----------------|----------------|-----|------|------|-----------|------------|--------|----|---------|-----------|----------|-----------|
| decode (T=1)    | q_proj         | 1   | 896  | 896  | 0.803M    | 395.50 KiB | 3.96   | 40 | memory  | 1.26us    | 9.31us   | 9.31us    |
| decode (T=1)    | k_proj         | 1   | 128  | 896  | 0.115M    | 58.00 KiB  | 3.86   | 40 | memory  | 0.18us    | 1.36us   | 1.36us    |
| decode (T=1)    | v_proj         | 1   | 128  | 896  | 0.115M    | 58.00 KiB  | 3.86   | 40 | memory  | 0.18us    | 1.36us   | 1.36us    |
| decode (T=1)    | o_proj         | 1   | 896  | 896  | 0.803M    | 395.50 KiB | 3.96   | 40 | memory  | 1.26us    | 9.31us   | 9.31us    |
| decode (T=1)    | gate_proj      | 1   | 4864 | 896  | 4.358M    | 2.09 MiB   | 3.98   | 40 | memory  | 6.86us    | 50.34us  | 50.34us   |
| decode (T=1)    | up_proj        | 1   | 4864 | 896  | 4.358M    | 2.09 MiB   | 3.98   | 40 | memory  | 6.86us    | 50.34us  | 50.34us   |
| decode (T=1)    | down_proj      | 1   | 896  | 4864 | 4.358M    | 2.09 MiB   | 3.98   | 40 | memory  | 6.86us    | 50.34us  | 50.34us   |
| decode (T=1)    | QK^T (GQA 广播)  | 1   | 1    | 64   | 0.001M    | 2.03 KiB   | 0.86   | 40 | memory  | 0.00us    | 0.05us   | 0.05us    |
| decode (T=1)    | PV (softmax 后) | 1   | 64   | 1    | 0.001M    | 2.03 KiB   | 0.86   | 40 | memory  | 0.00us    | 0.05us   | 0.05us    |
| prefill (T=256) | q_proj         | 256 | 896  | 896  | 205.521M  | 1.26 MiB   | 311.65 | 40 | compute | 323.72us  | 30.31us  | 323.72us  |
| prefill (T=256) | k_proj         | 256 | 128  | 896  | 29.360M   | 568.00 KiB | 100.96 | 40 | compute | 46.25us   | 13.36us  | 46.25us   |
| prefill (T=256) | v_proj         | 256 | 128  | 896  | 29.360M   | 568.00 KiB | 100.96 | 40 | compute | 46.25us   | 13.36us  | 46.25us   |
| prefill (T=256) | o_proj         | 256 | 896  | 896  | 205.521M  | 1.26 MiB   | 311.65 | 40 | compute | 323.72us  | 30.31us  | 323.72us  |
| prefill (T=256) | gate_proj      | 256 | 4864 | 896  | 1115.685M | 4.89 MiB   | 435.12 | 40 | compute | 1757.32us | 117.84us | 1757.32us |
| prefill (T=256) | up_proj        | 256 | 4864 | 896  | 1115.685M | 4.89 MiB   | 435.12 | 40 | compute | 1757.32us | 117.84us | 1757.32us |
| prefill (T=256) | down_proj      | 256 | 896  | 4864 | 1115.685M | 4.89 MiB   | 435.12 | 40 | compute | 1757.32us | 117.84us | 1757.32us |
| prefill (T=256) | QK^T (GQA 广播)  | 256 | 256  | 64   | 58.720M   | 2.25 MiB   | 49.78  | 40 | compute | 92.49us   | 54.21us  | 92.49us   |
| prefill (T=256) | PV (softmax 后) | 256 | 64   | 256  | 58.720M   | 2.25 MiB   | 49.78  | 40 | compute | 92.49us   | 54.21us  | 92.49us   |

## 3. KV Cache 预算（长上下文的关键约束）

KV 每 token = 2(层) × 24层 × 2 KV头 × 64 head_dim = 6144 元素; int4 权重下模型约 235.57 MiB。

| 上下文   | KV fp16    | KV int8    | 每 token fp16 | 每 token int8 | 占模型(int4)% | 占模型(int8 KV)% |
|-------|------------|------------|--------------|--------------|------------|---------------|
| 1024  | 12.00 MiB  | 6.00 MiB   | 12.00 KiB    | 6.00 KiB     | 5%         | 3%            |
| 4096  | 48.00 MiB  | 24.00 MiB  | 12.00 KiB    | 6.00 KiB     | 20%        | 10%           |
| 8192  | 96.00 MiB  | 48.00 MiB  | 12.00 KiB    | 6.00 KiB     | 41%        | 20%           |
| 32768 | 384.00 MiB | 192.00 MiB | 12.00 KiB    | 6.00 KiB     | 163%       | 82%           |

## 4. 片上 SRAM 分块与内存规划

SRAM 预算 512 KiB（可用, 双缓冲）; T_tile=32。权重常驻 DRAM 流式读取, 分块只缓存一个 N_tile 的权重。

| GEMM    | N    | K    | N_tile | K_tile | N×K tile 数 | SRAM 占用    | 占用率   |
|---------|------|------|--------|--------|------------|------------|-------|
| q/k/v/o | 896  | 896  | 896    | 224    | 1×4        | 448.00 KiB | 87.5% |
| gate/up | 4864 | 896  | 512    | 448    | 10×2       | 408.00 KiB | 79.7% |
| down    | 896  | 4864 | 896    | 256    | 1×19       | 480.00 KiB | 93.8% |

## 5. 算子映射表（模型算子 → NPU ISA / 数据布局 / 精度）

| 模型算子             | NPU 指令       | 数据形状/布局                           | 精度与规则                           | 瓶颈                         |
|------------------|--------------|-----------------------------------|---------------------------------|----------------------------|
| Embedding gather | EMBED        | [T] → [T,H]                       | 查表, 权重 int8/fp16                | DRAM 随机读 896·2B/token      |
| RMSNorm          | RMSNORM      | [T,H] fp16, w[H] fp16             | 方差 fp32 累加, eps=1e-6            | 向量, 384 cycle 固定开销         |
| q/k/v 投影 GEMM    | GEMM4        | W[N,K] int4 g128 打包 / x[T,K] fp16 | y=x·Wᵀ+b, fp32 累加               | prefill 算力瓶颈 / decode 带宽瓶颈 |
| RoPE             | ROPE         | [T,Hn,D] 前后半配对                    | cos/sin 表 fp16, theta=1e6       | 向量, 320 cycle              |
| KV 写入            | KVAPPEND     | cache[S,Hkv,D] 追加                 | 环形/分页地址, 支持 int8 KV             | decode 每步 2·Hkv·D·2B·L     |
| Attention (GQA)  | ATTNGQA      | Q[T,Hq,D], KV[S,Hkv,D]            | Hq/Hkv=7 广播, 因果掩码, softmax fp32 | 长上下文时 KV 读带宽主导             |
| MLP SwiGLU       | SILUMUL      | gate/up[T,I] fp16                 | silu(gate)·up                   | 向量                         |
| 输出投影 GEMM        | GEMM4        | W[H,I] int4 g128                  | fp32 累加                         | decode 带宽瓶颈                |
| LM Head          | GEMM8/GEMM4  | W[V,H] · [T,H]                    | 与 embedding 绑定; 混合精度可留 int8     | decode 最大单算子 (V=151936)    |
| ViT Patch Embed  | IM2COL+GEMM  | img[3,224,224] → [196,192]        | im2col 展开 stride=16             | 算力瓶颈                       |
| ViT Attention    | ATTNGQA(非因果) | [197,192] + cls token             | 无掩码, 全局可见                       | 197 token 全连接              |
| ViT MLP/GELU     | GEMMF + GELU | [197,768]                         | erf 版 GELU                      | 算力瓶颈                       |

## 6. 结论（规划口径）

- decode 单 token: 一层 memory 瓶颈, 整模型 24 层 GEMM 访存 171.77 MiB/token。

- prefill (256 token): 一层 GEMM 以 compute 为主。

- 结论: **decode 用 W4A16 降权重量, prefill 靠分块把 GEMM 拉回算力瓶颈; KV Cache 在 8K 以上必须 int8, 否则 KV 体积会超过 int4 权重**。
