# 迈特芯 · 模型适配与算子验证 Demo —— 汇总

## 0. 一句话结论

在 **MTX-NPU v0 假设档位** 上, 把 Qwen2.5-0.5B 从 fp32 参考 → W4A16 量化权重 (312 MiB 设备镜像) → NPU 指令级仿真跑通 prefill + 连续 decode: 与量化参考的 logits cosine 0.999987, 生成 token 与参考一致 8/8, TTFT 9.36 ms, TPOT 7.89 ms (127 token/s, 单路)。

链路 = 模型分析/部署规划 → 参考算子与 C++ 对齐 → AWQ/GPTQ/RTN 量化与权重打包 → NPU ISS 执行(逐层/逐步对齐) → 视觉编码器与视觉 token → 性能归因 → 回归门禁与故障定位。

## 1. 关键数字

| 维度    | 结果                                                                              | 证据                           |
|-------|---------------------------------------------------------------------------------|------------------------------|
| 模型    | Qwen2.5-0.5B-Instruct, 494.0 M 参数, 24 层 / hidden 896 / GQA 14:2                 | results/s1_model_analysis.md |
| 端到端性能 | TTFT 9.36 ms, TPOT 7.89 ms, 127 token/s, 带宽利用率 323 MiB/token                    | results/s6_perf.md           |
| 量化精度  | w4a16_rtn_mix8: 322.5 MiB (2.92×), logits cosine 0.9432, top-1 一致 71%, 输出一致 25% | results/s3_quant.md          |
| GPTQ  | 同 group_size 下层输出误差降到 RTN 的 0.679 倍 (102 个矩阵, 全部更优: True)                       | results/s3_quant.md          |
| AWQ   | 中位误差比 0.911, 80% 的 tensor 更优                                                    | results/s3_quant.md          |
| 算子对齐  | 12 个结构算子用例 + 13 个量化算子用例, 结构算子 fp32 口径全部 PASS / 量化算子 bit 级一致                     | results/s2_ops_alignment.md  |
| 权重打包  | 自定义格式 build/npu_device_image.bin (312.72 MiB), 元数据开销 4.0%, ISS 解包与参考逐 bit 一致    | results/s3_pack.md           |
| 端到端对齐 | ISS vs 量化模拟 cosine 0.999987, 逐步 argmax 全部一致: True                               | results/s5_e2e.md            |
| 视觉编码器 | DeiT-tiny W4A16, 197 token, ISS top-1 与 fp32 一致, cosine 0.9823                  | results/s8_vision.md         |
| 多模态   | 197 视觉 token + 4 文本 token = 201 序列长度走通设备 prefill/decode                         | results/s8_vision.md         |
| 回归门禁  | 11 PASS / 0 FAIL, 3 类故障注入全部定位到注入层(±1)                                           | results/s7_regression.md     |

## 2. 需求覆盖矩阵

| 需求条目                                 | 本 demo 对应证据                                                                                            | 结果文件                                         | 代码                                                   |
|--------------------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------|
| 职责1 模型分析与部署规划                        | Transformer/ViT 结构梳理, GEMM/Attention/GQA/MLP/Norm/RoPE/KV/RoPE 逐算子 roofline, 算子映射表, SRAM 分块与 KV 预算     | results/s1_model_analysis.md                 | src/s1_analyze_model.py                              |
| 职责2 算子适配与验证                          | 参考算子(Python) ↔ C++ native ↔ PyTorch 真值三方对齐; 形状/精度/边界条件; prefill/连续 decode/长上下文 2048 用例                 | results/s2_ops_alignment.md                  | src/s2_align.py, src/s2_ops_native.cpp               |
| 职责3 量化与精度                            | RTN / AWQ / GPTQ, 对称与非对称, 分组 g32/g128, fp16/INT8/INT4 与混合精度(lm_head int8), 权重打包格式导出, logits/生成结果判据     | results/s3_quant.md, results/s3_pack.md      | src/s3_quant.py, src/s3_pack.py                      |
| 职责4 FPGA/ASIC 端到端集成                  | 设备镜像加载(PCIe 计数) → 微码 → NPU ISS 执行 → 逐层/逐步回传对齐; 从单算子验证走到整模型 + 视觉编码器                                     | results/s5_e2e.md, results/npu_program_*.mtx | src/s4_iss.cpp, src/s4_gen_program.py, src/s5_e2e.py |
| 职责5 性能分析与自动化                         | TTFT/TPOT/吞吐/带宽/存储指标, 逐算子 cycle 与瓶颈归因(算力 vs 访存), 长上下文 KV 反超权重的定量临界点, 回归门禁 + 故障注入定位 + 可重复性              | results/s6_perf.md, results/s7_regression.md | src/s6_perf.py, src/s7_regress.py                    |
| 加分项·端侧部署经验                           | W4A16 全链路 + 设备侧数据布局/精度/内存规划, 量化档位统一在                                                                   | 本目录                                          | 整包                                                   |
| 加分项·AWQ/GPTQ/W4A16/权重打包              | AWQ 逐通道缩放搜索 + GPTQ 误差补偿 + int4 打包(偶列低半字节) + scale fp16 / zp int8 / 256B 对齐的自定义设备格式                     | results/s3_quant.md, results/s3_pack.md      | src/s3_quant.py, src/s3_pack.py, src/s4_iss.cpp      |
| 加分项·TVM/llama.cpp/vLLM/ONNX/TensorRT | 本 demo 自带 C++/ISA 堆栈; 框架侧证据见同级目录 04_推理框架与端侧部署 (llama.cpp 多模态改动 / TVM RISC-V codegen / vLLM DCU 部署)     | ../04_推理框架与端侧部署                              | -                                                    |
| 加分项·PCIe-DMA/DDR/RTL/FPGA/bring-up   | PCIe 传输计数 + DMA 建模 + 声明式 SRAM 规划; RTL 侧证据见 05_体系结构与NPU微架构/aipu-k3d4 (SystemVerilog + Verilator/UVM 仿真) | src/s4_iss.cpp                               | ../05_体系结构与NPU微架构                                    |
| 加分项·多模态/视觉 token/长上下文/连续 decode      | DeiT-tiny 视觉编码器整模型上 NPU + 视觉 token 投影进 LLM 的 201 token prefill 与连续 decode; 长上下文 KV 预算与临界点              | results/s8_vision.md, results/s6_perf.md     | src/s8_vision.py                                     |
| 工程目标·可复现验证链路                         | run_all.sh 一条命令跑通: 参考输出 → 量化权重 → 硬件运行结果 → 性能数据, 全部产出在 results/ 且带判据                                    | results/SUMMARY.md                           | run_all.sh                                           |

## 3. 复现方式

```bash
# 标准档 (含 AWQ/GPTQ 采样)
bash run_all.sh

# 快速档 (跳过 AWQ/GPTQ, 少几步 decode)
FAST=1 bash run_all.sh

# 换量化档位
VARIANT=w4a16_awq_mix8 bash run_all.sh

# 全量 GPTQ (慢)
GPTQ=all bash run_all.sh
```

## 4. 已知局限 (别当成芯片实测)

- **NPU ISS 是「功能 + 周期」模型**, 不是 RTL, 更不是硅: 功能上按硬件数据布局真算一遍(int4 解包/去量化/分组累加/fp16 激活存储), 所以能和参考对拍; 但 cycle/带宽是按 roofline + 效率因子**建模**出来的, 用于瓶颈归因和规划, 不能当流片数据。

- 硬件档位 (1024 MAC/cycle @1GHz, LPDDR5 51.2 GB/s, 512 KiB 可用 SRAM, PCIe Gen3 x4) 是**假设的目标档位**, 在 `hw/mtx_npu_v0.json` 里, 换档位重跑即可得到新的性能画像。

- 多模态里的 projector 是随机初始化的结构演示 (未训练), 验证的是链路与位置/长度正确性, 不是语义效果; 文本侧用贪心解码, 未做采样温度/重复惩罚对比。

- AWQ/GPTQ 的标定集只有 12 条短文本 (CPU 时间约束), 生产上要用上百条领域语料; GPTQ 默认只对采样层做 (全量在 `GPTQ=all`)。

- 参考实现 (Qwen2/ViT) 是手写展开的, 已和 transformers 的行为对齐 (生成文本可读、分类正确), 但没做全量 logits 逐元素比对。
