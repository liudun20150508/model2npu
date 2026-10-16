# 全链路判据总表 (需求 覆盖自查)

- 生成时间: 2026-10-16 05:23:07
- 结果: **18 PASS / 0 FAIL**; 产物新鲜度: 8/10 由当前代码生成

## 1. 判据

| # | 判据 | 需求关键词 | 说明 | 实测 | 门槛 | 判定 |
|---|------|----------|------|------|------|------|
| 1 | `ops.struct` | 行为面 参考算子 | 结构算子逐用例对齐 (12 例) | 12/12 通过, 位级用例最低 100.0% | 全部通过且位级 100% | PASS |
| 2 | `ops.quant` | INT4/INT8 量化算子 | 量化算子数值对齐 (13 例) | 13/13 通过, 最大 rel_err 7.56e-06, 最大 ULP 误差 0.227, 其中位级完全一致 9/13 例 (其余为累加顺序导致的 ULP 级差异) | 全部用例通过且 rel_err <= 1e-5 | PASS |
| 3 | `quant.variants` | FP16/BF16/INT8/INT4/AWQ/GPTQ | 量化方案覆盖 (含 AWQ/GPTQ/混合精度) | 6 档: fp16, w4a16_awq_mix8, w4a16_gptq_mix8, w4a16_rtn_g128, w4a16_rtn_mix8, w8a16_g128 | >= 4 档且含 AWQ+GPTQ+混合精度 | PASS |
| 4 | `pack.image` | 硬件权重格式导出 | 设备镜像导出 (权重 + 元数据 + 索引) | 312.72 MiB, 290 个张量, 元数据占比 4.01%, 变体 w4a16_rtn_mix8 | 镜像可导出且元数据开销 < 10% | PASS |
| 5 | `e2e.logits_cosine_bc` | logits 对比 | ISS 输出 vs 量化模拟 (prefill logits cosine) | 0.999987 | >= 0.999 | PASS |
| 6 | `e2e.decode_cosine_min` | 连续 decode | ISS vs 量化模拟, 各 decode 步 cosine 最小值 | 0.999987 | >= 0.999 | PASS |
| 7 | `e2e.gen_match_bc` | 生成结果对齐 | ISS 与量化模拟的生成 token 一致率 | 8/8 | >= 0.9 | PASS |
| 8 | `e2e.logits_cosine_ab` | 量化精度评估 | 量化 vs fp32 参考 (logits cosine) | 0.9432 | >= 0.9 | PASS |
| 9 | `e2e.top1_ab` | 量化精度评估 | 量化 vs fp32 参考 (top-1 一致率) | 0.7143 | >= 0.6 | PASS |
| 10 | `perf.ttft_ms` | TTFT | 首 token 延迟 | 9.36 ms | <= 50 ms | PASS |
| 11 | `perf.tpot_ms` | 吞吐 / TPOT | 单 token 延迟 | 7.89 ms (126.7 tok/s) | <= 15 ms | PASS |
| 12 | `perf.dram_per_token` | 访存量 / 带宽 | decode 每 token DRAM 访存 | 323.12 MiB | 有量化指标 (定性) | PASS |
| 13 | `regress.all_pass` | 回归判据 | 门禁回归明细 (11 条判据) | 11 PASS / 0 FAIL | 0 FAIL | PASS |
| 14 | `repeat.deterministic` | 可重复性 | 同一微码两次运行逐 bit 一致 | 1.0 | == 1.0 | PASS |
| 15 | `debug.localize` | 跨算子/系统链路定位问题 | 故障注入 -> 逐层误差曲线 -> 定位层 | 3/3 注入被定位到正确层 | 全部定位正确 | PASS |
| 16 | `vision.align` | ViT / 视觉编码器 | 视觉编码器 ISS vs 量化模拟 (logits cosine) | 0.982297 | >= 0.95 | PASS |
| 17 | `vision.top1` | ViT / 视觉编码器 | 视觉编码器 top-1 与 fp32 参考一致 | 一致 | 一致 | PASS |
| 18 | `vision.multimodal` | 多模态融合 / 视觉 token / KV Cache | 视觉 token 进 LLM: prefill + 连续 decode 数值健全性 | 序列 201 token, logits 全有限 True, KV 峰值 106.50 KiB | 全有限且无越界 | PASS |

## 2. 产物新鲜度 (结果是不是当前代码跑出来的)

| 阶段 | 产出 | 状态 | 备注 |
|------|------|------|------|
| S1 模型分析与部署规划 | `results/s1_model_analysis.json` | OK | 产出晚于所有依赖源码 |
| S2 算子适配与验证 | `results/s2_ops_alignment.json` | OK | 产出晚于所有依赖源码 |
| S3 量化与精度优化 | `results/s3_quant.json` | OK | 产出晚于所有依赖源码 |
| S3b 硬件权重格式导出 | `results/s3_pack.json` | **过期** | 源码更新于产出之后: s3_quant.py |
| S4b 微码生成 | `build/npu_program_w4a16_rtn_mix8_kvfp16.mtx` | OK | 产出晚于所有依赖源码 |
| S4a ISS 模拟器 | `build/npu_iss` | OK | 产出晚于所有依赖源码 |
| S5 端到端集成与数值对齐 | `results/s5_e2e.json` | OK | 产出晚于所有依赖源码 |
| S6 性能分析与瓶颈归因 | `results/s6_perf.json` | OK | 产出晚于所有依赖源码 |
| S7 回归判据与问题定位 | `results/s7_regression.json` | OK | 产出晚于所有依赖源码 |
| S8 视觉编码器 + 多模态链路 | `results/s8_vision.json` | **过期** | 源码更新于产出之后: s3_quant.py |

> 任一 FAIL 或过期项都会让 `verify_all.py` 返回非 0, 可直接挂 CI。
