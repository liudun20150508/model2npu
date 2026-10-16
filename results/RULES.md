# 判据裁决总表 (规则引擎)

- 规则文件: `rules/gates.json` (v1.0)
- 生成时间: 2026-10-16 05:39:09
- 结果: **22/22 PASS**; error 级失败 0, warn 级 0, 过期/缺失产物 0

| # | 判据 | 阶段 | 严重度 | 说明 | 实测 | 门槛 | 判定 |
|---|------|------|--------|------|------|------|------|
| 1 | `ops.struct` | S2 | error | 结构算子逐用例数值对齐 (Python↔C++↔torch) | 12 | 全部用例 passed | PASS |
| 2 | `ops.struct.bit_exact` | S2 | error | 位级用例 (fp16 舍入/边界/次正规) 逐 bit 一致率 | 100 | >= 100.0 | PASS |
| 3 | `ops.quant` | S2 | error | 量化算子 (INT4/INT8 分组去量化) 数值对齐 | 13 | 全部 passed 且 max rel_err <= 1e-5 | PASS |
| 4 | `quant.variants` | S3 | error | 量化方案覆盖度 (含 AWQ / GPTQ / 混合精度) | 6 | >= 4 档且含 awq / gptq | PASS |
| 5 | `quant.gptq_gain` | S3 | error | GPTQ 相对同 group RTN 的层输出误差比 (全部矩阵应更优) | 0.6795 | 全部更优, 中位 < 1.0 | PASS |
| 6 | `quant.awq_gain` | S3 | warn | AWQ 相对 RTN 的层输出误差比 | 0.911 | 中位 < 1.0 且更优占比 >= 60% | PASS |
| 7 | `pack.bit_exact` | S3b | error | 硬件权重格式解包逐 bit 一致 (ISS dump vs 参考) | 100 | >= 100.0 | PASS |
| 8 | `pack.image` | S3b | error | 设备镜像导出与元数据开销 | 327913472 | 元数据开销 < 10% | PASS |
| 9 | `e2e.logits_cosine_bc` | S5 | error | ISS vs 量化模拟 (prefill logits cosine) | 0.999987 | >= 0.999 | PASS |
| 10 | `e2e.gen_match_bc` | S5 | error | ISS 与量化模拟的生成 token 一致率 | 8/8 | 全部一致 | PASS |
| 11 | `e2e.logits_cosine_ab` | S5 | error | 量化 vs fp32 参考 (logits cosine) | 0.943227 | >= 0.9 | PASS |
| 12 | `e2e.top1_ab` | S5 | warn | 量化 vs fp32 参考 (top-1 一致率) | 0.714286 | >= 0.6 | PASS |
| 13 | `perf.ttft_ms` | S6 | error | 首 token 延迟 TTFT | 9.361978 | <= 50.0 | PASS |
| 14 | `perf.tpot_ms` | S6 | error | 单 token 延迟 TPOT | 7.891218 | <= 15.0 | PASS |
| 15 | `perf.dram_per_token` | S6 | info | decode 每 token DRAM 访存量 (瓶颈归因输入) | 338819204 | > 0 | PASS |
| 16 | `regress.all_pass` | S7 | error | 门禁回归明细 (gates 字典逐条) | 0 | == 0 | PASS |
| 17 | `repeat.deterministic` | S7 | error | 同一微码两次运行逐 bit 一致 | 1 | == 1.0 | PASS |
| 18 | `debug.localize` | S7 | error | 故障注入能否定位到注入层 (跨层/跨算子定位能力) | 3 | 全部定位正确 | PASS |
| 19 | `vision.align` | S8 | error | 视觉编码器 ISS vs 量化模拟 (logits cosine) | 0.982297 | >= 0.95 | PASS |
| 20 | `vision.top1` | S8 | error | 视觉编码器 top-1 与 fp32 参考一致 | True | == True | PASS |
| 21 | `vision.multimodal` | S8 | error | 视觉 token 进 LLM: logits 健全性 + KV Cache 无越界 | 109056 | 全有限且序列长度/ KV 合理 | PASS |
| 22 | `realhw.acl_exec` | 真机 | warn | 真机 Ascend: .om 在设备上执行成功 (日志留证) | 1 | 存在真机执行日志 | PASS |

## 产物新鲜度（结果是否由当前代码产生）

| 阶段 | 产出 | 状态 | 备注 |
|------|------|------|------|
| S1 模型分析与部署规划 | `results/s1_model_analysis.json` | OK | 产出晚于依赖源码 |
| S2 算子适配与验证 | `results/s2_ops_alignment.json` | OK | 产出晚于依赖源码 |
| S3 量化与精度优化 | `results/s3_quant.json` | OK | 产出晚于依赖源码 |
| S3b 权重打包与镜像 | `results/s3_pack.json` | OK | 产出晚于依赖源码 |
| S4a 微码生成 | `build/npu_program_w4a16_rtn_mix8_kvfp16.mtx` | OK | 产出晚于依赖源码 |
| S4b ISS 可执行 | `build/npu_iss` | OK | 产出晚于依赖源码 |
| S5 端到端集成与数值对齐 | `results/s5_e2e.json` | OK | 产出晚于依赖源码 |
| S6 性能分析与瓶颈归因 | `results/s6_perf.json` | OK | 产出晚于依赖源码 |
| S7 回归门禁与问题定位 | `results/s7_regression.json` | OK | 产出晚于依赖源码 |
| S8 视觉编码器与多模态链路 | `results/s8_vision.json` | OK | 产出晚于依赖源码 |
| S9 汇总报告 | `results/SUMMARY.md` | OK | 产出晚于依赖源码 |

> error 级失败与过期产物会让退出码非 0，可直接挂 CI/门禁；
> warn 级只提示，不阻断。修改阈值请编辑 `rules/gates.json`，不要改脚本。
