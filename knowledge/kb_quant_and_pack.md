# 量化与权重打包：cosine 高 ≠ 格式对

## 症状

- 量化后 logits cosine 还有 0.9+，但 top-1 已经飘；
- 同一量化算法换一个打包顺序，结果略微变差；
- 仿真（去量化后算）很好，设备（按打包格式解包后算）不行。

## 结论

**必须做解包逐 bit 自检**。本仓库的判据 `pack.bit_exact` 就是把 ISS 解包出来的权重与参考
去量化结果逐 bit 比较（阈值 100%）。只比 cosine 会漏掉：

1. **半字节顺序**：int4 两个数打包在一个字节里，偶列放低 4 bit 还是高 4 bit；
2. **zero-point 类型与偏移**：zp 存 int8 时是否带偏移（对称量化 z=0，非对称 z≠0）；
3. **scale 的 dtype**：scale 存 fp16 还是 fp32，会引入肉眼看不见的偏差；
4. **对齐填充**：每 256 B 对齐后，尾部填充字节若被当成数据读入，会污染最后一组。

这四类错误里,前三类都能让 cosine 依旧很高。

## 量化方案的取舍（本仓库实测口径）

同一模型（Qwen2.5-0.5B-Instruct）同一硬件档位下：

| 方案 | 存储 | logits cosine vs fp32 | 备注 |
|---|---|---|---|
| fp16 | 942 MiB | 0.999993 | 精度基准 |
| W8A16 g128 | 493 MiB | 0.999432 | 精度上限对照 |
| W4A16 RTN g128 | 446 MiB | 0.942823 | 最省事 |
| W4A16 RTN + lm_head int8 | 322 MiB | 0.943227 | 端侧主档 |
| W4A16 AWQ + lm_head int8 | 323 MiB | 0.980592 | 精度收益大 |
| W4A16 GPTQ + lm_head int8 | 322 MiB | 0.964906 | 层输出误差中位降到 RTN 的 0.685 倍 |

要点：

- **分组越细精度越好但元数据越贵**：g32 相比 g128 的精度收益，要用元数据开销衡量；
- **GPTQ 的收益看层输出相对误差**（本仓库逐 tensor 记录在 `results/s3_weight_errors.csv`），
  不要只看最终 cosine；
- **标定集决定 AWQ/GPTQ 上限**：本仓库用 12 条中英混合短文本，生产上应换成上百条领域语料；
  换标定集后所有数字都会变，所以判据文件里不写死「必须超过某绝对值」，而是写「优于同 group RTN」。

## 相关文件

- `src/s3_quant.py`：RTN / AWQ / GPTQ / 混合精度，逐 tensor 误差 CSV
- `src/s3_pack.py`：自定义 int4 打包 + 设备镜像 + 索引
- `src/s4_iss.cpp`：`--dump-dequant` 输出解包结果，供逐 bit 自检
- `rules/gates.json`：`pack.*` / `quant.*` 判据
