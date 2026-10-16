# 性能归因：先判 bound 类型，再动优化

## 症状

- TTFT / TPOT 不达标，但不知道先优化哪个算子；
- 换了硬件档位（频率、带宽）参数后，结论大变；
- 报告里性能数字很好看，但落到硅上完全不是这个量级。

## 方法

1. **算 roofline**：每个算子算 `AI = MAC / Byte`，与机器 ridge point 比较
   （本仓库 S1 输出 `results/s1_op_*.csv`，含每个算子 bound 判定与理论时间）；
2. **看瓶颈集中在哪类算子**：本仓库实测 Qwen2.5-0.5B 在端侧档位下，
   `GEMM4` 与 `LMHEAD` 合计占了绝大多数 cycle（`results/s6_perf.md` 的逐算子表）；
3. **L2/DRAM 视角**：decode 每 token 的 DRAM 访存量（本仓库 323 MiB/token）直接决定
   TPOT 下界，权重带宽是端侧的第一约束；
4. **KV Cache 反超点**：长上下文下 KV 读取会超过权重读取成为主导
   （本仓库给出 int8 KV 的反超临界 token 数），这是决定「要不要做 KV 压缩」的量化依据。

## 校准纪律（最容易犯的错）

性能模型里的 `efficiency.*`（PE/DRAM/SRAM/PCIe 效率因子）与 `op_overhead_cycles`
**必须用目标硬件的 micro-benchmark 或 RTL 仿真校准**，否则 cycle 数字只能当量级参考。

本仓库的做法：

- 档位参数全部集中在 `hw/*.json`，报告里显式区分「实测」与「建模」；
- 判据里只对**相对关系**设阈值（如 TTFT ≤ 50 ms 这类工程目标），不宣称与流片一致。

## 相关文件

- `hw/mtx_npu_v0.json`：档位参数（换 json 即换芯片假设）
- `src/s1_analyze_model.py`：逐算子 roofline 与 SRAM 分块规划
- `src/s6_perf.py`：TTFT/TPOT/带宽/存储指标与逐算子归因
- `rules/gates.json`：`perf.*` 判据
