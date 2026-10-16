# 迁移到你的 NPU（以迈特芯为例）

本仓库的判据层与硬件无关，只有「性能模型层」和「实现层」需要按目标芯片替换。
下面按改动量从小到大给出三步，每一步都给出验收方式。

## 三步总览

| 步骤 | 改什么 | 文件 | 改动量 | 验收 |
|---|---|---|---|---|
| 1 | 硬件档位参数 | 新建 `hw/mtx_npu_xxx.json` | 30 分钟 | `MTX_HW=hw/xxx.json python3 src/s6_perf.py` 出新的性能画像 |
| 2 | 指令语义 / 数据布局 | `src/s4_iss.cpp`（或替换成自家 ISS/RTL 仿真） | 数天 | `python3 src/s5_e2e.py` 仍 ≥ 0.999 cosine |
| 3 | 真机/仿真后端对接 | `src/ascend_*.sh|py` 换成自家工具链 | 数天 | 同一套 `verify_all.py` 裁决 |

## 第 1 步：换硬件档位（必做，最简单）

复制 `hw/mtx_npu_v0.json`，把里面对应字段换成自家指标：

| 字段 | 含义 | 数据来源 |
|---|---|---|
| `mac_per_cycle` / `pe_array` | 每周期乘累加上限 | 阵列规模 × PE 吞吐（例如 32×32 阵列 = 1024 MAC/cycle） |
| `freq_mhz` | 目标频率 | 时序收敛后的频率目标 |
| `sram_kib` / `sram_usable_kib` | 片上总容量 / 可用容量 | SRAM 实例化报告 |
| `dram.bandwidth_gbps` / `dram.type` | 片外带宽与类型 | LPDDR5/DDR/HBM 规格书 |
| `host_link` | 与 Host 的连接带宽与固定开销 | PCIe Gen3 x4 / USB / SPI 实测 |
| `efficiency.*` | PE / DRAM / SRAM / PCIe 效率因子 | 用自家 micro-benchmark 或 RTL 仿真校准（**这是最需要按芯片改的一项**）|
| `op_overhead_cycles` | 每个算子的固定开销（指令下发、状态机切换） | 逐算子 micro-benchmark |

```bash
MTX_HW=hw/mtx_npu_xxx.json python3 src/s1_analyze_model.py   # 算子 roofline 随之变化
MTX_HW=hw/mtx_npu_xxx.json bash run_all.sh                   # 全链路换档位重跑
```

判据阈值不用改：算力变了只会影响 TTFT/TPOT 这类性能数字，不会改变数值对齐判据。

## 第 2 步：换指令语义与数据布局

自研 ISA 仿真器 `src/s4_iss.cpp` 里，**真正与芯片绑定的只有三件事**：

1. **权重格式**：int4 分组打包的字段顺序（`scale` fp16 / `zp` int8 / 每 256 B 对齐 /
   偶列低半字节）与解包顺序。自家硬件格式不同，就改 `dequant` 相关函数。
2. **指令集与流水开销**：`GEMM4/GEMM8/GEMMF/RMSNORM/LAYERNORM/ROPE/ATTNGQA/SOFTMAX/
   SILUMUL/GELU/ADD/KVAPPEND/EMBED/IM2COL/LMHEAD/SAMPLE/ADDTAB` 等指令的语义与
   周期公式（`op_overhead_cycles` + 效率因子）。
3. **数据布局**：激活用 fp16 存储、GQA 的 KV Cache 布局（`[层][head][seq][dim]`）、
   prefill 与 decode 共用同一份微码模板。

微码模板 `src/s4_gen_program.py`（模型算子 → ISA 指令）与判据脚本**不需要改**。

```bash
g++ -std=c++17 -O3 -march=native -fopenmp -o build/npu_iss src/s4_iss.cpp
python3 src/s4_gen_program.py --variant w4a16_rtn_mix8 --decode 8
python3 src/s5_e2e.py --variant w4a16_rtn_mix8 --decode 8     # 对齐判据仍应 ≥ 0.999
```

如果你已经有自家 RTL/UVM 环境（例如 `05_体系结构与NPU微架构/aipu-k3d4` 那套 SystemVerilog +
Verilator + UVM），可以把它当成后端：本仓库提供的是**输入（量化权重 + 微码/激励 + 期望值）**
与**判据**，RTL 仿真与 ISS 二选一即可。

## 第 3 步：接真机 / 接自家工具链

仓库里的 Ascend 链路演示了「真机最小闭环」应该长什么样，替换成自家工具链时保持同样四段：

```
图编译(ATC/自家编译器) → 设备加载 → 设备执行 → 结果回传
src/ascend_build_om.sh   load_from_file   acl.mdl.execute   D2H memcpy
```

验收：同一份 `verify_all.py` 里的 `e2e.*` 判据在真机数值上重新裁决，
若真机精度与仿真不一致，用 `s7_regress.py` 的故障注入 + 逐层 cosine 曲线定位到具体层。

## 验收清单（交给对方团队时照这张表走）

- [ ] `bash install.sh` 全绿（Python/编译器/模型）
- [ ] `hw/<your_npu>.json` 填完，`efficiency.*` 有 micro-benchmark 校准依据
- [ ] `bash run_all.sh` 12 步全 OK，`python3 src/verify_all.py` 20/20 PASS 且无过期产物
- [ ] `results/s2_ops_alignment.md`：结构算子 + 量化算子用例全部通过
- [ ] `results/s3_pack.md`：自家权重格式解包与参考逐 bit 一致
- [ ] `results/s5_e2e.md`：ISS/真机 vs 量化模拟 cosine ≥ 0.999，生成 token 一致
- [ ] `results/s6_perf.md`：TTFT/TPOT/访存量有明确归因（算力 bound 还是访存 bound）
- [ ] `results/s7_regression.md`：门禁 0 FAIL，故障注入能被定位到注入层
- [ ] 报告的「已知局限」章节写明哪些是实测、哪些是建模

## 最容易踩的三个坑

1. **把建模数字当实测**：`efficiency.*` 与 `op_overhead_cycles` 不校准，性能数字只能当量级参考；
2. **权重格式只对了数值没对 bit**：解包顺序错一半位宽，往往 cosine 还很高（0.9+）但 top-1 已经飘，
   必须用 `s3_pack.py` 的逐 bit 自检兜住；
3. **只跑 prefill 不跑长上下文**：KV Cache 布局错误在短序列上常常看不出来，
   必须跑连续 decode 与长上下文（`s5_e2e.py --decode 8`、`s6_perf.py` 的 KV 临界点）。
