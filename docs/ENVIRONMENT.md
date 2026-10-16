# 实测环境与复现记录

本仓库所有数字都在下面这台机器上跑出来，`results/logs/` 保留了运行日志。

## 硬件

| 项 | 值 |
|---|---|
| NPU | Ascend 910B3 × 1 chip（NPU id 1，Bus-Id `0000:C2:00.0`）；HBM 64 GB（实测占用 3673 MB）；Health OK；实测功耗 ~92 W，温度 53 °C |
| CPU | Kunpeng-920，aarch64，256 逻辑核，1 thread/core，主频上限 3.0 GHz（实测编译/量化阶段满载 64~127 线程） |
| 内存 | 502 GB total（量化阶段单进程 RSS 峰值 ~10.5 GB） |
| 磁盘 | 300 GB overlay（仓库 + 模型 + 中间产物约 30 GB，其中 fp16 权重包单个 1.4 GB） |

## 软件

| 项 | 值 | 采集命令 |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) | `cat /etc/os-release` |
| 内核 | 5.10.0-216.0.0.115.oe2203sp4.aarch64 | `uname -r` |
| NPU 驱动 | 26.1.1（ascendhal 7.35.23） | `npu-smi info` / `cat /usr/local/Ascend/driver/version.info` |
| CANN | 9.1.0（`atc`、`pyACL` 可用） | `ls /usr/local/Ascend/` |
| 编译器 | g++ (Ubuntu 13.3.0) 13.3.0 | `g++ --version` |
| Python | 3.11.15 | `python3 -V` |
| 框架 | torch 2.11.0 / torch_npu 2.10.0.post4 / numpy 2.3.5 | `pip list` |

> ⚠️ torch 2.11.0 与 torch_npu 2.10.0.post4 的 C++ ABI 不匹配：
> `import torch_npu` 报 `undefined symbol: _ZN3c1013MessageLoggerC1EPKciib`。
> 因此真机验证走 CANN 原生 `acl`（`src/ascend_acl_run.py`），不经过 PyTorch。

## 复现记录（2026-10-16）

| 步骤 | 耗时 | 备注 |
|---|---|---|
| `bash run_all.sh` 全链路 12 步 | **12.5 分钟** | 其中最慢 S3 量化 601 s |
| 其中 S1 模型分析 | 15 s | 494.0M 参数逐算子 roofline |
| 其中 S2 算子对齐（含 C++ native） | 37 s | 12 + 13 个用例 |
| 其中 S5 端到端三端对齐（decode 8） | 27 s | cosine 0.999987 |
| 其中 S8 视觉 + 多模态 | 17 s | 201 token 序列 |
| 其中 S7 回归门禁 + 故障定位 | 28 s | 11 PASS / 0 FAIL |
| `python3 src/verify_all.py` | 秒级 | 20/20 PASS，无过期产物 |
| 真机：`atc` 编译 mobilenetv2 | ~30 s | 产出 9.5 MB `.om` |
| 真机：`acl` 执行 `.om`（910B3） | 4 s | 输出 1000 类 logits |
| `GPTQ=all` 全量量化（可选） | 28 分钟 | 168 个矩阵全部优于 RTN，中位 0.661 |

## 已知环境坑

1. **ATC 的 TBE 初始化需要 numpy**：ATC 会调用 PATH 里的 `python3`
   （本机 `/usr/bin/python3` 是 3.12 且无 numpy）→ 报
   `Environment_Error_Import_Python_Module_Failed`。
   解决：把带 numpy 的 python3 前置到 PATH（见 `src/ascend_build_om.sh`）。
2. **默认 python3 可能没有 torch/numpy**：本机真正的解释器是
   `/usr/local/python3.11.15/bin/python`；`install.sh` 会自动挑带依赖的那个。
3. **单进程内存**：`s3_quant.py --gptq all` 会先把 fp16 权重包读进内存
   （RSS ~10.5 GB），内存小于 16 GB 的机器建议用 `--gptq sample` 或 `FAST=1`。
