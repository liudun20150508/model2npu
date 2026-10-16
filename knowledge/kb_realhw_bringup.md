# 真机 bring-up：四段分开验，环境坑先于算法坑

## 方法：把真机链路拆成四段，逐段验证

| 段 | 做什么 | 本仓库对应 | 通过标志 |
|---|---|---|---|
| 1 图编译 | `ONNX → .om` | `src/ascend_build_om.sh`（ATC） | 产出 `.om` 且大小合理 |
| 2 设备加载 | 加载模型到设备 | `acl.mdl.load_from_file` | 返回非空 model_id |
| 3 设备执行 | 执行一次前向 | `acl.mdl.execute` | 返回码 0 |
| 4 结果回传 | 输出搬回 Host | `acl.rt.memcpy(D2H)` | 输出张量形状/数量正确 |

先证明**最小闭环**通（例如 mobilenetv2 这种小模型），再上目标大模型。本仓库实测：
ATC 编译约 30 s 产出 9.5 MB `.om`，910B3 上执行 4 s，输出 1000 类 logits。

## 环境坑（先于算法坑，实测踩过）

1. **图编译器依赖 Python 环境**：ATC 会拉 PATH 里的 `python3` 去初始化算子库，
   若该解释器没有 numpy，会报 `Environment_Error_Import_Python_Module_Failed`，
   表现为「编译失败」但其实与模型无关 —— 把带 numpy 的 python3 前置到 PATH 即可；
2. **框架与硬件插件版本必须配对**：本机 `torch 2.11.0` 与 `torch_npu 2.10.0.post4`
   ABI 不匹配，`import torch_npu` 直接报 `undefined symbol`。
   绕开方式：走 CANN 原生 `acl`（本仓库做法），不经过 PyTorch；
3. **输入 shape 含 -1**：onnx 里动态维度要在编译时显式指定 `--input_shape`；
4. **设备可见性**：容器里设备节点（如 `/dev/davinci1`）与 `device_id` 的映射要对齐，
   用 `npu-smi info` 与 `acl.get_soc_name()` 双向确认。

## 与仿真链路的关系

真机验证的是「图编译 + 设备执行」的真实性；ISA 仿真验证的是「算子数值 + 性能模型」的可解释性。
两者共用同一套判据（`rules/gates.json`），所以换后端不用改判据 —— 这正是本框架的核心设计。

## 相关文件

- `src/ascend_build_om.sh`、`src/ascend_acl_run.py`
- `results/npu_real_hw.md`、`results/logs/ascend_real_hw.log`
- `docs/ENVIRONMENT.md`：本机完整环境与耗时记录
