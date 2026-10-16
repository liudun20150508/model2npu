# 真机 NPU 验证 (Ascend 910B3)

结论：**真机链路的最小闭环已在本机跑通** —— `ONNX → ATC → .om → pyACL 在 910B3 上执行`，
输出 1000 类 logits。这条链路是 CANN 原生路线，和本 demo 的自研 ISA 微码不是同一套东西。

## 环境

| 项 | 值 |
|---|---|
| 芯片 | Ascend 910B3（`npu-smi info` Health OK，HBM 3673/65536 MB 占用）|
| 驱动 / 工具链 | npu-smi 26.1.1；CANN 9.1.0（`atc`、`pyACL` 均可用）|
| 设备节点 | `/dev/davinci1`、`/dev/devmm_svm`、`/dev/hisi_hdc` |
| 已知坑 1 | ATC 会拉默认 `python3` 跑 TBE 初始化，`/usr/bin/python3`(3.12) 没有 numpy → `Environment_Error_Import_Python_Module_Failed`；把带 numpy 的 python3 放到 PATH 前面即可 |
| 已知坑 2 | `torch 2.11.0` 与 `torch_npu 2.10.0.post4` C++ ABI 不匹配（`undefined symbol: _ZN3c1013MessageLoggerC1EPKciib`），PyTorch 直连 NPU 需换版本配对；本验证走 acl，不受影响 |

## 复现命令

```bash
# 1. 编译 om (9.5 MB, 约 30 s)
bash src/ascend_build_om.sh \
  "../04_推理框架与端侧部署/绿联_edge-infer/data/mobilenetv2-12.onnx" \
  /tmp/npu_smoke/mbv2_910b "input:1,3,224,224"

# 2. 在 910B3 上执行
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 src/ascend_acl_run.py /tmp/npu_smoke/mbv2_910b.om 0
```

## 实测输出（原始日志见 `results/logs/ascend_real_hw.log`）

```
device 0 soc: Ascend910B3
inputs 1 outputs 1
output[0] 1000 floats, top5 idx [701, 405, 557, 21, 417] logits [3.7344, 3.6152, 3.209, 3.1465, 3.1426]
ACL 真机执行 OK
```

输入是 `linspace(-1,1)` 的演示数据（不是真实图片），所以 logits 本身没有语义；
这一步验证的是 **图编译 + 设备加载 + 设备侧执行 + 结果回传** 四段是否通。

## 边界（必须讲清楚）

1. 真机跑的是 **mobilenetv2**（最小闭环），**不是** demo 里的 Qwen2.5-0.5B / DeiT-tiny。
   要把 0.5B 搬上真机，还需要：导出 ONNX → ATC 图编译（采样算子需自写）→ 接 KV Cache 与
   bf16/int8 权重格式，属另一个 1–2 小时的活。
2. demo 的设备侧（`src/s4_iss.cpp` + `hw/mtx_npu_v0.json`）是**自研指令集仿真器**，
   cycle/带宽是按 roofline 建模的，不能当流片数据；它的价值是判据口径与真机一致，
   换后端不改判据。
3. 因此对外表述建议：「设备侧是自研 ISA 指令级仿真（功能 + 周期模型），
   真机闭环在 Ascend 910B3 上验证过（ATC + ACL）」，不要表述成「我们的模型跑在 NPU 上」。
