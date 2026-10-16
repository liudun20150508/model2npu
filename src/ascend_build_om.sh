#!/usr/bin/env bash
# Ascend 真机链路 (1/2): ONNX → ATC → .om
#
# 本机 CANN 9.1.0 的 ATC 会去拉 "python3" 来跑 TBE 算子库初始化, 而默认 python3
# (/usr/bin/python3, 3.12) 没有 numpy, 会报 Environment_Error_Import_Python_Module_Failed。
# 把带 numpy 的 python3 放到 PATH 前面即可。
#
# 用法: bash src/ascend_build_om.sh <model.onnx> <输出前缀> [输入shape]
#   bash src/ascend_build_om.sh ../04_推理框架与端侧部署/绿联_edge-infer/data/mobilenetv2-12.onnx \
#        /tmp/npu_smoke/mbv2_910b "input:1,3,224,224"
set -euo pipefail

ONNX=${1:?用法: ascend_build_om.sh <model.onnx> <out_prefix> [input_shape]}
OUT=${2:?缺输出前缀}
SHAPE=${3:-}
PY3=${PY3:-/usr/local/python3.11.15/bin}

if [ ! -d "$PY3" ]; then echo "找不到带 numpy 的 python: $PY3"; exit 2; fi
export PATH="$PY3:$PATH"
export PYTHONPATH="$PY3/../lib/python3.11/site-packages:${PYTHONPATH:-}"

args=(--model="$ONNX" --framework=5 --output="$OUT" --soc_version="${SOC:-Ascend910B3}")
[ -n "$SHAPE" ] && args+=(--input_shape="$SHAPE")
echo "atc ${args[*]}"
atc "${args[@]}"
ls -la "$OUT.om"
