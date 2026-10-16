#!/usr/bin/env bash
# 一键部署: 装依赖自检 → 编译 → 下载模型 → 跑全链路 → 判据裁决。
#   bash deploy.sh              # 完整 (标准档, 12 分钟量级)
#   FAST=1 bash deploy.sh       # 跳过 AWQ/GPTQ, 更快
#   SKIP_MODELS=1 bash deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
FAST=${FAST:-0}

echo "==== 1/5 环境安装与自检 ===="
bash install.sh

echo
echo "==== 2/5 模型 ===="
if [ "${SKIP_MODELS:-0}" = "1" ]; then
  echo "跳过 (SKIP_MODELS=1)"
else
  bash scripts/download_models.sh
fi

echo
echo "==== 3/5 全链路复跑 ===="
if [ "$FAST" = "1" ]; then FAST=1 bash run_all.sh; else bash run_all.sh; fi

echo
echo "==== 4/5 判据总闸口 ===="
set +e
python3 src/verify_all.py
RC=$?
set -e

echo
echo "==== 5/5 结果 ===="
echo "判据总表 : results/ALL_GATES.md"
echo "一页纸结论: results/SUMMARY.md"
echo "原始数据 : results/*.json (含各阶段判据明细)"
if [ "$RC" != "0" ]; then
  echo "!! 存在 FAIL 或过期产物, 不要对外声称全绿 (退出码 $RC)"
  exit "$RC"
fi
echo "全部判据 PASS。"
