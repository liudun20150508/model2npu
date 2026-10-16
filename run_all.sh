#!/usr/bin/env bash
# 一键跑通「模型参考输出 → 量化权重 → NPU 运行结果 → 性能数据」的可复现验证链路。
#
#   bash run_all.sh              # 标准档 (含 AWQ / GPTQ 采样)
#   FAST=1 bash run_all.sh       # 快速档 (跳过 AWQ/GPTQ, 少跑几步 decode)
#   GPTQ=all bash run_all.sh     # 全量 GPTQ (慢, 30 分钟量级)
#   VARIANT=w4a16_awq_mix8 bash run_all.sh
set -u
cd "$(dirname "$0")"
ROOT=$(pwd)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}

# --- 找带 numpy/torch 的 Python (本机默认 python3 可能没有) ---
if [ -n "${PY:-}" ]; then :; else
  for c in python3 /usr/local/python3.11.15/bin/python /usr/local/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import numpy, torch" >/dev/null 2>&1; then
      PY="$c"; break
    fi
  done
fi
if [ -z "${PY:-}" ]; then echo "找不到带 numpy+torch 的 python (可用 PY=... 指定)"; exit 2; fi

FAST=${FAST:-0}
GPTQ=${GPTQ:-sample}
VARIANT=${VARIANT:-w4a16_rtn_mix8}
DECODE=${DECODE:-8}
if [ "$FAST" = "1" ]; then GPTQ=none; DECODE=4; fi
mkdir -p results/logs build
LOG=results/logs/run_all.log
echo "python : $PY" | tee "$LOG"
echo "variant: $VARIANT   gptq: $GPTQ   decode: $DECODE" | tee -a "$LOG"

step () {           # step <名字> <命令...>
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "==================== [$name] $(date +%H:%M:%S) ====================" | tee -a "$LOG"
  local t0=$SECONDS
  if "$@" >>"$LOG" 2>&1; then
    echo "  OK   $name  ($((SECONDS - t0)) s)" | tee -a "$LOG"
  else
    echo "  FAIL $name  (见 $LOG)" | tee -a "$LOG"
    FAILED="${FAILED:-} $name"
  fi
}

# ---------- 0. 环境与模型 ----------
step "00 环境检查" "$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import os, sys
sys.path.insert(0, "src")
import mtx_common as C
print("ROOT", C.ROOT)
print("型号目录:", os.path.split(C.QWEN_DIR)[1], "存在" if os.path.exists(C.QWEN_DIR) else "缺失")
t = os.path.join(C.QWEN_DIR, "model.safetensors")
print("权重:", C.human_bytes(os.path.getsize(t)) if os.path.exists(t) else "缺失")
print("视觉编码器:", "存在" if os.path.isdir(C.VIT_DIR) else "缺失")
missing = [p for p in (C.QWEN_DIR, C.VIT_DIR) if not os.path.isdir(p)]
if missing:
    print("先下载模型: HF_ENDPOINT=https://hf-mirror.com python -c \"from huggingface_hub import snapshot_download; "
          "snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen2.5-0.5b-instruct')\"")
EOF

# ---------- 1. 模型分析与部署规划 ----------
step "S1 模型分析/部署规划" "$PY" src/s1_analyze_model.py

# ---------- 2. 算子适配与验证 ----------
step "S2 算子对齐 (含 C++ native)" "$PY" src/s2_align.py

# ---------- 3. 量化 / 打包 / 设备镜像 ----------
step "S4 编译 NPU ISS" g++ -std=c++17 -O3 -march=native -fopenmp -o build/npu_iss src/s4_iss.cpp
step "S3 量化 (RTN/AWQ/GPTQ)" "$PY" src/s3_quant.py --gptq "$GPTQ"
step "S3b 设备镜像打包" "$PY" src/s3_pack.py --variant "$VARIANT"
step "S3b 权重格式 bit 级自检" ./build/npu_iss --dump-dequant build/deq \
     --image build/npu_device_image.bin --idx build/npu_device_image.idx \
     --hw hw/mtx_npu_v0.json

# ---------- 4. 端到端: 参考 → 量化 → NPU ----------
step "S5 端到端三端对齐" "$PY" src/s5_e2e.py --variant "$VARIANT" --decode "$DECODE"
step "S6 性能分析与瓶颈归因" "$PY" src/s6_perf.py --variant "$VARIANT"

# ---------- 5. 视觉编码器 + 多模态 ----------
step "S8 ViT 视觉编码 + 视觉 token" "$PY" src/s8_vision.py

# ---------- 6. 回归门禁与问题定位 ----------
step "S7 回归门禁 + 故障定位" "$PY" src/s7_regress.py --variant "$VARIANT" --allow-fail

# ---------- 7. 汇总 ----------
step "S9 汇总报告" "$PY" src/s9_report.py --variant "$VARIANT"

echo "" | tee -a "$LOG"
if [ -n "${FAILED:-}" ]; then
  echo "有步骤失败:${FAILED}   详见 $LOG" | tee -a "$LOG"
else
  echo "全部步骤完成。汇总: results/SUMMARY.md" | tee -a "$LOG"
fi
