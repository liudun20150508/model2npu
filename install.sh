#!/usr/bin/env bash
# model2npu 环境安装 / 自检: 检查 Python 依赖 + 编译器, 编译 ISS 与 C++ 参考算子, 跑一次冒烟。
#   bash install.sh          # 完整安装(编译 + 冒烟)
#   bash install.sh --check  # 只做环境检查, 不编译
set -u
cd "$(dirname "$0")"
ROOT=$(pwd)
# torch_npu 与 torch 版本不配对时, import torch 会自动加载 torch_npu 并报 ABI 错误;
# 本框架不需要 torch 的 NPU 后端, 关掉自动加载即可 (真机链路走 CANN 原生 acl)。
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1
FAILED=""

say ()  { printf "%s\n" "$*"; }
ok ()   { printf "  [OK]   %s\n" "$*"; }
bad ()  { printf "  [FAIL] %s\n" "$*"; FAILED="$FAILED $1"; }

say "== 1. Python =="
PY=""
for c in "${PY:-}" python3 /usr/local/python3.11.15/bin/python /usr/bin/python3; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c "import numpy, torch, tokenizers, PIL" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  bad "Python"
  say "     需要 numpy / torch / tokenizers / pillow 都能 import 的 Python >= 3.10"
  say "     安装: python3 -m pip install -r requirements.txt"
else
  ok "Python: $PY ($("$PY" -V 2>&1))"
  "$PY" - <<'EOF'
import numpy, torch, tokenizers, PIL
print("  [OK]   numpy %s / torch %s / tokenizers %s / pillow %s"
      % (numpy.__version__, torch.__version__, tokenizers.__version__, PIL.__version__))
EOF
fi

say "== 2. C++ 编译器 =="
if command -v g++ >/dev/null 2>&1; then
  ok "g++: $(g++ --version | head -1)"
else
  bad "g++"; say "     apt-get install -y g++  (需要 C++17, 支持 -fopenmp)"
fi

say "== 3. 模型 =="
for d in qwen2.5-0.5b-instruct deit-tiny-patch16-224; do
  if [ -d "models/$d" ]; then ok "models/$d ($(du -sh "models/$d" | cut -f1))"
  else bad "$d"; say "     缺失 -> bash scripts/download_models.sh"; fi
done
if [ -f "assets/demo.jpg" ]; then ok "assets/demo.jpg (ViT/多模态阶段的示例图)"
else bad "demo.jpg"; say "     S8 需要一张示例图: 放一张 224x224 以上的 RGB 图片到 assets/demo.jpg"; fi

say "== 4. 可选: Ascend 真机链路 =="
if command -v atc >/dev/null 2>&1; then ok "atc: $(command -v atc)"
else say "  [SKIP] 未装 CANN (只用自研 ISA 仿真时不需要)"; fi
if command -v npu-smi >/dev/null 2>&1; then
  ok "npu-smi: $(npu-smi info 2>/dev/null | sed -n '5p' | tr -s ' ' | cut -d'|' -f2-3 | tr -d ' ')"
else say "  [SKIP] 未检测到 npu-smi"; fi

if [ "$CHECK_ONLY" = "1" ]; then
  say ""; say "只做检查: 未编译。"
else
  say "== 5. 编译 =="
  mkdir -p build
  if g++ -std=c++17 -O3 -march=native -fopenmp -o build/npu_iss src/s4_iss.cpp 2>build/iss_build.log; then
    ok "自研 ISA 仿真器 build/npu_iss ($(du -h build/npu_iss | cut -f1))"
  else
    bad "ISS 编译"; tail -20 build/iss_build.log
  fi
  if g++ -std=c++17 -O2 -fopenmp -o build/s2_ops_native src/s2_ops_native.cpp 2>build/s2_build.log; then
    ok "C++ 参考算子 build/s2_ops_native"
  else
    bad "s2 native 编译"; tail -20 build/s2_build.log
  fi

  say "== 6. 冒烟: 算子对齐 (不需要模型) =="
  if [ -n "$PY" ] && "$PY" src/s2_align.py >build/smoke_s2.log 2>&1; then
    ok "s2 算子对齐: $(grep -c '"passed": true' results/s2_ops_alignment.json 2>/dev/null) 个用例通过"
  else
    bad "冒烟"; tail -20 build/smoke_s2.log
  fi
fi

say ""
if [ -n "$FAILED" ]; then
  say "有未通过项:$FAILED —— 修好后重跑 bash install.sh"
  exit 1
fi
say "环境就绪。下一步:"
say "  bash scripts/download_models.sh     # 若模型缺失"
say "  bash run_all.sh                     # 全链路 (12 分钟量级), 快速档 FAST=1 bash run_all.sh"
say "  python3 src/verify_all.py           # 只看 20 条判据总表 -> results/ALL_GATES.md"
