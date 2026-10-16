#!/usr/bin/env bash
# 下载 demo 需要的两个模型 (默认走国内镜像 HF_ENDPOINT=https://hf-mirror.com)。
#   bash scripts/download_models.sh
#   HF_ENDPOINT=https://huggingface.co bash scripts/download_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=${PY:-}
if [ -z "$PY" ]; then
  for c in python3 /usr/local/python3.11.15/bin/python; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c "import numpy" >/dev/null 2>&1 && PY="$c" && break
  done
fi
[ -n "$PY" ] || { echo "找不到可用的 python3"; exit 2; }

"$PY" -c "import huggingface_hub" 2>/dev/null || {
  echo "缺少 huggingface_hub, 正在安装..."
  "$PY" -m pip install -q "huggingface_hub>=0.20" || {
    echo "安装失败。可手动下载: 把 Qwen/Qwen2.5-0.5B-Instruct 的文件放进 models/qwen2.5-0.5b-instruct/,"
    echo "把 facebook/deit-tiny-patch16-224 的文件放进 models/deit-tiny-patch16-224/"; exit 2; }
}

echo "HF_ENDPOINT=$HF_ENDPOINT"
"$PY" - <<'EOF'
import os
from huggingface_hub import snapshot_download

jobs = [("Qwen/Qwen2.5-0.5B-Instruct", "models/qwen2.5-0.5b-instruct",
         ["config.json", "generation_config.json", "merges.txt", "model.safetensors",
          "tokenizer.json", "tokenizer_config.json", "vocab.json"]),
        ("facebook/deit-tiny-patch16-224", "models/deit-tiny-patch16-224",
         ["config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"])]
for repo, local, files in jobs:
    if os.path.exists(os.path.join(local, "config.json")):
        print("[skip] %s 已存在" % local); continue
    print("[get ] %s -> %s" % (repo, local))
    snapshot_download(repo_id=repo, local_dir=local, allow_patterns=files)
    print("[ok  ] %s" % local)
EOF

echo "模型就绪:"
du -sh models/* 2>/dev/null || true
