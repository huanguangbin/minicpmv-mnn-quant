#!/usr/bin/env bash
# 下载模型：官方 MNN 量化版（对标用）+ 原始 PyTorch 权重（量化源头）
# 注意：约 1.5GB + 2.9GB，仅下载不进 git
set -e
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"
PY=".venv/bin/python"

echo "===== 下载官方 MNN 量化版（taobao-mnn/MiniCPM-V-4.7-1B-MNN，约 1.5GB）====="
mkdir -p models/MiniCPM-V-4.7-1B-MNN
cd models/MiniCPM-V-4.7-1B-MNN
BASE="https://huggingface.co/taobao-mnn/MiniCPM-V-4.7-1B-MNN/resolve/main"
for f in config.json llm_config.json conversion.json export_args.json \
         tokenizer.mtok llm.mnn visual.mnn \
         llm.mnn.weight visual.mnn.weight embeddings_bf16.bin; do
  [ -f "$f" ] || curl -L -C - --retry 3 -o "$f" "$BASE/$f"
done
cd "$PROJ_ROOT"

echo "===== 下载原始 PyTorch 权重（openbmb/MiniCPM-V-4.7-1B，约 2.9GB）====="
$PY -c "
from huggingface_hub import snapshot_download
snapshot_download('openbmb/MiniCPM-V-4.7-1B', local_dir='models/MiniCPM-V-4.7-1B-hf')
print('done')
"

echo "===== 生成 Metal 运行配置 ====="
$PY - <<'EOF'
import json
from pathlib import Path
base = json.loads(Path('models/MiniCPM-V-4.7-1B-MNN/config.json').read_text())
cfg = dict(base)
cfg.update(backend_type='metal', precision='low', thread_num=4,
           sampler_type='greedy', temperature=0.0, top_k=1)
cfg['mllm'] = dict(backend_type='metal', precision='low', thread_num=4, memory='low')
Path('models/MiniCPM-V-4.7-1B-MNN/config-test-metal.json').write_text(json.dumps(cfg, indent=2))
print('done')
EOF

echo "✅ 模型下载完成: models/"
