#!/usr/bin/env bash
# 环境准备：uv venv + Python 依赖
# 前置：已安装 uv（brew install uv）、cmake（brew install cmake）
set -e
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "===== 创建 Python 3.12 虚拟环境（uv）====="
uv venv --python 3.12 .venv

echo "===== 安装 Python 依赖 ====="
uv pip install \
  torch transformers onnx onnxslim onnxruntime \
  safetensors numpy sentencepiece tqdm yaspin \
  Pillow requests huggingface_hub accelerate einops \
  datasets peft torchvision

echo "===== 验证 ====="
.venv/bin/python -c "
import torch, transformers, onnx
print('torch', torch.__version__, '| MPS:', torch.backends.mps.is_available())
print('transformers', transformers.__version__)
print('onnx', onnx.__version__)
"
echo "✅ 环境就绪: $PROJ_ROOT/.venv"
