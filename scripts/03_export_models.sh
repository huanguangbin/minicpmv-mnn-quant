#!/usr/bin/env bash
# 核心步骤：从原始 PyTorch 权重导出三个精度版本（FP16 / INT8 / HQQ4）
# 用法：./03_export_models.sh [fp16|int8|hqq4|all]  （默认 all）
set -e
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT/mnn/transformers/llm/export"

PY="$PROJ_ROOT/.venv/bin/python"
SRC="$PROJ_ROOT/models/MiniCPM-V-4.7-1B-hf"
MNNConvert="$PROJ_ROOT/mnn/build/MNNConvert"
MODELS="$PROJ_ROOT/models"
TARGET="${1:-all}"

export_fp16() {
  echo "===== 导出 FP16 基线（quant_bit 16 = 不量化）====="
  $PY llmexport.py --path "$SRC" --export mnn \
    --quant_bit 16 --visual_quant_bit 16 --lm_quant_bit 16 \
    --mnnconvert "$MNNConvert" --dst_path "$MODELS/MiniCPM-V-4.7-1B-FP16"
}

export_int8() {
  echo "===== 导出 INT8（weight-only RTN）====="
  $PY llmexport.py --path "$SRC" --export mnn \
    --quant_bit 8 --visual_quant_bit 8 --lm_quant_bit 8 \
    --mnnconvert "$MNNConvert" --dst_path "$MODELS/MiniCPM-V-4.7-1B-INT8"
}

export_hqq4() {
  echo "===== 导出 HQQ4（对标官方：语言 HQQ4 + 视觉默认 HQQ8）====="
  $PY llmexport.py --path "$SRC" --export mnn \
    --hqq --quant_bit 4 --quant_block 64 --scale_bit 16 \
    --mnnconvert "$MNNConvert" --dst_path "$MODELS/MiniCPM-V-4.7-1B-HQQ4"
}

case "$TARGET" in
  fp16) export_fp16 ;;
  int8) export_int8 ;;
  hqq4) export_hqq4 ;;
  all)  export_fp16; export_int8; export_hqq4 ;;
  *) echo "用法: $0 [fp16|int8|hqq4|all]"; exit 1 ;;
esac

echo ""
echo "===== 为新导出的模型生成 Metal 运行配置 ====="
cd "$PROJ_ROOT"
$PY - <<'EOF'
import json
from pathlib import Path
for name in ['MiniCPM-V-4.7-1B-FP16', 'MiniCPM-V-4.7-1B-INT8', 'MiniCPM-V-4.7-1B-HQQ4']:
    d = Path('models') / name
    if not (d / 'config.json').exists():
        continue
    base = json.loads((d / 'config.json').read_text())
    cfg = dict(base)
    cfg.update(backend_type='metal', precision='low', thread_num=4,
               sampler_type='greedy', temperature=0.0, top_k=1)
    cfg['mllm'] = dict(backend_type='metal', precision='low', thread_num=4, memory='low')
    (d / 'config-test-metal.json').write_text(json.dumps(cfg, indent=2))
    print(f'  ✅ {name}/config-test-metal.json')
EOF
echo "✅ 导出完成"
