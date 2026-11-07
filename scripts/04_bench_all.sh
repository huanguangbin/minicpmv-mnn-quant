#!/usr/bin/env bash
# 速度基准：对四个模型测 prefill/decode 速度（llm_bench）
set -e
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"
BENCH="mnn/build/llm_bench"

# 参数：-p 512 测 prefill(处理512 token) / -n 128 测 decode(生成128 token) / -rep 2 重复取均值
for m in MiniCPM-V-4.7-1B-FP16 MiniCPM-V-4.7-1B-INT8 MiniCPM-V-4.7-1B-HQQ4 MiniCPM-V-4.7-1B-MNN; do
  d="models/$m"
  [ -f "$d/config-test-metal.json" ] || { echo "跳过 $m（无配置）"; continue; }
  echo "========== $m =========="
  $BENCH -m "$d/config-test-metal.json" -a metal -c 1 -t 4 -p 512 -n 128 -rep 2 2>&1 \
    | grep -E "pp512|tg128"
  echo ""
done
echo "✅ 基准测试完成"
