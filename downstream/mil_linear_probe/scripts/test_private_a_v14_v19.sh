#!/bin/bash
# 批量重跑 v19 / v14 在 private_a test 上的结果（abmil_full + abmil_simple）。
# clam 仍在训练时可先不跑；训完后再单独：
#   bash scripts/test_private_a_by_region.sh topk_v19_last clam
#
# 用法：
#   bash scripts/test_private_a_v14_v19.sh

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS=(topk_v19_last)
MIL_TYPES=(abmil_full abmil_simple clam)

for MODEL_ID in "${MODELS[@]}"; do
  for MIL_TYPE in "${MIL_TYPES[@]}"; do
    echo ""
    echo "################################################################"
    echo "# Retest private_a: ${MODEL_ID} / ${MIL_TYPE}"
    echo "################################################################"
    bash scripts/test_private_a_by_region.sh "${MODEL_ID}" "${MIL_TYPE}"
  done
done

echo ""
echo "All private_a retests finished."
