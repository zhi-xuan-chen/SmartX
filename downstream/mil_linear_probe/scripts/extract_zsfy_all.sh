#!/bin/bash
# ZSFY: 依次运行所有模型的特征提取脚本

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== ZSFY 全模型特征提取 =========="

for name in ark_base ark_large maco chexzero chess gloria rad_dino siglip2 our_ark_base our_ark_large our_rad_dino our_siglip2; do
  script="extract_zsfy_${name}.sh"
  if [[ -f "$script" ]]; then
    echo ""
    echo ">>> 运行 $script"
    bash "$script" || { echo "失败: $script"; exit 1; }
  else
    echo "跳过（不存在）: $script"
  fi
done

echo ""
echo "========== ZSFY 全模型特征提取完成 =========="
