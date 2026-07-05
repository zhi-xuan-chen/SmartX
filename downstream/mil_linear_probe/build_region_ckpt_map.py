"""
根据实验目录结构，自动为每个 Region 选出「最佳 checkpoint」。

典型目录结构（由 train_mil_lp.py 创建）：

  <ckpt_root>/
    {dataset_name}_{Region}/
      {model_id}/
        bs{batch}_lr{lr}_{mil_type}/
          version_{k}/
            checkpoints/
              epoch=XX-val_custom_auc=0.XXXX.ckpt
              last.ckpt

本脚本的目标：
  - 给定 ckpt_root / dataset_name / model_id / mil_type
  - 对于每个 Region：
      1) 选择最大的 version_k
      2) 在该 version 的 checkpoints 中，优先选 epoch=..-val_custom_auc=..ckpt 作为 best
         - 如无，则回退到 last.ckpt
  - 输出 JSON: { "Chest": "/path/to/ckpt.ckpt", "Pelvis": "..." }
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _parse_version(dir_name: str) -> Optional[int]:
    """从 'version_3' 解析出 3。"""
    m = re.match(r"version_(\d+)$", dir_name)
    if not m:
        return None
    return int(m.group(1))


def _parse_epoch(ckpt_name: str) -> Optional[int]:
    """从 'epoch=34-val_custom_auc=0.1234.ckpt' 中解析 epoch。"""
    m = re.search(r"epoch=(\d+)", ckpt_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def _parse_val_auc(ckpt_name: str) -> Optional[float]:
    """从 'epoch=34-val_custom_auc=0.1234.ckpt' 中解析 val_custom_auc。"""
    m = re.search(r"val_custom_auc=([0-9]+(?:\.[0-9]+)?)", ckpt_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def build_region_ckpt_map(
    ckpt_root: Path,
    dataset_name: str,
    model_id: str,
    mil_type: str,
    regions: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    从给定的 ckpt_root 中，为每个 Region 挑选一个最佳 checkpoint。
    """
    ckpt_root = ckpt_root.expanduser().resolve()
    if not ckpt_root.exists():
        raise FileNotFoundError(f"ckpt_root 不存在: {ckpt_root}")

    ds_prefix = f"{dataset_name}_"
    region_filter = {str(r) for r in regions} if regions else None

    region_to_ckpt: Dict[str, str] = {}

    for ds_region_dir in sorted(ckpt_root.iterdir()):
        if not ds_region_dir.is_dir():
            continue
        name = ds_region_dir.name
        if not name.startswith(ds_prefix):
            continue

        region = name[len(ds_prefix) :]
        if region_filter and region not in region_filter:
            continue

        model_dir = ds_region_dir / model_id
        if not model_dir.is_dir():
            continue

        # 在 model_dir 下寻找形如 'bs*_lr*_{mil_type}' 的训练配置目录
        run_dirs = [
            d
            for d in model_dir.iterdir()
            if d.is_dir() and d.name.endswith(f"_{mil_type}")
        ]
        if not run_dirs:
            continue

        # 如同一 Region 下存在多个 bs/lr 组合，简单地按名称排序后取最后一个
        run_dirs = sorted(run_dirs, key=lambda p: p.name)
        run_dir = run_dirs[-1]

        # 在 run_dir 下选择最大的 version_k
        version_dirs = []
        for d in run_dir.iterdir():
            if not d.is_dir():
                continue
            v = _parse_version(d.name)
            if v is not None:
                version_dirs.append((v, d))
        if not version_dirs:
            continue

        version_dirs.sort(key=lambda x: x[0])
        _, best_version_dir = version_dirs[-1]

        ckpt_dir = best_version_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            continue

        ckpt_files = [f for f in ckpt_dir.glob("*.ckpt") if f.is_file()]
        if not ckpt_files:
            continue

        # 优先选择带 val_custom_auc 的 epoch=.. ckpt，其次才是 last.ckpt
        best_candidates = []
        last_candidate = None
        for f in ckpt_files:
            if re.match(r"last(-v\d+)?\.ckpt$", f.name):
                if f.name == "last.ckpt":
                    last_candidate = f
                continue
            auc = _parse_val_auc(f.name)
            if auc is None:
                continue
            best_candidates.append((auc, _parse_epoch(f.name), f))

        chosen: Optional[Path] = None
        if best_candidates:
            # auc 相同时取更大的 epoch（更接近训练末期的 best）
            best_candidates.sort(key=lambda x: (-(x[0] or 0.0), -(x[1] if x[1] is not None else -1)))
            chosen = best_candidates[0][2]
        elif last_candidate is not None:
            chosen = last_candidate

        if chosen is not None:
            region_to_ckpt[region] = str(chosen)

    return region_to_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="根据 ckpt_root 自动生成 Region->best ckpt 映射 JSON"
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        required=True,
        help="MIL 实验根目录，例如 ${SMARTX_STORAGE_ROOT}/.../experiments/MIL_LP",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="训练时使用的 dataset_name，例如 private_a / nyfs",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="特征/模型子目录名，例如 ark_base / ark_large",
    )
    parser.add_argument(
        "--mil_type",
        type=str,
        required=True,
        help="MIL 类型，对应配置中的 mil_type，例如 abmil_simple / clam 等",
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help="可选，只对特定 Region 构建映射，例如 Chest Pelvis",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 JSON 文件路径",
    )

    args = parser.parse_args()

    ckpt_root = Path(args.ckpt_root)
    region_to_ckpt = build_region_ckpt_map(
        ckpt_root=ckpt_root,
        dataset_name=args.dataset_name,
        model_id=args.model_id,
        mil_type=args.mil_type,
        regions=args.regions,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(region_to_ckpt, f, indent=2, ensure_ascii=False)

    print(f"共找到 {len(region_to_ckpt)} 个 Region 的 checkpoint 映射。")
    print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()

