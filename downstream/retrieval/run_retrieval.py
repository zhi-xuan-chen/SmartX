#!/usr/bin/env python3
"""Run SmartX image-text retrieval on a generic case manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.modules import create_clip_style_model_global
from downstream.retrieval.datasets import case_manifest_rows, load_case_manifest
from downstream.retrieval.metrics import metrics_from_similarity, ranks_from_similarity

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return _expand_env(payload)


def _expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def resolve_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def optional_path_value(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text or "$" in text:
        return None
    return text


from downstream.modules.images import array_to_pil as _array_to_pil, load_image


def build_image_transform(
    image_size: int = 768,
    *,
    in_channels: int = 3,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    from torchvision import transforms

    if mean is None:
        mean = (0.485, 0.456, 0.406) if in_channels == 3 else (0.5,)
    if std is None:
        std = (0.229, 0.224, 0.225) if in_channels == 3 else (0.5,)
    return transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


class RetrievalImageDataset(Dataset):
    def __init__(self, cases, transform, in_channels: int = 3) -> None:
        self.items = []
        for case_idx, case in enumerate(cases):
            for image_path in case.image_paths:
                self.items.append((image_path, case_idx))
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, case_idx = self.items[index]
        image = load_image(image_path, in_channels=self.in_channels)
        return self.transform(image), int(case_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmartX retrieval.")
    parser.add_argument("--config", default="downstream/retrieval/configs/retrieval.example.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _model_config(cfg: dict, checkpoint_override: str | None) -> dict:
    model_cfg = cfg.get("model", {})
    payload = {
        "model_name": model_cfg.get("name", "smartx_large"),
        "checkpoint": optional_path_value(checkpoint_override or model_cfg.get("checkpoint")),
        "repo_id": model_cfg.get("repo_id"),
        "filename": model_cfg.get("filename"),
        "text_encoder_name": model_cfg.get("text_encoder_name", "pubmedbert"),
        "output_dim": model_cfg.get("output_dim", 512),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _encode_texts(text_model, tokenizer, texts: list[str], *, batch_size: int, device: torch.device):
    features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding reports"):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = text_model(**tokens)
            encoded = getattr(output, "pooler_output", output)
            features.append(F.normalize(encoded, dim=-1).cpu())
    return torch.cat(features, dim=0)


def _encode_images(vision_model, cases, transform, *, batch_size: int, num_workers: int, in_channels: int, device: torch.device):
    dataset = RetrievalImageDataset(cases, transform, in_channels=in_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    case_features = [[] for _ in cases]
    with torch.no_grad():
        for images, case_indices in tqdm(dataloader, desc="Encoding images"):
            output = vision_model(images.to(device, non_blocking=device.type == "cuda"))
            encoded = getattr(output, "pooler_output", output)
            encoded = F.normalize(encoded, dim=-1).cpu()
            for feature, case_idx in zip(encoded, case_indices.tolist()):
                case_features[case_idx].append(feature)
    missing = [idx for idx, values in enumerate(case_features) if not values]
    if missing:
        raise ValueError(f"Cases without encoded images: {missing[:5]}")
    return [torch.stack(values, dim=0) for values in case_features]


def _similarity(case_image_features, text_features: torch.Tensor, aggregation: str) -> np.ndarray:
    text_features = F.normalize(text_features, dim=-1)
    rows = []
    if aggregation == "mean":
        image_features = []
        for features in case_image_features:
            image_features.append(F.normalize(features.mean(dim=0), dim=0))
        matrix = torch.stack(image_features, dim=0) @ text_features.T
        return matrix.numpy()
    if aggregation == "max":
        for features in case_image_features:
            rows.append((features @ text_features.T).max(dim=0).values)
        return torch.stack(rows, dim=0).numpy()
    raise ValueError("aggregation must be 'mean' or 'max'")


def _topk_rows(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    k = min(k, scores.shape[1])
    candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def _write_outputs(output_dir: Path, cases, similarity: np.ndarray, metrics: dict, aggregation: str, top_k: int, write_per_case: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(case_manifest_rows(cases)).to_csv(output_dir / "manifest.csv", index=False)
    np.save(output_dir / "similarity.npy", similarity.astype(np.float32, copy=False))
    summary = {
        "num_cases": len(cases),
        "aggregation": aggregation,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for direction, values in metrics.items():
        rows.append({"direction": direction, "num_cases": len(cases), "aggregation": aggregation, **values})
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)

    if not write_per_case:
        return
    i2t_ranks = ranks_from_similarity(similarity, "image_to_text")
    t2i_ranks = ranks_from_similarity(similarity, "text_to_image")
    top_text = _topk_rows(similarity, top_k)
    top_image = _topk_rows(similarity.T, top_k)
    per_case = []
    for idx, case in enumerate(cases):
        per_case.append(
            {
                "case_uid": case.case_uid,
                "image_to_text_rank": int(i2t_ranks[idx]),
                "text_to_image_rank": int(t2i_ranks[idx]),
                "matched_score": float(similarity[idx, idx]),
                f"top{top_k}_text_case_uid": json.dumps([cases[item].case_uid for item in top_text[idx]]),
                f"top{top_k}_image_case_uid": json.dumps([cases[item].case_uid for item in top_image[idx]]),
            }
        )
    pd.DataFrame(per_case).to_csv(output_dir / "per_case_ranks.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    manifest_path = resolve_path(dataset_cfg.get("manifest"), Path.cwd())
    if manifest_path is None:
        raise ValueError("Set dataset.manifest in the retrieval config.")

    result = load_case_manifest(
        manifest_path,
        dataset=dataset_cfg.get("name", "custom"),
        split=dataset_cfg.get("split", "test"),
        image_root=resolve_path(dataset_cfg.get("image_root"), Path.cwd()),
        image_column=dataset_cfg.get("image_column"),
        text_column=dataset_cfg.get("text_column"),
        case_column=dataset_cfg.get("case_column"),
        max_cases=int(dataset_cfg.get("max_cases", 0)),
        subset_seed=int(dataset_cfg.get("subset_seed", 0)),
        verify_images=bool(dataset_cfg.get("verify_images", False)),
    )
    cases = list(result.cases)
    if not cases:
        raise ValueError("Retrieval manifest produced no usable cases.")

    device = torch.device(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    vision_model, text_model, tokenizer, model_info = create_clip_style_model_global(
        _model_config(cfg, args.checkpoint)
    )
    vision_model.to(device).eval()
    text_model.to(device).eval()

    batch_size = int(eval_cfg.get("batch_size", 16))
    image_size = int(eval_cfg.get("image_size", 768))
    in_channels = int(eval_cfg.get("in_channels", 3))
    transform = build_image_transform(image_size=image_size, in_channels=in_channels)
    text_features = _encode_texts(
        text_model,
        tokenizer,
        [case.report_text for case in cases],
        batch_size=batch_size,
        device=device,
    )
    image_features = _encode_images(
        vision_model,
        cases,
        transform,
        batch_size=batch_size,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        in_channels=in_channels,
        device=device,
    )
    aggregation = eval_cfg.get("aggregation", "mean")
    similarity = _similarity(image_features, text_features, aggregation)
    ks = tuple(int(k) for k in eval_cfg.get("recall_k", [1, 5, 10, 50, 100]))
    metrics = metrics_from_similarity(similarity, ks=ks)

    output_dir = resolve_path(args.output_dir or eval_cfg.get("output_dir"), Path.cwd())
    if output_dir is None:
        raise ValueError("Set evaluation.output_dir or pass --output-dir")
    payload_dir = output_dir / dataset_cfg.get("name", "custom") / dataset_cfg.get("split", "test")
    _write_outputs(
        payload_dir,
        cases,
        similarity,
        metrics,
        aggregation,
        top_k=int(eval_cfg.get("top_k", 5)),
        write_per_case=bool(eval_cfg.get("write_per_case", True)),
    )
    (payload_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": result.dataset,
                "split": result.split,
                "source_rows": result.source_rows,
                "skipped_missing_text": result.skipped_missing_text,
                "skipped_missing_images": result.skipped_missing_images,
                "notes": list(result.notes),
                "model": model_info,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote retrieval outputs to {payload_dir}")


if __name__ == "__main__":
    main()
