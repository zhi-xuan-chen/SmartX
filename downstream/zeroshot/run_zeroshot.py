#!/usr/bin/env python3
"""Run SmartX zero-shot classification with positive/negative prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.zeroshot.data import (
    ClassificationManifestDataset,
    build_image_transform,
    load_yaml_config,
    optional_path_value,
    resolve_path,
    write_json,
)
from downstream.modules import create_clip_style_model_global
from downstream.zeroshot.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public SmartX zero-shot evaluation.")
    parser.add_argument("--config", default="downstream/zeroshot/configs/zeroshot.example.yaml")
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


def _prompt_pairs(dataset_cfg: dict, prompt_cfg: dict) -> tuple[list[str], list[list[str]], list[list[str]]]:
    if prompt_cfg.get("classes"):
        class_names = [item["name"] for item in prompt_cfg["classes"]]
        positive = [item.get("positive", item.get("prompts", [])) for item in prompt_cfg["classes"]]
        negative = [item.get("negative", []) for item in prompt_cfg["classes"]]
    else:
        class_names = list(dataset_cfg.get("class_names", []))
        pos_templates = prompt_cfg.get("positive_templates") or [
            "This X-ray demonstrates {class_name}.",
            "The radiograph shows evidence of {class_name}.",
        ]
        neg_templates = prompt_cfg.get("negative_templates") or [
            "This X-ray does not show {class_name}.",
            "The radiograph shows no evidence of {class_name}.",
        ]
        positive = [
            [template.format(class_name=class_name) for template in pos_templates]
            for class_name in class_names
        ]
        negative = [
            [template.format(class_name=class_name) for template in neg_templates]
            for class_name in class_names
        ]
    if not class_names:
        raise ValueError("Configure dataset.class_names or prompts.classes")
    for idx, class_name in enumerate(class_names):
        if not positive[idx]:
            positive[idx] = [f"This X-ray demonstrates {class_name}."]
        if not negative[idx]:
            negative[idx] = [f"This X-ray does not show {class_name}."]
    return class_names, positive, negative


def _encode_texts(text_model, tokenizer, prompts: list[str], *, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    output = text_model(**tokens)
    features = getattr(output, "pooler_output", output)
    return F.normalize(features, dim=-1)


def _build_text_weights(text_model, tokenizer, prompts_per_class: list[list[str]], device: torch.device) -> torch.Tensor:
    weights = []
    with torch.no_grad():
        for prompts in prompts_per_class:
            encoded = _encode_texts(text_model, tokenizer, prompts, device=device)
            weights.append(F.normalize(encoded.mean(dim=0), dim=0))
    return torch.stack(weights, dim=0)


def _encode_images(vision_model, dataloader, device: torch.device):
    features = []
    labels = []
    sample_ids = []
    with torch.no_grad():
        for images, batch_labels, ids in tqdm(dataloader, desc="Encoding images"):
            output = vision_model(images.to(device, non_blocking=device.type == "cuda"))
            batch_features = getattr(output, "pooler_output", output)
            features.append(F.normalize(batch_features, dim=-1).cpu())
            labels.append(batch_labels.cpu())
            sample_ids.extend(str(item) for item in ids)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0), sample_ids


def _manifest_path(dataset_cfg: dict) -> Path:
    manifests = dataset_cfg.get("manifests")
    split = dataset_cfg.get("split", "test")
    if manifests:
        return resolve_path(manifests[split], Path.cwd())
    return resolve_path(dataset_cfg.get("manifest"), Path.cwd())


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    prompt_cfg = cfg.get("prompts", {})
    class_names, positive_prompts, negative_prompts = _prompt_pairs(dataset_cfg, prompt_cfg)

    image_size = int(eval_cfg.get("image_size", 768))
    in_channels = int(eval_cfg.get("in_channels", 3))
    transform = build_image_transform(image_size=image_size, in_channels=in_channels)
    dataset = ClassificationManifestDataset(
        _manifest_path(dataset_cfg),
        image_root=resolve_path(dataset_cfg.get("image_root"), Path.cwd()),
        split=dataset_cfg.get("split_filter"),
        image_column=dataset_cfg.get("image_column"),
        label_column=dataset_cfg.get("label_column"),
        label_columns=dataset_cfg.get("label_columns"),
        class_names=class_names,
        task_type=dataset_cfg.get("task_type", "multi-label"),
        transform=transform,
        in_channels=in_channels,
        verify_images=bool(dataset_cfg.get("verify_images", False)),
    )
    if len(dataset) == 0:
        raise ValueError("Zero-shot manifest produced no samples.")

    batch_size = int(eval_cfg.get("batch_size", 16))
    num_workers = int(eval_cfg.get("num_workers", 0))
    loader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": num_workers}
    dataloader = DataLoader(dataset, **loader_kwargs)

    device = torch.device(eval_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    vision_model, text_model, tokenizer, model_info = create_clip_style_model_global(
        _model_config(cfg, args.checkpoint)
    )
    vision_model.to(device).eval()
    text_model.to(device).eval()

    positive_weights = _build_text_weights(text_model, tokenizer, positive_prompts, device)
    negative_weights = _build_text_weights(text_model, tokenizer, negative_prompts, device)
    image_features, labels, sample_ids = _encode_images(vision_model, dataloader, device)

    image_features = image_features.to(device)
    s_pos = image_features @ positive_weights.T
    s_neg = image_features @ negative_weights.T
    delta = (s_pos - s_neg).cpu().numpy()
    p_sigmoid = 1.0 / (1.0 + np.exp(-delta))
    p_softmax = np.exp(delta - delta.max(axis=1, keepdims=True))
    p_softmax = p_softmax / p_softmax.sum(axis=1, keepdims=True)

    task_type = dataset_cfg.get("task_type", "multi-label")
    metric_scores = p_softmax if task_type == "multi-class" else delta
    metrics = compute_classification_metrics(
        labels.numpy(),
        metric_scores,
        task_type=task_type,
        threshold=0.0 if task_type != "multi-class" else 0.5,
    )

    output_dir = resolve_path(args.output_dir or eval_cfg.get("output_dir"), Path.cwd())
    if output_dir is None:
        raise ValueError("Set evaluation.output_dir or pass --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "metrics.json",
        {
            "dataset": dataset_cfg.get("name", "custom"),
            "split": dataset_cfg.get("split", "test"),
            "class_names": class_names,
            "model": model_info,
            "metrics": metrics,
        },
    )

    rows = {"sample_id": sample_ids}
    for idx, class_name in enumerate(class_names):
        rows[f"{class_name}_s_pos"] = s_pos[:, idx].cpu().numpy()
        rows[f"{class_name}_s_neg"] = s_neg[:, idx].cpu().numpy()
        rows[f"{class_name}_delta"] = delta[:, idx]
        rows[f"{class_name}_p_sigmoid"] = p_sigmoid[:, idx]
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "prompts.json").write_text(
        json.dumps(
            {
                "class_names": class_names,
                "positive_prompts": positive_prompts,
                "negative_prompts": negative_prompts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote zero-shot outputs to {output_dir}")


if __name__ == "__main__":
    main()
