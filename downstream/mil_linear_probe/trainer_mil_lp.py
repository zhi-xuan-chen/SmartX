"""
MIL 线性探针（可切换 ABMIL / CLAM 聚合）Lightning 模型。
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import sys
from pathlib import Path

# 确保能找到 downstream/utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import meanAUC, meanAP, meanF1, meanMCC


class AttentionMIL(nn.Module):
    """简单的 ABMIL 注意力池化（Ilse et al.）"""

    def __init__(self, in_dim: int, attn_hidden_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, V, D]
            mask: [B, V]，True 表示有效视角
        Returns:
            pooled: [B, D]
            attn: [B, V]
        """
        # [B, V, 1] -> [B, V]
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            # 无效视角赋极小值，避免 softmax 权重
            # 使用 -1e4 以确保兼容 float16（float16 范围约 [-65504, 65504]）
            fill_value = -1e4
            scores = scores.masked_fill(~mask, fill_value)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        if mask is not None:
            invalid = mask.sum(dim=1) == 0
            if invalid.any():
                attn = attn.clone()
                pooled = pooled.clone()
                attn[invalid] = 0
                pooled[invalid] = 0
        return pooled, attn


class GatedMultiHeadAttentionMIL(nn.Module):
    """完整 ABMIL：gated attention + 多头。"""

    def __init__(
        self,
        in_dim: int,
        attn_hidden_dim: int = 256,
        n_heads: int = 1,
        gated: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.gated = gated
        self.attn_V = nn.Linear(in_dim, attn_hidden_dim)
        self.attn_U = nn.Linear(in_dim, attn_hidden_dim) if gated else None
        self.attn_proj = nn.Linear(attn_hidden_dim, n_heads)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, V, D]
            mask: [B, V]
        Returns:
            pooled: [B, D]（多头后取平均以保持维度）
            attn: [B, V, n_heads]
        """
        v = torch.tanh(self.attn_V(x))
        u = torch.sigmoid(self.attn_U(x)) if self.gated else 1.0
        h = v * u

        scores = self.attn_proj(h)  # [B, V, H]
        if mask is not None:
            # 使用 -1e4 以确保兼容 float16（float16 范围约 [-65504, 65504]）
            fill_value = -1e4
            scores = scores.masked_fill(~mask.unsqueeze(-1), fill_value)

        attn = torch.softmax(scores, dim=1)  # [B, V, H]
        pooled_heads = torch.sum(attn.unsqueeze(-1) * x.unsqueeze(2), dim=1)  # [B, H, D]

        if self.n_heads > 1:
            pooled = pooled_heads.mean(dim=1)  # [B, D]
        else:
            pooled = pooled_heads.squeeze(1)  # [B, D]

        if mask is not None:
            invalid = mask.sum(dim=1) == 0
            if invalid.any():
                attn = attn.clone()
                pooled = pooled.clone()
                attn[invalid] = 0
                pooled[invalid] = 0

        return pooled, attn


class CLAMCore(nn.Module):
    """基于 CLAM 的注意力 MIL 核心（按类分支的注意力 + 分类头）。

    支持两种输入形式：
    - 单个 bag：h 形状为 [N, ins_dim]，返回 logits:[1, C]，attn:[C, N]
    - 批量 bag：h 形状为 [B, V, ins_dim]，mask:[B, V]，返回 logits:[B, C]，attn:[B, C, V]
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: str = "small",
        dropout: float = 0.25,
        n_classes: int = 2,
        ins_dim: int = 1024,
    ):
        super().__init__()
        self.size_dict = {"small": [ins_dim, 512, 256], "big": [ins_dim, 512, 384]}
        size = self.size_dict[size_arg]

        # 特征压缩
        self.feature_extractor = nn.Sequential(
            nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)
        )

        # Gated Attention
        self.gate = gate
        self.attention_V = nn.Sequential(nn.Linear(size[1], size[2]), nn.Tanh())
        self.attention_U = (
            nn.Sequential(nn.Linear(size[1], size[2]), nn.Sigmoid())
            if gate
            else None
        )
        self.attention_weights = nn.Linear(size[2], n_classes)

        # 按类别的分类头
        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for _ in range(n_classes)])
        self.n_classes = n_classes

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:
                - [N, ins_dim]：单个 bag，N 为 patch 数
                - [B, V, ins_dim]：批量 bag，B 为 batch 大小，V 为每个 bag 的 patch 数
            mask: 仅在批量模式下使用，形状为 [B, V]，True 表示有效 patch
        Returns:
            logits:
                - [1, n_classes]（单个 bag）
                - [B, n_classes]（批量）
            attn:
                - [n_classes, N]（单个 bag）
                - [B, n_classes, V]（批量）
        """
        if h.dim() == 2:
            # 兼容旧接口：单个 bag
            h_feat = self.feature_extractor(h)  # [N, H]

            a_v = self.attention_V(h_feat)
            a_u = self.attention_U(h_feat) if self.gate else 1.0
            a_combined = a_v * a_u

            A = self.attention_weights(a_combined)  # [N, C]
            A = torch.transpose(A, 1, 0)  # [C, N]
            A = F.softmax(A, dim=1)

            M = torch.mm(A, h_feat)  # [C, H]

            logits = torch.empty((1, self.n_classes), device=h.device, dtype=h.dtype)
            for c in range(self.n_classes):
                logits[0, c] = self.classifiers[c](M[c])

            return logits, A

        if h.dim() != 3:
            raise ValueError(
                f"CLAMCore expects h with dim 2 or 3, but got shape {tuple(h.shape)}"
            )

        # 批量模式：h [B, V, D]
        B, V, _ = h.shape
        h_feat = self.feature_extractor(h)  # [B, V, H]

        a_v = self.attention_V(h_feat)  # [B, V, H']
        a_u = self.attention_U(h_feat) if self.gate else 1.0  # [B, V, H']
        a_combined = a_v * a_u  # [B, V, H']

        A = self.attention_weights(a_combined)  # [B, V, C]
        A = A.permute(0, 2, 1)  # [B, C, V]

        if mask is not None:
            # mask: [B, V] -> [B, 1, V]
            mask_expanded = mask.unsqueeze(1)  # True: 有效
            # 使用 -1e4 以兼容 float16
            A = A.masked_fill(~mask_expanded, -1e4)

        A = F.softmax(A, dim=2)  # softmax over V

        # 加权求和得到每个类别的 bag 表征: [B, C, H]
        # einsum: (B,C,V) x (B,V,H) -> (B,C,H)
        M = torch.einsum("bcv,bvh->bch", A, h_feat)

        # 分类：对每个类别的表征分别过线性层
        logits = torch.empty(
            (B, self.n_classes), device=h.device, dtype=h.dtype
        )  # [B, C]
        for c in range(self.n_classes):
            # M[:, c, :] -> [B, H] -> [B, 1]
            logits[:, c] = self.classifiers[c](M[:, c, :]).squeeze(-1)

        return logits, A


class MILLinearProbe(pl.LightningModule):
    """可切换 CLAM / ABMIL 聚合的 MIL 线性探针模型。"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)

        # 训练超参
        self.lr = cfg.get("lr", 1e-3)
        self.weight_decay = cfg.get("weight_decay", 1e-4)
        self.warmup_steps = cfg.get("warmup_steps", 0)
        self.total_steps = cfg.get("total_steps", None)

        # 模型超参
        self.feature_dim = cfg["feature_dim"]
        self.num_classes = cfg["num_classes"]
        dropout = cfg.get("dropout", 0.1)
        self.mil_type = cfg.get("mil_type", "clam")

        if self.mil_type == "clam":
            self.mil_model = CLAMCore(
                gate=cfg.get("clam_gate", True),
                size_arg=cfg.get("clam_size", "small"),
                dropout=dropout,
                n_classes=self.num_classes,
                ins_dim=self.feature_dim,
            )
            self.classifier = None
            self.dropout = None
        else:
            attn_hidden = cfg.get("attn_hidden_dim", 256)
            attn_heads = cfg.get("attn_heads", 1)
            attn_gated = cfg.get("attn_gated", True)

            if self.mil_type == "abmil_full":
                self.mil_model = GatedMultiHeadAttentionMIL(
                    in_dim=self.feature_dim,
                    attn_hidden_dim=attn_hidden,
                    n_heads=attn_heads,
                    gated=attn_gated,
                )
            elif self.mil_type == "abmil_simple":
                self.mil_model = AttentionMIL(
                    self.feature_dim, attn_hidden_dim=attn_hidden
                )
            else:
                raise ValueError(f"Unsupported mil_type: {self.mil_type}")

            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.feature_dim, self.num_classes)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

        # 多标签 BCE 损失
        self.criterion = nn.BCEWithLogitsLoss()

        # 缓存预测用于自定义指标
        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
        self.test_regions: List[str] = []

        self._init_metrics()

    def _init_metrics(self):
        """初始化 torchmetrics。"""
        task = "multilabel" if self.num_classes > 1 else "binary"
        task_kwargs = {"num_labels": self.num_classes} if task == "multilabel" else {}

        self.train_acc = torchmetrics.Accuracy(task=task, **task_kwargs)
        self.val_acc = torchmetrics.Accuracy(task=task, **task_kwargs)
        self.test_acc = torchmetrics.Accuracy(task=task, **task_kwargs)

        auroc_kwargs = task_kwargs.copy()
        if task == "multilabel":
            auroc_kwargs["average"] = "macro"
        self.val_auroc = torchmetrics.AUROC(task=task, **auroc_kwargs)
        self.val_f1 = torchmetrics.F1Score(task=task, **task_kwargs)
        self.test_auroc = torchmetrics.AUROC(task=task, **auroc_kwargs)
        self.test_f1 = torchmetrics.F1Score(task=task, **task_kwargs)

    def forward(self, bags, mask=None):
        """
        Args:
            bags: [B, V, D]
            mask: [B, V]，True 表示有效 patch
        Returns:
            logits: [B, num_classes]
            attn_list/attn: CLAM 为 List[Tensor]，ABMIL 为 Tensor
        """
        if self.mil_type == "clam":
            # CLAM 批量路径：模型内部负责处理 mask
            logits, attn = self.mil_model(bags, mask)
            return logits, attn

        # ABMIL 路径：批次内一次性计算
        pooled, attn = self.mil_model(bags, mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits, attn

    def _shared_step(self, batch, stage: str):
        if len(batch) == 4:
            bags, mask, labels, regions = batch
        else:
            bags, mask, labels = batch
            regions = None
        
        # Debug: print actual batch size (only for first few steps)
        if not hasattr(self, '_batch_size_debug_count'):
            self._batch_size_debug_count = 0
        if self._batch_size_debug_count < 3:
            print(f"[DEBUG] {stage} batch size: bags.shape={bags.shape}, mask.shape={mask.shape}, labels.shape={labels.shape}")
            self._batch_size_debug_count += 1
        
        logits, _ = self(bags, mask)
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        labels_int = labels.int()

        if stage == "train":
            acc = self.train_acc(preds, labels_int)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.val_acc(preds, labels_int)
            self.val_auroc(probs, labels_int)
            self.val_f1(preds, labels_int)
            # 存储用于自定义指标
            self.val_predictions.append(probs.detach().cpu().numpy())
            self.val_labels.append(labels.detach().cpu().numpy())
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)
            self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
            self.log("val/auroc", self.val_auroc, on_epoch=True)
            self.log("val/f1", self.val_f1, on_epoch=True)
        else:
            self.test_acc(preds, labels_int)
            self.test_auroc(probs, labels_int)
            self.test_f1(preds, labels_int)
            self.test_predictions.append(probs.detach().cpu().numpy())
            self.test_labels.append(labels.detach().cpu().numpy())
            if regions is not None:
                self.test_regions.extend(list(regions))
            self.log("test/loss", loss, on_epoch=True)
            self.log("test/acc", self.test_acc, on_epoch=True)
            self.log("test/auroc", self.test_auroc, on_epoch=True)
            self.log("test/f1", self.test_f1, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        if len(self.val_predictions) == 0:
            return
        val_preds = np.concatenate(self.val_predictions, axis=0)
        val_labels = np.concatenate(self.val_labels, axis=0)
        try:
            mean_auc, _ = meanAUC(val_labels, val_preds)
            mean_ap, _ = meanAP(val_labels, val_preds)
            mean_f1, _ = meanF1(val_labels, val_preds)
            mean_mcc, _ = meanMCC(val_labels, val_preds)
            self.log("val/custom_auc", mean_auc, on_epoch=True, prog_bar=True)
            self.log("val/custom_ap", mean_ap, on_epoch=True, prog_bar=True)
            self.log("val/custom_f1", mean_f1, on_epoch=True, prog_bar=True)
            self.log("val/custom_mcc", mean_mcc, on_epoch=True, prog_bar=True)
            print(
                f"\nValidation Metrics - AUC:{mean_auc:.4f} AP:{mean_ap:.4f} "
                f"F1:{mean_f1:.4f} MCC:{mean_mcc:.4f}"
            )
        except Exception as e:
            print(f"Error computing validation metrics: {e}")
        self.val_predictions.clear()
        self.val_labels.clear()

    def _resolve_test_thresholds(self, n_classes: int) -> np.ndarray:
        """Resolve per-class binarisation thresholds for test-time F1/MCC.

        Lookup order:
          1. ``self.val_thresholds_for_test``: explicit override set by a
             caller (e.g., external test scripts that need to slice by
             ``used_indices_in_ckpt``).
          2. Sidecar ``val_thresholds.json`` next to the loaded checkpoint
             (written by ``SaveValThresholdsCallback`` at training time, or
             retroactively by ``save_val_thresholds.py``).
          3. Fallback to uniform 0.5 with a warning.
        """
        override = getattr(self, "val_thresholds_for_test", None)
        if override is not None:
            thr = np.asarray(override, dtype=np.float64)
            if thr.shape[0] != n_classes:
                print(
                    f"[val_thresholds] WARN: override length {thr.shape[0]} "
                    f"!= n_classes {n_classes}; falling back to 0.5."
                )
                return np.full(n_classes, 0.5, dtype=np.float64)
            return thr

        # Try sidecar JSON near the loaded ckpt.
        ckpt_path = getattr(self, "_loaded_ckpt_path", None)
        if ckpt_path is None:
            # Lightning records the most-recent load path on the trainer.
            try:
                trainer = self.trainer
                ckpt_path = getattr(trainer, "ckpt_path", None)
            except RuntimeError:
                ckpt_path = None
        if ckpt_path:
            try:
                from val_thresholds_utils import load_thresholds_for_ckpt
                hparams = getattr(self, "hparams", None)
                cols = None
                if hparams is not None:
                    cols = (
                        hparams.get("disease_cols")
                        if isinstance(hparams, dict)
                        else getattr(hparams, "disease_cols", None)
                    )
                thr = load_thresholds_for_ckpt(
                    ckpt_path,
                    expected_disease_cols=list(cols) if cols is not None else None,
                )
                if thr.shape[0] == n_classes:
                    return thr
                print(
                    f"[val_thresholds] WARN: sidecar threshold length "
                    f"{thr.shape[0]} != n_classes {n_classes}; falling back to 0.5."
                )
            except Exception as exc:
                print(f"[val_thresholds] WARN: failed to load sidecar: {exc}")

        print("[val_thresholds] WARN: using uniform 0.5 thresholds (no sidecar found).")
        return np.full(n_classes, 0.5, dtype=np.float64)

    def on_test_epoch_end(self):
        if len(self.test_predictions) == 0:
            return
        test_preds = np.concatenate(self.test_predictions, axis=0)
        test_labels = np.concatenate(self.test_labels, axis=0)
        test_regions = self.test_regions if len(self.test_regions) == len(test_preds) else None
        try:
            mean_auc, class_aucs = meanAUC(test_labels, test_preds)
            mean_ap, class_aps = meanAP(test_labels, test_preds)
            mean_f1, class_f1s = meanF1(test_labels, test_preds)
            mean_mcc, class_mccs = meanMCC(test_labels, test_preds)
            self.log("test/custom_auc", mean_auc, on_epoch=True)
            self.log("test/custom_ap", mean_ap, on_epoch=True)
            self.log("test/custom_f1", mean_f1, on_epoch=True)
            self.log("test/custom_mcc", mean_mcc, on_epoch=True)
            for i, (auc, ap, f1, mcc) in enumerate(
                zip(class_aucs, class_aps, class_f1s, class_mccs)
            ):
                self.log(f"test/class_{i}_auc", auc, on_epoch=True)
                self.log(f"test/class_{i}_ap", ap, on_epoch=True)
                self.log(f"test/class_{i}_f1", f1, on_epoch=True)
                self.log(f"test/class_{i}_mcc", mcc, on_epoch=True)
            print(
                f"\nTest Metrics - AUC:{mean_auc:.4f} AP:{mean_ap:.4f} "
                f"F1:{mean_f1:.4f} MCC:{mean_mcc:.4f}"
            )
        except Exception as e:
            print(f"Error computing test metrics: {e}")

        # 保存逐样本结果，便于 bootstrap
        if hasattr(self, "results_save_path") and self.results_save_path:
            try:
                import pandas as pd

                # F1/MCC 用 val 上选好的 per-class Youden's J 阈值，
                # 而不是在 test set 上重新 oracle。阈值来源（按优先级）：
                #   1. self.val_thresholds_for_test （由 test 入口脚本注入）
                #   2. sidecar val_thresholds.json （位于 ckpt 同目录）
                #   3. fallback 0.5
                thresholds = self._resolve_test_thresholds(n_classes=test_preds.shape[1])
                binary_preds = (test_preds > thresholds[None, :]).astype(int)
                binary_labels = test_labels.astype(int)
                correct = (binary_preds == binary_labels).all(axis=1).astype(int)

                results_df = pd.DataFrame(
                    {
                        "ground_truth": binary_labels.tolist(),
                        "prediction": binary_preds.tolist(),
                        "prediction_probs": test_preds.tolist(),
                        "correct": correct,
                        "region": test_regions if test_regions is not None else ["unknown"] * len(test_preds),
                    }
                )
                results_df.to_csv(self.results_save_path, index=False)
                print(f"Saved test results to: {self.results_save_path}")
            except Exception as e:
                print(f"Error saving test results: {e}")

        self.test_predictions.clear()
        self.test_labels.clear()
        self.test_regions.clear()

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.lr, betas=(0.9, 0.95), eps=1e-8
        )
        if self.total_steps and self.warmup_steps > 0:
            scheduler = self._create_scheduler(optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def _create_scheduler(self, optimizer):
        """线性 warmup + 余弦衰减。"""

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / max(1, self.warmup_steps)
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

