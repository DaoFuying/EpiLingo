from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

# ============================================================
# 0. dynamic import
# ============================================================

def load_framework_module(framework_path: str):
    framework_path = os.path.abspath(framework_path)
    module_name = Path(framework_path).stem

    spec = importlib.util.spec_from_file_location(module_name, framework_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# 1. Ablation config
# ============================================================

@dataclass
class AblationVariant:
    name: str
    use_projection: bool
    use_omics_adapter: bool
    use_species_adapter: bool
    use_task_adapter: bool
    use_asymmetric_pooling: bool
    use_conditional_decoder: bool


# ============================================================
# 2. Lightweight modules
# ============================================================

class ConditionalBottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.cond_proj = nn.Linear(cond_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = h
        x = self.norm(h)
        x = self.down(x)
        x = x + self.cond_proj(cond).unsqueeze(1)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x


class FeatureFiLM(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return h * (1.0 + gamma) + beta


class AsymmetricContextPooling(nn.Module):
    def __init__(self, hidden_dim: int, window: int = 3):
        super().__init__()
        self.window = window
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, h: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = h.shape
        outputs = []
        for i in range(bsz):
            c = int(center_idx[i].item())
            left_start = max(0, c - self.window)
            left_end = c
            right_start = c + 1
            right_end = min(seq_len, c + 1 + self.window)

            left = h[i, left_start:left_end].mean(dim=0) if left_end > left_start else torch.zeros(dim, device=h.device, dtype=h.dtype)
            center = h[i, c]
            right = h[i, right_start:right_end].mean(dim=0) if right_end > right_start else torch.zeros(dim, device=h.device, dtype=h.dtype)

            stacked = torch.cat([left, center, right], dim=-1)
            weights = torch.softmax(self.gate(stacked), dim=-1)

            pooled = torch.cat([
                weights[0] * left,
                weights[1] * center,
                weights[2] * right,
            ], dim=-1)
            outputs.append(pooled)

        return torch.stack(outputs, dim=0)


class CenterTokenPooling(nn.Module):
    def forward(self, h: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(h.size(0), device=h.device)
        return h[batch_idx, center_idx]


class ConditionalDecoder(nn.Module):
    def __init__(self, seq_dim: int, cond_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(seq_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(seq_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq_feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([seq_feat, cond], dim=-1)
        gate = self.gate(x)
        gated_seq = seq_feat * gate
        logits = self.classifier(torch.cat([gated_seq, cond], dim=-1)).squeeze(-1)
        return logits


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
# 3. Ablation model
# ============================================================

class AblationEpiLingo(nn.Module):
    def __init__(self, base_cfg, variant: AblationVariant, fw):
        super().__init__()
        self.cfg = base_cfg
        self.variant = variant
        self.fw = fw

        self.dna_rna_backbone = fw.AutoModel.from_pretrained(base_cfg.dna_rna_backbone_name)
        self.protein_backbone = fw.AutoModel.from_pretrained(base_cfg.protein_backbone_name)

        dna_hidden = self.dna_rna_backbone.config.hidden_size
        protein_hidden = self.protein_backbone.config.hidden_size

        if base_cfg.freeze_backbone:
            fw.freeze_all_parameters(self.dna_rna_backbone)
            fw.freeze_all_parameters(self.protein_backbone)
            fw.unfreeze_last_n_transformer_layers(self.dna_rna_backbone, base_cfg.unfreeze_last_n_layers)
            fw.unfreeze_last_n_transformer_layers(self.protein_backbone, base_cfg.unfreeze_last_n_layers)

        self.omics_embedding = nn.Embedding(base_cfg.num_omics_types, base_cfg.omics_emb_dim)
        self.species_embedding = nn.Embedding(base_cfg.num_species, base_cfg.species_emb_dim)
        self.task_embedding = nn.Embedding(base_cfg.num_tasks, base_cfg.task_emb_dim)
        self.cond_dim = base_cfg.omics_emb_dim + base_cfg.species_emb_dim + base_cfg.task_emb_dim

        # --------------------------------------------------
        # With projection: shared downstream dim
        # --------------------------------------------------
        if variant.use_projection:
            self.dna_proj = nn.Linear(dna_hidden, base_cfg.hidden_dim) if dna_hidden != base_cfg.hidden_dim else nn.Identity()
            self.protein_proj = nn.Linear(protein_hidden, base_cfg.hidden_dim) if protein_hidden != base_cfg.hidden_dim else nn.Identity()

            hidden_dim = base_cfg.hidden_dim

            self.omics_adapter = ConditionalBottleneckAdapter(hidden_dim, base_cfg.adapter_bottleneck_dim, base_cfg.omics_emb_dim, base_cfg.dropout)
            self.species_adapter = ConditionalBottleneckAdapter(hidden_dim, base_cfg.adapter_bottleneck_dim, base_cfg.species_emb_dim, base_cfg.dropout)
            self.task_adapter = ConditionalBottleneckAdapter(hidden_dim, base_cfg.adapter_bottleneck_dim, base_cfg.task_emb_dim, base_cfg.dropout)
            self.film = FeatureFiLM(hidden_dim, self.cond_dim)

            if variant.use_asymmetric_pooling:
                self.pooler = AsymmetricContextPooling(hidden_dim, base_cfg.context_window_tokens)
                seq_dim = hidden_dim * 3
            else:
                self.pooler = CenterTokenPooling()
                seq_dim = hidden_dim

            if variant.use_conditional_decoder:
                self.decoder = ConditionalDecoder(seq_dim, self.cond_dim, base_cfg.decoder_hidden_dim, base_cfg.dropout)
            else:
                self.decoder = SimpleClassifier(seq_dim, base_cfg.decoder_hidden_dim, base_cfg.dropout)

        # --------------------------------------------------
        # Without projection: branch-specific downstream heads
        # --------------------------------------------------
        else:
            self.dna_proj = nn.Identity()
            self.protein_proj = nn.Identity()

            self.dna_omics_adapter = ConditionalBottleneckAdapter(dna_hidden, base_cfg.adapter_bottleneck_dim, base_cfg.omics_emb_dim, base_cfg.dropout)
            self.dna_species_adapter = ConditionalBottleneckAdapter(dna_hidden, base_cfg.adapter_bottleneck_dim, base_cfg.species_emb_dim, base_cfg.dropout)
            self.dna_task_adapter = ConditionalBottleneckAdapter(dna_hidden, base_cfg.adapter_bottleneck_dim, base_cfg.task_emb_dim, base_cfg.dropout)
            self.dna_film = FeatureFiLM(dna_hidden, self.cond_dim)

            self.protein_omics_adapter = ConditionalBottleneckAdapter(protein_hidden, base_cfg.adapter_bottleneck_dim, base_cfg.omics_emb_dim, base_cfg.dropout)
            self.protein_species_adapter = ConditionalBottleneckAdapter(protein_hidden, base_cfg.adapter_bottleneck_dim, base_cfg.species_emb_dim, base_cfg.dropout)
            self.protein_task_adapter = ConditionalBottleneckAdapter(protein_hidden, base_cfg.adapter_bottleneck_dim, base_cfg.task_emb_dim, base_cfg.dropout)
            self.protein_film = FeatureFiLM(protein_hidden, self.cond_dim)

            if variant.use_asymmetric_pooling:
                self.dna_pooler = AsymmetricContextPooling(dna_hidden, base_cfg.context_window_tokens)
                self.protein_pooler = AsymmetricContextPooling(protein_hidden, base_cfg.context_window_tokens)
                dna_seq_dim = dna_hidden * 3
                protein_seq_dim = protein_hidden * 3
            else:
                self.dna_pooler = CenterTokenPooling()
                self.protein_pooler = CenterTokenPooling()
                dna_seq_dim = dna_hidden
                protein_seq_dim = protein_hidden

            if variant.use_conditional_decoder:
                self.dna_decoder = ConditionalDecoder(dna_seq_dim, self.cond_dim, base_cfg.decoder_hidden_dim, base_cfg.dropout)
                self.protein_decoder = ConditionalDecoder(protein_seq_dim, self.cond_dim, base_cfg.decoder_hidden_dim, base_cfg.dropout)
            else:
                self.dna_decoder = SimpleClassifier(dna_seq_dim, base_cfg.decoder_hidden_dim, base_cfg.dropout)
                self.protein_decoder = SimpleClassifier(protein_seq_dim, base_cfg.decoder_hidden_dim, base_cfg.dropout)

    def _select_backbone(self, omics_id: torch.Tensor):
        unique = torch.unique(omics_id)
        if len(unique) != 1:
            raise ValueError("Each mini-batch must contain exactly one omics type.")
        oid = int(unique[0].item())
        if oid in [self.fw.OMICS2ID["DNA"], self.fw.OMICS2ID["RNA"]]:
            return "dna"
        return "protein"

    def forward(self, input_ids, attention_mask, omics_id, species_id, task_id, center_idx, return_features=False):
        branch = self._select_backbone(omics_id)

        if branch == "dna":
            outputs = self.dna_rna_backbone(input_ids=input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state
            h = self.dna_proj(h)
        else:
            outputs = self.protein_backbone(input_ids=input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state
            h = self.protein_proj(h)

        e_omics = self.omics_embedding(omics_id)
        e_species = self.species_embedding(species_id)
        e_task = self.task_embedding(task_id)
        cond = torch.cat([e_omics, e_species, e_task], dim=-1)

        if self.variant.use_projection:
            if self.variant.use_omics_adapter:
                h = self.omics_adapter(h, e_omics)
            if self.variant.use_species_adapter:
                h = self.species_adapter(h, e_species)
            if self.variant.use_task_adapter:
                h = self.task_adapter(h, e_task)
            if self.variant.use_omics_adapter or self.variant.use_species_adapter or self.variant.use_task_adapter:
                h = self.film(h, cond)

            seq_feat = self.pooler(h, center_idx)

            if self.variant.use_conditional_decoder:
                logits = self.decoder(seq_feat, cond)
            else:
                logits = self.decoder(seq_feat)

        else:
            if branch == "dna":
                if self.variant.use_omics_adapter:
                    h = self.dna_omics_adapter(h, e_omics)
                if self.variant.use_species_adapter:
                    h = self.dna_species_adapter(h, e_species)
                if self.variant.use_task_adapter:
                    h = self.dna_task_adapter(h, e_task)
                if self.variant.use_omics_adapter or self.variant.use_species_adapter or self.variant.use_task_adapter:
                    h = self.dna_film(h, cond)

                seq_feat = self.dna_pooler(h, center_idx)

                if self.variant.use_conditional_decoder:
                    logits = self.dna_decoder(seq_feat, cond)
                else:
                    logits = self.dna_decoder(seq_feat)

            else:
                if self.variant.use_omics_adapter:
                    h = self.protein_omics_adapter(h, e_omics)
                if self.variant.use_species_adapter:
                    h = self.protein_species_adapter(h, e_species)
                if self.variant.use_task_adapter:
                    h = self.protein_task_adapter(h, e_task)
                if self.variant.use_omics_adapter or self.variant.use_species_adapter or self.variant.use_task_adapter:
                    h = self.protein_film(h, cond)

                seq_feat = self.protein_pooler(h, center_idx)

                if self.variant.use_conditional_decoder:
                    logits = self.protein_decoder(seq_feat, cond)
                else:
                    logits = self.protein_decoder(seq_feat)

        probs = torch.sigmoid(logits)
        out = {"logits": logits, "probs": probs}
        if return_features:
            out["seq_feat"] = seq_feat
            out["cond"] = cond
        return out


# ============================================================
# 4. Trainer
# ============================================================

class AblationTrainer:
    def __init__(self, model, train_cfg, fw):
        self.model = model
        self.train_cfg = train_cfg
        self.fw = fw
        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=train_cfg.mixed_precision and self.device.type == "cuda"
        )

    def build_scheduler(self, num_training_steps: int):
        num_warmup_steps = int(self.train_cfg.warmup_ratio * num_training_steps)
        self.scheduler = self.fw.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _move_batch(self, batch):
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return moved

    def train_one_epoch(self, loader, epoch: int):
        self.model.train()
        losses = []

        for step, batch in enumerate(loader, start=1):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                self.device.type,
                enabled=self.train_cfg.mixed_precision and self.device.type == "cuda"
            ):
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    omics_id=batch["omics_id"],
                    species_id=batch["species_id"],
                    task_id=batch["task_id"],
                    center_idx=batch["center_idx"],
                    return_features=False,
                )
                loss = F.binary_cross_entropy_with_logits(out["logits"], batch["label"])

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            losses.append(float(loss.detach().cpu()))

            if step % self.train_cfg.logging_steps == 0:
                print(f"[Epoch {epoch} | Step {step}] train_loss={np.mean(losses):.4f}")

        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(self, loader, return_curves: bool = False):
        self.model.eval()
        all_probs, all_labels, all_losses = [], [], []

        for batch in loader:
            batch = self._move_batch(batch)

            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                omics_id=batch["omics_id"],
                species_id=batch["species_id"],
                task_id=batch["task_id"],
                center_idx=batch["center_idx"],
                return_features=False,
            )

            loss = F.binary_cross_entropy_with_logits(out["logits"], batch["label"])
            all_losses.append(float(loss.detach().cpu()))
            all_probs.append(out["probs"].detach().cpu().numpy())
            all_labels.append(batch["label"].detach().cpu().numpy())

        probs = np.concatenate(all_probs).astype(float)
        labels = np.concatenate(all_labels).astype(int)

        metrics = self.fw.binary_metrics_from_probs(probs, labels)
        metrics["eval_loss"] = float(np.mean(all_losses))

        try:
            metrics["AUPRC"] = float(average_precision_score(labels, probs))
        except Exception:
            metrics["AUPRC"] = float("nan")

        if not return_curves:
            return metrics

        # ROC
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thresholds,
        })

        # PRC
        precision, recall, pr_thresholds = precision_recall_curve(labels, probs)

        # sklearn 的 precision_recall_curve 返回:
        # len(precision) = len(recall) = len(pr_thresholds) + 1
        pr_df = pd.DataFrame({
            "precision": precision,
            "recall": recall,
        })

        # 为了和你之前的保存风格一致，补一个 threshold 列
        threshold_full = np.append(pr_thresholds, np.nan)
        pr_df["threshold"] = threshold_full

        raw_df = pd.DataFrame({
            "label": labels,
            "prob": probs,
            "pred_label_0.5": (probs >= 0.5).astype(int),
        })

        return {
            "metrics": metrics,
            "roc_df": roc_df,
            "pr_df": pr_df,
            "raw_df": raw_df,
        }



    def fit(self, train_loader, valid_loader):
        total_steps = len(train_loader) * self.train_cfg.num_train_epochs
        self.build_scheduler(total_steps)

        best_auc = -1.0
        best_state = None

        for epoch in range(1, self.train_cfg.num_train_epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            valid_metrics = self.evaluate(valid_loader, return_curves=False)

            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} | "
                f"valid AUROC={valid_metrics['AUROC']:.4f}, "
                f"F1={valid_metrics['F1']:.4f}, MCC={valid_metrics['MCC']:.4f}"
            )

            if not math.isnan(valid_metrics["AUROC"]) and valid_metrics["AUROC"] > best_auc:
                best_auc = valid_metrics["AUROC"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 最终返回包含 metrics + roc_df + pr_df + raw_df 的完整结果
        return self.evaluate(valid_loader, return_curves=True)

# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation for EpiLingo")
    parser.add_argument("--framework_path", type=str, default="EpilingoHierarchicalFramework4.py")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./ablation_results")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fw = load_framework_module(args.framework_path)

    ModelConfig = fw.ModelConfig
    TrainConfig = fw.TrainConfig
    set_seed = fw.set_seed
    load_all_datasets = fw.load_all_datasets
    build_vocab = fw.build_vocab
    EpiLingoDataset = fw.EpiLingoDataset
    OmicsGroupedBatchSampler = fw.OmicsGroupedBatchSampler

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    train_cfg.seed = args.seed
    if args.epochs is not None:
        train_cfg.num_train_epochs = args.epochs
    if args.train_batch_size is not None:
        train_cfg.train_batch_size = args.train_batch_size
    if args.eval_batch_size is not None:
        train_cfg.eval_batch_size = args.eval_batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.device is not None:
        train_cfg.device = args.device

    set_seed(train_cfg.seed)

    dna_dir = os.path.join(args.data_root, "DNA")
    rna_dir = os.path.join(args.data_root, "RNA")
    ptm_dir = os.path.join(args.data_root, "PTM")

    train_samples, valid_samples = load_all_datasets(
        dna_dir=dna_dir,
        rna_dir=rna_dir,
        ptm_dir=ptm_dir,
    )

    all_samples = train_samples + valid_samples
    species2id = build_vocab([s.species for s in all_samples])
    task2id = build_vocab([s.task_name for s in all_samples])

    model_cfg.num_species = len(species2id)
    model_cfg.num_tasks = len(task2id)

    dna_rna_tokenizer = fw.AutoTokenizer.from_pretrained(model_cfg.dna_rna_backbone_name)
    protein_tokenizer = fw.AutoTokenizer.from_pretrained(model_cfg.protein_backbone_name, do_lower_case=False)

    train_dataset = EpiLingoDataset(
        samples=train_samples,
        species2id=species2id,
        task2id=task2id,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        cfg=model_cfg,
    )
    valid_dataset = EpiLingoDataset(
        samples=valid_samples,
        species2id=species2id,
        task2id=task2id,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        cfg=model_cfg,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=OmicsGroupedBatchSampler(
            train_dataset,
            batch_size=train_cfg.train_batch_size,
            shuffle=True,
        ),
        num_workers=train_cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=OmicsGroupedBatchSampler(
            valid_dataset,
            batch_size=train_cfg.eval_batch_size,
            shuffle=False,
        ),
        num_workers=train_cfg.num_workers,
    )

    variants = [
        AblationVariant(
            name="Backbone only",
            use_projection=False,
            use_omics_adapter=False,
            use_species_adapter=False,
            use_task_adapter=False,
            use_asymmetric_pooling=False,
            use_conditional_decoder=False,
        ),
        AblationVariant(
            name="+ projection",
            use_projection=True,
            use_omics_adapter=False,
            use_species_adapter=False,
            use_task_adapter=False,
            use_asymmetric_pooling=False,
            use_conditional_decoder=False,
        ),
        AblationVariant(
            name="+ omics adapter",
            use_projection=True,
            use_omics_adapter=True,
            use_species_adapter=False,
            use_task_adapter=False,
            use_asymmetric_pooling=False,
            use_conditional_decoder=False,
        ),
        AblationVariant(
            name="+ species adapter",
            use_projection=True,
            use_omics_adapter=True,
            use_species_adapter=True,
            use_task_adapter=False,
            use_asymmetric_pooling=False,
            use_conditional_decoder=False,
        ),
        AblationVariant(
            name="+ task adapter",
            use_projection=True,
            use_omics_adapter=True,
            use_species_adapter=True,
            use_task_adapter=True,
            use_asymmetric_pooling=False,
            use_conditional_decoder=False,
        ),
        AblationVariant(
            name="+ asymmetric pooling",
            use_projection=True,
            use_omics_adapter=True,
            use_species_adapter=True,
            use_task_adapter=True,
            use_asymmetric_pooling=True,
            use_conditional_decoder=False,
        ),
        AblationVariant(
            name="Full EpiLingo",
            use_projection=True,
            use_omics_adapter=True,
            use_species_adapter=True,
            use_task_adapter=True,
            use_asymmetric_pooling=True,
            use_conditional_decoder=True,
        ),
    ]

    rows = []

    for variant in variants:
        print("\n" + "=" * 80)
        print(f"Running ablation variant: {variant.name}")
        print("=" * 80)

        model = AblationEpiLingo(model_cfg, variant, fw)
        trainer = AblationTrainer(model, train_cfg, fw)
        result = trainer.fit(train_loader, valid_loader)

        metrics = result["metrics"]
        roc_df = result["roc_df"]
        pr_df = result["pr_df"]
        raw_df = result["raw_df"]

        # 文件名安全化
        safe_name = (
            variant.name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("/", "_")
        )

        roc_path = os.path.join(args.output_dir, f"{safe_name}_roc.csv")
        pr_path = os.path.join(args.output_dir, f"{safe_name}_pr.csv")
        raw_path = os.path.join(args.output_dir, f"{safe_name}_raw_predictions.csv")

        roc_df.to_csv(roc_path, index=False)
        pr_df.to_csv(pr_path, index=False)
        raw_df.to_csv(raw_path, index=False)

        print(f"Saved ROC: {roc_path}")
        print(f"Saved PR : {pr_path}")
        print(f"Saved RAW: {raw_path}")

        row = {
            "Model variant": variant.name,
            "AUROC": metrics.get("AUROC", float("nan")),
            "AUPRC": metrics.get("AUPRC", float("nan")),
            "ACC": metrics.get("ACC", float("nan")),
            "F1": metrics.get("F1", float("nan")),
            "MCC": metrics.get("MCC", float("nan")),
            "Precision": metrics.get("Precision", float("nan")),
            "Recall": metrics.get("Recall", float("nan")),
            "eval_loss": metrics.get("eval_loss", float("nan")),
            "use_projection": variant.use_projection,
            "use_omics_adapter": variant.use_omics_adapter,
            "use_species_adapter": variant.use_species_adapter,
            "use_task_adapter": variant.use_task_adapter,
            "use_asymmetric_pooling": variant.use_asymmetric_pooling,
            "use_conditional_decoder": variant.use_conditional_decoder,
        }
        rows.append(row)


    results_df = pd.DataFrame(rows)

    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    #xlsx_path = os.path.join(args.output_dir, "ablation_results.xlsx")

    results_df.to_csv(csv_path, index=False)
    #results_df.to_excel(xlsx_path, index=False)

    print("\nSaved ablation results:")
    print(f" - {csv_path}")
    #print(f" - {xlsx_path}")
    print("\nAblation summary:")
    print(results_df[["Model variant", "AUROC", "F1", "MCC"]].to_string(index=False))


if __name__ == "__main__":
    main()