from __future__ import annotations

import glob
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except Exception as e:
    raise ImportError("Please install transformers first: pip install transformers") from e


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seq_to_kmer(seq: str, k: int = 6) -> str:
    seq = seq.strip().upper()
    if len(seq) < k:
        return seq
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    return " ".join(kmers)


def read_label_seq_file(filepath: str) -> pd.DataFrame:
    """
    Robust reader for files containing label/seq information.
    Supports:
    1. Standard header: label, seq
    2. Alternative header names: Label, Sequence, class, y, etc.
    3. No header: assume first two columns are label and seq
    """
    sep = "\t" if filepath.endswith(".tsv") else ","

    # Try with header
    df = pd.read_csv(filepath, sep=sep)
    original_cols = list(df.columns)
    df.columns = [str(c).strip().lower() for c in df.columns]

    label_candidates = ["label", "labels", "y", "class", "target"]
    seq_candidates = ["seq", "sequence", "sequences", "text"]

    label_col = next((c for c in label_candidates if c in df.columns), None)
    seq_col = next((c for c in seq_candidates if c in df.columns), None)

    if label_col is not None and seq_col is not None:
        return df[[label_col, seq_col]].rename(columns={label_col: "label", seq_col: "seq"})

    # Try no-header mode
    df_no_header = pd.read_csv(filepath, sep=sep, header=None)
    if df_no_header.shape[1] < 2:
        raise ValueError(
            f"{filepath} must contain at least two columns for label and seq. "
            f"Detected columns: {original_cols}"
        )
    df_no_header = df_no_header.iloc[:, :2].copy()
    df_no_header.columns = ["label", "seq"]
    return df_no_header


# ============================================================
# 2. Configs
# ============================================================

@dataclass
class ModelConfig:
    dna_rna_backbone_name: str = "zhihan1996/DNA_bert_6"
    protein_backbone_name: str = "Rostlab/prot_bert"

    # shared hidden dim after projection
    hidden_dim: int = 768

    omics_emb_dim: int = 32
    species_emb_dim: int = 64
    task_emb_dim: int = 64

    adapter_bottleneck_dim: int = 128
    decoder_hidden_dim: int = 256
    dropout: float = 0.1

    use_asymmetric_context: bool = True
    context_window_tokens: int = 3

    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 2

    max_length_dna_rna: int = 64   # 41bp -> 36 6-mers + special tokens
    max_length_protein: int = 41

    num_omics_types: int = 3
    num_species: int = 64
    num_tasks: int = 64
    num_labels: int = 2

    kmer: int = 6

    use_species_consistency_loss: bool = False
    species_consistency_weight: float = 0.05

    use_task_contrastive_loss: bool = False
    task_contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.1


@dataclass
class TrainConfig:
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    logging_steps: int = 100
    seed: int = 42

    gradient_clip_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 0


# ============================================================
# 3. Metadata
# ============================================================

OMICS2ID: Dict[str, int] = {
    "DNA": 0,
    "RNA": 1,
    "Protein": 2,
}


@dataclass
class Sample:
    sequence: str
    label: int
    omics_type: str
    species: str
    task_name: str
    center_position: int
    sample_id: Optional[str] = None


# ============================================================
# 4. Name normalization
# ============================================================

def build_vocab(items: List[str]) -> Dict[str, int]:
    uniq = sorted(set(items))
    return {name: idx for idx, name in enumerate(uniq)}


def normalize_species_name(name: str) -> str:
    mapping = {
        # DNA/RNA short names
        "arabidopsis": "Arabidopsis_thaliana",
        "chinensis": "Rosa_chinensis",
        "equisetifolia": "Casuarina_equisetifolia",
        "fragaria": "Fragaria_vesca",

        # PTM short names
        "arachis": "Arachis_hypogaea",
        "brachypodium": "Brachypodium_distachyon",
        "camellia": "Camellia_sinensis",
        "carica": "Carica_papaya",
        "gossypium": "Gossypium_hirsutum",
        "jatropha": "Jatropha_curcas",
        "nicotiana": "Nicotiana_tabacum",
        "oryza": "Oryza_sativa",
        "physcomitrium": "Physcomitrium_patens",
        "triticum": "Triticum_aestivum",
        "vitis": "Vitis_vinifera",

        # full names
        "Arabidopsis_thaliana": "Arabidopsis_thaliana",
        "Rosa_chinensis": "Rosa_chinensis",
        "Casuarina_equisetifolia": "Casuarina_equisetifolia",
        "Fragaria_vesca": "Fragaria_vesca",
        "Arachis_hypogaea": "Arachis_hypogaea",
        "Brachypodium": "Brachypodium_distachyon",
        "Brachypodium_distachyon": "Brachypodium_distachyon",
        "Camellia_sinensis": "Camellia_sinensis",
        "Carica_papaya": "Carica_papaya",
        "Gossypium_hirsutum": "Gossypium_hirsutum",
        "Jatropha_curcas": "Jatropha_curcas",
        "Nicotiana_tabacum": "Nicotiana_tabacum",
        "Oryza_sativa": "Oryza_sativa",
        "Physcomitrium_patens": "Physcomitrium_patens",
        "Triticum_aestivum": "Triticum_aestivum",
        "Vitis_vinifera": "Vitis_vinifera",
    }
    return mapping.get(name, name)


def normalize_task_name(name: str) -> str:
    mapping = {
        "4mC": "4mC",
        "6mA": "6mA",
        "m5C": "m5C",
        "m6A": "m6A",
        "Psi": "Psi",

        "acetylation": "Kac",
        "crotonylation": "Kcr",
        "crontonylation": "Kcr",
        "phosphorylation": "Phos",
        "2-hydroxyisobutyrylation": "Khib",
        "2-Hydroxyisobutyrylation": "Khib",
        "succinylation": "Ksu",
        "S-Nitrosylation": "S-Nitro",

        "Kac": "Kac",
        "Kcr": "Kcr",
        "Phos": "Phos",
        "Khib": "Khib",
        "Ksu": "Ksu",
        "S-Nitro": "S-Nitro",
    }
    return mapping.get(name, name)


# ============================================================
# 5. Filename parsing
# ============================================================

def parse_dna_rna_filename(filepath: str, omics_type: str) -> Tuple[str, str, str, str]:
    stem = Path(filepath).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filepath}")

    task_raw = parts[0]
    species_raw = parts[1]
    split_raw = parts[-1].lower()

    task_name = normalize_task_name(task_raw)
    species = normalize_species_name(species_raw)

    if split_raw == "train":
        split = "train"
    elif split_raw in ["dev", "valid", "val"]:
        split = "valid"
    else:
        raise ValueError(f"Unknown split in filename: {filepath}")

    return omics_type, species, task_name, split


def parse_ptm_filename(filepath: str) -> Tuple[str, str, str, str]:
    stem = Path(filepath).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected PTM filename format: {filepath}")

    task_raw = parts[0]
    species_raw = parts[1]
    split_raw = parts[-1].lower()

    task_name = normalize_task_name(task_raw)
    species = normalize_species_name(species_raw)
    omics_type = "Protein"

    if split_raw == "train":
        split = "train"
    elif split_raw in ["dev", "valid", "val"]:
        split = "valid"
    else:
        raise ValueError(f"Unknown PTM split in filename: {filepath}")

    return omics_type, species, task_name, split


# ============================================================
# 6. Data loading
# ============================================================

def infer_center_position(seq: str, omics_type: str) -> int:
    return len(seq) // 2


def load_one_dataset_file(filepath: str, omics_type: Optional[str] = None) -> Tuple[List[Sample], str]:
    if omics_type in ["DNA", "RNA"]:
        omics_type, species, task_name, split = parse_dna_rna_filename(filepath, omics_type)
    else:
        omics_type, species, task_name, split = parse_ptm_filename(filepath)

    df = read_label_seq_file(filepath)

    samples: List[Sample] = []
    for i, row in df.iterrows():
        seq = str(row["seq"]).strip()
        if not seq or seq.lower() == "nan":
            continue

        try:
            label = int(row["label"])
        except Exception:
            continue

        if label not in [0, 1]:
            continue

        samples.append(
            Sample(
                sequence=seq,
                label=label,
                omics_type=omics_type,
                species=species,
                task_name=task_name,
                center_position=infer_center_position(seq, omics_type),
                sample_id=f"{Path(filepath).stem}_{i}",
            )
        )

    return samples, split


def load_all_datasets(
    dna_dir: str,
    rna_dir: str,
    ptm_dir: str,
) -> Tuple[List[Sample], List[Sample]]:
    train_samples: List[Sample] = []
    valid_samples: List[Sample] = []

    dna_files = sorted(glob.glob(os.path.join(dna_dir, "*.csv"))) + sorted(glob.glob(os.path.join(dna_dir, "*.tsv")))
    rna_files = sorted(glob.glob(os.path.join(rna_dir, "*.csv"))) + sorted(glob.glob(os.path.join(rna_dir, "*.tsv")))
    ptm_files = sorted(glob.glob(os.path.join(ptm_dir, "*.csv"))) + sorted(glob.glob(os.path.join(ptm_dir, "*.tsv")))

    for fp in dna_files:
        samples, split = load_one_dataset_file(fp, omics_type="DNA")
        if split == "train":
            train_samples.extend(samples)
        elif split == "valid":
            valid_samples.extend(samples)

    for fp in rna_files:
        samples, split = load_one_dataset_file(fp, omics_type="RNA")
        if split == "train":
            train_samples.extend(samples)
        elif split == "valid":
            valid_samples.extend(samples)

    for fp in ptm_files:
        samples, split = load_one_dataset_file(fp, omics_type=None)
        if split == "train":
            train_samples.extend(samples)
        elif split == "valid":
            valid_samples.extend(samples)

    return train_samples, valid_samples


def inspect_data(samples: List[Sample], name: str) -> None:
    print(f"\n[{name}] total = {len(samples)}")
    print("omics:", Counter([s.omics_type for s in samples]))
    print("labels:", Counter([s.label for s in samples]))
    print("tasks:", Counter([s.task_name for s in samples]))
    print("species:", Counter([s.species for s in samples]))
    print("seq_len(top10):", Counter([len(s.sequence) for s in samples]).most_common(10))


# ============================================================
# 7. Dataset
# ============================================================

class EpiLingoDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        species2id: Dict[str, int],
        task2id: Dict[str, int],
        dna_rna_tokenizer,
        protein_tokenizer,
        cfg: ModelConfig,
    ) -> None:
        self.samples = samples
        self.species2id = species2id
        self.task2id = task2id
        self.dna_rna_tokenizer = dna_rna_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _prepare_protein_sequence(seq: str) -> str:
        seq = seq.upper().replace("U", "X").replace("Z", "X").replace("O", "X")
        return " ".join(list(seq))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        omics_id = OMICS2ID[s.omics_type]
        species_id = self.species2id[s.species]
        task_id = self.task2id[s.task_name]

        if s.omics_type in {"DNA", "RNA"}:
            text = seq_to_kmer(s.sequence, self.cfg.kmer)
            encoding = self.dna_rna_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.cfg.max_length_dna_rna,
                return_tensors="pt",
            )
            max_len = self.cfg.max_length_dna_rna
        else:
            text = self._prepare_protein_sequence(s.sequence)
            encoding = self.protein_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.cfg.max_length_protein,
                return_tensors="pt",
            )
            max_len = self.cfg.max_length_protein

        center_idx = min(max(s.center_position, 1), max_len - 2)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(float(s.label), dtype=torch.float32),
            "omics_id": torch.tensor(omics_id, dtype=torch.long),
            "species_id": torch.tensor(species_id, dtype=torch.long),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "center_idx": torch.tensor(center_idx, dtype=torch.long),
            "sample_id": s.sample_id or f"sample_{idx}",
            "omics_type_name": s.omics_type,
            "species_name": s.species,
            "task_name_str": s.task_name,
        }


# ============================================================
# 8. Backbone helpers
# ============================================================

def freeze_all_parameters(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_last_n_transformer_layers(model: nn.Module, n: int) -> None:
    if n <= 0:
        return

    possible_paths = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
    ]

    target_layers = None
    for path in possible_paths:
        current = model
        ok = True
        for attr in path.split("."):
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            target_layers = current
            break

    if target_layers is None:
        for p in model.parameters():
            p.requires_grad = True
        return

    total = len(target_layers)
    start = max(0, total - n)
    for i in range(start, total):
        for p in target_layers[i].parameters():
            p.requires_grad = True


# ============================================================
# 9. Adapters and decoder
# ============================================================

class ConditionalBottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int, cond_dim: int, dropout: float = 0.1) -> None:
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
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return h * (1.0 + gamma) + beta


class AsymmetricContextPooling(nn.Module):
    def __init__(self, hidden_dim: int, window: int = 3) -> None:
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
        batch_indices = torch.arange(h.size(0), device=h.device)
        return h[batch_indices, center_idx]


class ConditionalDecoder(nn.Module):
    def __init__(self, seq_dim: int, cond_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
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
        gate_input = torch.cat([seq_feat, cond], dim=-1)
        gate = self.gate(gate_input)
        gated_seq = seq_feat * gate
        logits = self.classifier(torch.cat([gated_seq, cond], dim=-1)).squeeze(-1)
        return logits


# ============================================================
# 10. Main model
# ============================================================

class EpiLingoHierarchical(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.dna_rna_backbone = AutoModel.from_pretrained(cfg.dna_rna_backbone_name)
        self.protein_backbone = AutoModel.from_pretrained(cfg.protein_backbone_name)

        dna_hidden = self.dna_rna_backbone.config.hidden_size
        protein_hidden = self.protein_backbone.config.hidden_size

        print(f"DNA/RNA backbone hidden size: {dna_hidden}")
        print(f"Protein backbone hidden size: {protein_hidden}")

        if cfg.freeze_backbone:
            freeze_all_parameters(self.dna_rna_backbone)
            freeze_all_parameters(self.protein_backbone)
            unfreeze_last_n_transformer_layers(self.dna_rna_backbone, cfg.unfreeze_last_n_layers)
            unfreeze_last_n_transformer_layers(self.protein_backbone, cfg.unfreeze_last_n_layers)

        # Project both branches to same shared dim
        self.dna_proj = nn.Linear(dna_hidden, cfg.hidden_dim) if dna_hidden != cfg.hidden_dim else nn.Identity()
        self.protein_proj = nn.Linear(protein_hidden, cfg.hidden_dim) if protein_hidden != cfg.hidden_dim else nn.Identity()

        self.omics_embedding = nn.Embedding(cfg.num_omics_types, cfg.omics_emb_dim)
        self.species_embedding = nn.Embedding(cfg.num_species, cfg.species_emb_dim)
        self.task_embedding = nn.Embedding(cfg.num_tasks, cfg.task_emb_dim)

        total_cond_dim = cfg.omics_emb_dim + cfg.species_emb_dim + cfg.task_emb_dim

        self.omics_adapter = ConditionalBottleneckAdapter(cfg.hidden_dim, cfg.adapter_bottleneck_dim, cfg.omics_emb_dim, cfg.dropout)
        self.species_adapter = ConditionalBottleneckAdapter(cfg.hidden_dim, cfg.adapter_bottleneck_dim, cfg.species_emb_dim, cfg.dropout)
        self.task_adapter = ConditionalBottleneckAdapter(cfg.hidden_dim, cfg.adapter_bottleneck_dim, cfg.task_emb_dim, cfg.dropout)
        self.film = FeatureFiLM(cfg.hidden_dim, total_cond_dim)

        if cfg.use_asymmetric_context:
            self.pooler = AsymmetricContextPooling(cfg.hidden_dim, cfg.context_window_tokens)
            seq_feat_dim = cfg.hidden_dim * 3
        else:
            self.pooler = CenterTokenPooling()
            seq_feat_dim = cfg.hidden_dim

        self.decoder = ConditionalDecoder(seq_feat_dim, total_cond_dim, cfg.decoder_hidden_dim, cfg.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        omics_id: torch.Tensor,
        species_id: torch.Tensor,
        task_id: torch.Tensor,
        center_idx: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:

        unique = torch.unique(omics_id)
        if len(unique) != 1:
            raise ValueError("Each mini-batch must contain exactly one omics type. Use OmicsGroupedBatchSampler.")

        oid = int(unique[0].item())

        if oid in [OMICS2ID["DNA"], OMICS2ID["RNA"]]:
            outputs = self.dna_rna_backbone(input_ids=input_ids, attention_mask=attention_mask)
            h = self.dna_proj(outputs.last_hidden_state)
        else:
            outputs = self.protein_backbone(input_ids=input_ids, attention_mask=attention_mask)
            h = self.protein_proj(outputs.last_hidden_state)

        e_omics = self.omics_embedding(omics_id)
        e_species = self.species_embedding(species_id)
        e_task = self.task_embedding(task_id)

        h = self.omics_adapter(h, e_omics)
        h = self.species_adapter(h, e_species)
        h = self.task_adapter(h, e_task)

        cond = torch.cat([e_omics, e_species, e_task], dim=-1)
        h = self.film(h, cond)

        seq_feat = self.pooler(h, center_idx)
        logits = self.decoder(seq_feat, cond)
        probs = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probs": probs,
        }
        if return_features:
            result["seq_feat"] = seq_feat
            result["cond"] = cond
        return result


# ============================================================
# 11. Losses
# ============================================================

def binary_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def species_consistency_loss(seq_feat: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    count = 0
    bsz = seq_feat.size(0)

    for i in range(bsz):
        for j in range(i + 1, bsz):
            if task_ids[i] == task_ids[j] and labels[i] == labels[j] and species_ids[i] != species_ids[j]:
                loss = loss + F.mse_loss(seq_feat[i], seq_feat[j])
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=seq_feat.device)
    return loss / count


def supervised_contrastive_task_loss(seq_feat: torch.Tensor, task_ids: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    feat = F.normalize(seq_feat, dim=-1)
    sim = torch.matmul(feat, feat.t()) / temperature

    labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
    task_eq = task_ids.unsqueeze(1) == task_ids.unsqueeze(0)
    positive_mask = labels_eq & task_eq

    eye = torch.eye(seq_feat.size(0), device=seq_feat.device, dtype=torch.bool)
    positive_mask = positive_mask & (~eye)
    logits_mask = ~eye

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / (positive_mask.float().sum(dim=1) + 1e-12)
    return -mean_log_prob_pos.mean()


# ============================================================
# 12. Metrics and curve saving
# ============================================================

def binary_metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    denom = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "AUROC": auc,
    }


def safe_dataset_name(name: str) -> str:
    return name.replace("|", "__").replace("/", "_").replace(" ", "_")


def save_overall_curve_data(
    labels: np.ndarray,
    probs: np.ndarray,
    out_dir: str = "overall_curve_data",
    prefix: str = "overall_validation",
) -> Dict[str, str]:
    """
    Save overall ROC and PR curve data only (no plotting).
    """
    os.makedirs(out_dir, exist_ok=True)

    labels = labels.astype(int)
    probs = probs.astype(float)

    result_paths = {
        "roc_csv": "",
        "pr_csv": "",
        "summary_csv": "",
    }

    if len(np.unique(labels)) < 2:
        print("Warning: overall labels contain only one class; skipping overall ROC/PR saving.")
        return result_paths

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": roc_thresholds,
    })
    roc_csv = os.path.join(out_dir, f"{prefix}_roc.csv")
    roc_df.to_csv(roc_csv, index=False)
    result_paths["roc_csv"] = roc_csv

    # PR
    precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
    threshold_padded = np.full(len(precision), np.nan)
    if len(pr_thresholds) > 0:
        threshold_padded[:len(pr_thresholds)] = pr_thresholds

    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": threshold_padded,
    })
    pr_csv = os.path.join(out_dir, f"{prefix}_pr.csv")
    pr_df.to_csv(pr_csv, index=False)
    result_paths["pr_csv"] = pr_csv

    # Summary
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(labels, probs))
    except Exception:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(labels, probs))
    except Exception:
        auprc = float("nan")

    summary_df = pd.DataFrame([{
        "prefix": prefix,
        "n_samples": int(len(labels)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "AUROC": auroc,
        "AUPRC": auprc,
        "roc_csv": roc_csv,
        "pr_csv": pr_csv,
    }])
    summary_csv = os.path.join(out_dir, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    result_paths["summary_csv"] = summary_csv

    return result_paths


def save_per_dataset_curve_data(
    df_records: pd.DataFrame,
    out_dir: str = "per_dataset_curve_data"
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for dataset_name, sub_df in df_records.groupby("dataset"):
        labels = sub_df["label"].values.astype(int)
        probs = sub_df["prob"].values.astype(float)

        safe_name = safe_dataset_name(dataset_name)
        roc_file = os.path.join(out_dir, f"{safe_name}_roc.csv")
        pr_file = os.path.join(out_dir, f"{safe_name}_pr.csv")

        roc_saved = 0
        pr_saved = 0
        auroc = float("nan")
        auprc = float("nan")

        if len(np.unique(labels)) >= 2:
            fpr, tpr, roc_thresholds = roc_curve(labels, probs)
            roc_df = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "threshold": roc_thresholds,
            })
            roc_df.to_csv(roc_file, index=False)
            roc_saved = 1

            precision, recall, pr_thresholds = precision_recall_curve(labels, probs)
            auprc = float(average_precision_score(labels, probs))

            threshold_padded = np.full(len(precision), np.nan)
            if len(pr_thresholds) > 0:
                threshold_padded[:len(pr_thresholds)] = pr_thresholds

            pr_df = pd.DataFrame({
                "precision": precision,
                "recall": recall,
                "threshold": threshold_padded,
            })
            pr_df.to_csv(pr_file, index=False)
            pr_saved = 1

            try:
                from sklearn.metrics import roc_auc_score
                auroc = float(roc_auc_score(labels, probs))
            except Exception:
                auroc = float("nan")

        summary_rows.append({
            "dataset": dataset_name,
            "omics_type": sub_df["omics_type"].iloc[0],
            "task_name": sub_df["task_name"].iloc[0],
            "species_name": sub_df["species_name"].iloc[0],
            "n_samples": int(len(sub_df)),
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
            "AUROC": auroc,
            "AUPRC": auprc,
            "roc_saved": roc_saved,
            "pr_saved": pr_saved,
            "roc_file": roc_file if roc_saved == 1 else "",
            "pr_file": pr_file if pr_saved == 1 else "",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "curve_data_summary.csv"), index=False)
    return summary_df


# ============================================================
# 13. Trainer
# ============================================================

class EpiLingoTrainer:
    def __init__(self, model: EpiLingoHierarchical, train_cfg: TrainConfig, model_cfg: ModelConfig) -> None:
        self.model = model
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = torch.device(train_cfg.device)
        self.model.to(self.device)

        backbone_params = []
        head_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": train_cfg.learning_rate},
                {"params": head_params, "lr": train_cfg.learning_rate},
            ],
            weight_decay=train_cfg.weight_decay,
        )

        self.scheduler = None
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=train_cfg.mixed_precision and self.device.type == "cuda"
        )

    def build_scheduler(self, num_training_steps: int) -> None:
        num_warmup_steps = int(self.train_cfg.warmup_ratio * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return moved

    def compute_total_loss(self, output: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = output["logits"]
        labels = batch["label"]
        total_loss = binary_classification_loss(logits, labels)
        loss_dict = {"cls_loss": float(total_loss.detach().cpu())}

        if self.model_cfg.use_species_consistency_loss and "seq_feat" in output:
            sc_loss = species_consistency_loss(output["seq_feat"], batch["label"], batch["task_id"], batch["species_id"])
            total_loss = total_loss + self.model_cfg.species_consistency_weight * sc_loss
            loss_dict["species_consistency_loss"] = float(sc_loss.detach().cpu())

        if self.model_cfg.use_task_contrastive_loss and "seq_feat" in output:
            tc_loss = supervised_contrastive_task_loss(
                output["seq_feat"], batch["task_id"], batch["label"], self.model_cfg.contrastive_temperature
            )
            total_loss = total_loss + self.model_cfg.task_contrastive_weight * tc_loss
            loss_dict["task_contrastive_loss"] = float(tc_loss.detach().cpu())

        loss_dict["total_loss"] = float(total_loss.detach().cpu())
        return total_loss, loss_dict

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = []

        for step, batch in enumerate(loader, start=1):
            batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                self.device.type,
                enabled=self.train_cfg.mixed_precision and self.device.type == "cuda"
            ):
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    omics_id=batch["omics_id"],
                    species_id=batch["species_id"],
                    task_id=batch["task_id"],
                    center_idx=batch["center_idx"],
                    return_features=(self.model_cfg.use_species_consistency_loss or self.model_cfg.use_task_contrastive_loss),
                )
                loss, _ = self.compute_total_loss(output, batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss.append(float(loss.detach().cpu()))

            if step % self.train_cfg.logging_steps == 0:
                print(f"[Epoch {epoch} | Step {step}] train_loss = {np.mean(running_loss):.4f}")

        return {"train_loss": float(np.mean(running_loss))}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()

        all_probs, all_labels, all_losses = [], [], []
        dataset_records = []

        for batch in loader:
            omics_type_names = list(batch["omics_type_name"])
            species_names = list(batch["species_name"])
            task_name_strs = list(batch["task_name_str"])
            sample_ids = list(batch["sample_id"])

            batch = self._move_batch(batch)

            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                omics_id=batch["omics_id"],
                species_id=batch["species_id"],
                task_id=batch["task_id"],
                center_idx=batch["center_idx"],
                return_features=False,
            )

            loss = binary_classification_loss(output["logits"], batch["label"])
            all_losses.append(float(loss.detach().cpu()))

            logits = output["logits"].detach().cpu().numpy()
            probs = output["probs"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

            for i in range(len(probs)):
                dataset_name = f"{omics_type_names[i]}|{task_name_strs[i]}|{species_names[i]}"
                dataset_records.append({
                    "dataset": dataset_name,
                    "omics_type": omics_type_names[i],
                    "task_name": task_name_strs[i],
                    "species_name": species_names[i],
                    "sample_id": sample_ids[i],
                    "label": float(labels[i]),
                    "prob": float(probs[i]),
                    "logit": float(logits[i]),
                })

        probs_all = np.concatenate(all_probs)
        labels_all = np.concatenate(all_labels)

        overall_metrics = binary_metrics_from_probs(probs_all, labels_all)
        overall_metrics["eval_loss"] = float(np.mean(all_losses))

        try:
            overall_metrics["AUPRC"] = float(average_precision_score(labels_all.astype(int), probs_all))
        except Exception:
            overall_metrics["AUPRC"] = float("nan")

        df_records = pd.DataFrame(dataset_records)
        per_dataset_results = []

        for dataset_name, sub_df in df_records.groupby("dataset"):
            sub_probs = sub_df["prob"].values.astype(float)
            sub_labels = sub_df["label"].values.astype(int)
            sub_logits = sub_df["logit"].values.astype(float)

            sub_metrics = binary_metrics_from_probs(sub_probs, sub_labels)

            try:
                sub_metrics["AUPRC"] = float(average_precision_score(sub_labels, sub_probs))
            except Exception:
                sub_metrics["AUPRC"] = float("nan")

            sub_metrics["eval_loss"] = float(
                F.binary_cross_entropy_with_logits(
                    torch.tensor(sub_logits, dtype=torch.float32),
                    torch.tensor(sub_labels, dtype=torch.float32)
                ).item()
            )

            sub_metrics["dataset"] = dataset_name
            sub_metrics["omics_type"] = sub_df["omics_type"].iloc[0]
            sub_metrics["task_name"] = sub_df["task_name"].iloc[0]
            sub_metrics["species_name"] = sub_df["species_name"].iloc[0]
            sub_metrics["n_samples"] = int(len(sub_df))
            sub_metrics["n_pos"] = int((sub_labels == 1).sum())
            sub_metrics["n_neg"] = int((sub_labels == 0).sum())

            per_dataset_results.append(sub_metrics)

        per_dataset_df = pd.DataFrame(per_dataset_results)
        if not per_dataset_df.empty:
            per_dataset_df = per_dataset_df.sort_values(
                by=["omics_type", "task_name", "species_name"]
            ).reset_index(drop=True)

        return {
            "overall": overall_metrics,
            "per_dataset": per_dataset_df,
            "raw_predictions": df_records,
        }

    def fit(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None) -> None:
        total_steps = len(train_loader) * self.train_cfg.num_train_epochs
        self.build_scheduler(total_steps)

        best_auc = -1.0
        best_state = None

        for epoch in range(1, self.train_cfg.num_train_epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            print(f"[Epoch {epoch}] train_loss = {train_metrics['train_loss']:.4f}")

            if valid_loader is not None:
                valid_results = self.evaluate(valid_loader)
                valid_metrics = valid_results["overall"]
                print(
                    f"[Epoch {epoch}] valid_loss = {valid_metrics['eval_loss']:.4f}, "
                    f"AUROC = {valid_metrics['AUROC']:.4f}, ACC = {valid_metrics['ACC']:.4f}, "
                    f"F1 = {valid_metrics['F1']:.4f}, MCC = {valid_metrics['MCC']:.4f}"
                )

                if not math.isnan(valid_metrics["AUROC"]) and valid_metrics["AUROC"] > best_auc:
                    best_auc = valid_metrics["AUROC"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best model with AUROC = {best_auc:.4f}")


# ============================================================
# 14. Omics-aware batching
# ============================================================

class OmicsGroupedBatchSampler:
    """
    Keep each mini-batch within one omics type because the model routes each
    batch to one backbone branch.
    """
    def __init__(self, dataset: EpiLingoDataset, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.groups: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for idx, s in enumerate(dataset.samples):
            self.groups[OMICS2ID[s.omics_type]].append(idx)

    def __iter__(self):
        group_indices = {k: v[:] for k, v in self.groups.items()}

        if self.shuffle:
            for v in group_indices.values():
                random.shuffle(v)

        batches = []
        for indices in group_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return sum(math.ceil(len(v) / self.batch_size) for v in self.groups.values())


# ============================================================
# 15. Main
# ============================================================

def main() -> None:
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    set_seed(train_cfg.seed)

    data_root = "./data"
    dna_dir = os.path.join(data_root, "DNA")
    rna_dir = os.path.join(data_root, "RNA")
    ptm_dir = os.path.join(data_root, "PTM")

    train_samples, valid_samples = load_all_datasets(
        dna_dir=dna_dir,
        rna_dir=rna_dir,
        ptm_dir=ptm_dir,
    )

    inspect_data(train_samples, "train")
    inspect_data(valid_samples, "valid")

    all_samples = train_samples + valid_samples
    all_species = [s.species for s in all_samples]
    all_tasks = [s.task_name for s in all_samples]

    species2id = build_vocab(all_species)
    task2id = build_vocab(all_tasks)

    model_cfg.num_species = len(species2id)
    model_cfg.num_tasks = len(task2id)

    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Species2id: {species2id}")
    print(f"Task2id: {task2id}")
    print(f"Device: {train_cfg.device}")

    dna_rna_tokenizer = AutoTokenizer.from_pretrained(model_cfg.dna_rna_backbone_name)
    protein_tokenizer = AutoTokenizer.from_pretrained(model_cfg.protein_backbone_name, do_lower_case=False)

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
        batch_sampler=OmicsGroupedBatchSampler(train_dataset, batch_size=train_cfg.train_batch_size, shuffle=True),
        num_workers=train_cfg.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=OmicsGroupedBatchSampler(valid_dataset, batch_size=train_cfg.eval_batch_size, shuffle=False),
        num_workers=train_cfg.num_workers,
    )

    model = EpiLingoHierarchical(model_cfg)
    trainer = EpiLingoTrainer(model, train_cfg, model_cfg)

    trainer.fit(train_loader, valid_loader)

    # Save the best-loaded model and metadata for downstream external inference
    torch.save(trainer.model.state_dict(), "best_model.pt")
    torch.save({
        "species2id": species2id,
        "task2id": task2id,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
    }, "training_metadata.pt")
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "species2id": species2id,
        "task2id": task2id,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
    }, "full_checkpoint.pt")

    print("\nSaved model files:")
    print(" - best_model.pt")
    print(" - training_metadata.pt")
    print(" - full_checkpoint.pt")

    print("\nFinal validation metrics:")
    valid_results = trainer.evaluate(valid_loader)

    print("\n[Overall validation metrics]")
    for k, v in valid_results["overall"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Save overall ROC / PR curve data only
    overall_labels = valid_results["raw_predictions"]["label"].values.astype(int)
    overall_probs = valid_results["raw_predictions"]["prob"].values.astype(float)
    overall_curve_paths = save_overall_curve_data(
        overall_labels,
        overall_probs,
        out_dir="overall_curve_data",
        prefix="overall_validation",
    )

    print("\n[Per-dataset validation metrics]")
    per_dataset_df = valid_results["per_dataset"]
    if len(per_dataset_df) > 0:
        print(per_dataset_df.to_string(index=False))
        per_dataset_df.to_csv("per_dataset_validation_metrics.csv", index=False)
        valid_results["raw_predictions"].to_csv("per_dataset_raw_predictions.csv", index=False)

        curve_summary_df = save_per_dataset_curve_data(
            valid_results["raw_predictions"],
            out_dir="per_dataset_curve_data"
        )

        print("\nSaved:")
        print(" - per_dataset_validation_metrics.csv")
        print(" - per_dataset_raw_predictions.csv")
        print(" - per_dataset_curve_data/")
        print(" - per_dataset_curve_data/curve_data_summary.csv")
        print(" - overall_curve_data/")
        for k, v in overall_curve_paths.items():
            if v:
                print(f" - {v}")

        print("\n[Curve data summary]")
        print(curve_summary_df.to_string(index=False))
    else:
        print("No per-dataset results found.")
        print("\nSaved overall curve data:")
        for k, v in overall_curve_paths.items():
            if v:
                print(f" - {v}")


if __name__ == "__main__":
    main()