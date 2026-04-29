from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


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
# 1. helpers
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_balanced_subset(
    samples: List[Any],
    max_per_group: int = 100,
    group_by: tuple = ("task_name", "species", "label"),
    seed: int = 42,
) -> List[Any]:
    rng = random.Random(seed)
    groups: Dict[tuple, List[Any]] = {}

    for s in samples:
        key = tuple(getattr(s, g) for g in group_by)
        groups.setdefault(key, []).append(s)

    subset = []
    for _, items in groups.items():
        if len(items) <= max_per_group:
            subset.extend(items)
        else:
            subset.extend(rng.sample(items, max_per_group))
    return subset


def filter_samples(
    samples: List[Any],
    allowed_omics: Optional[List[str]] = None,
    task_name: Optional[str] = None,
    species_name: Optional[str] = None,
) -> List[Any]:
    out = samples
    if allowed_omics is not None and len(allowed_omics) > 0:
        out = [s for s in out if s.omics_type in allowed_omics]
    if task_name is not None:
        out = [s for s in out if s.task_name == task_name]
    if species_name is not None:
        out = [s for s in out if s.species == species_name]
    return out


def build_dataset(
    samples,
    species2id,
    task2id,
    dna_rna_tokenizer,
    protein_tokenizer,
    model_cfg,
    EpiLingoDataset,
):
    return EpiLingoDataset(
        samples=samples,
        species2id=species2id,
        task2id=task2id,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        cfg=model_cfg,
    )


def compute_hidden_states_before_pooling(
    model,
    input_ids,
    attention_mask,
    omics_id,
    species_id,
    task_id,
    OMICS2ID,
):
    unique = torch.unique(omics_id)
    if len(unique) != 1:
        raise ValueError("Each batch must contain exactly one omics type.")

    oid = int(unique[0].item())
    is_dna_rna = oid in [OMICS2ID["DNA"], OMICS2ID["RNA"]]
    if not is_dna_rna:
        raise ValueError("This script is for DNA/RNA only.")

    outputs = model.dna_rna_backbone(input_ids=input_ids, attention_mask=attention_mask)
    h = model.dna_proj(outputs.last_hidden_state)

    e_omics = model.omics_embedding(omics_id)
    e_species = model.species_embedding(species_id)
    e_task = model.task_embedding(task_id)

    h = model.omics_adapter(h, e_omics)
    h = model.species_adapter(h, e_species)
    h = model.task_adapter(h, e_task)

    cond = torch.cat([e_omics, e_species, e_task], dim=-1)
    h = model.film(h, cond)
    return h, cond


def decode_from_hidden(model, h, cond, center_idx):
    seq_feat = model.pooler(h, center_idx)
    logits = model.decoder(seq_feat, cond)
    probs = torch.sigmoid(logits)
    return logits, probs


def token_to_base_importance(token_imp: torch.Tensor, target_seq_len: int = 41) -> torch.Tensor:
    """
    Map token-level attribution to base-level attribution.
    For 41bp with DNABERT-6:
      token_len = 36, k=6
    """
    B, L = token_imp.shape
    if L == target_seq_len:
        return token_imp

    if L < target_seq_len:
        k = target_seq_len - L + 1
        out = []
        for b in range(B):
            pos_scores = torch.zeros(target_seq_len, device=token_imp.device, dtype=token_imp.dtype)
            counts = torch.zeros(target_seq_len, device=token_imp.device, dtype=token_imp.dtype)
            for t in range(L):
                start = t
                end = min(target_seq_len, t + k)
                pos_scores[start:end] += token_imp[b, t]
                counts[start:end] += 1.0
            pos_scores = pos_scores / torch.clamp(counts, min=1.0)
            out.append(pos_scores)
        return torch.stack(out, dim=0)

    return token_imp[:, :target_seq_len]


def compute_saliency(
    model,
    h: torch.Tensor,
    cond: torch.Tensor,
    center_idx: torch.Tensor,
    seq_len: int = 41,
):
    h = h.clone().detach().requires_grad_(True)
    logits, probs = decode_from_hidden(model, h, cond, center_idx)
    target = logits.sum()
    grad = torch.autograd.grad(target, h, retain_graph=False, create_graph=False)[0]
    token_saliency = torch.sum(torch.abs(grad), dim=-1)
    base_saliency = token_to_base_importance(token_saliency, target_seq_len=seq_len)
    return base_saliency, logits.detach(), probs.detach()


def compute_integrated_gradients(
    model,
    h: torch.Tensor,
    cond: torch.Tensor,
    center_idx: torch.Tensor,
    steps: int = 32,
    seq_len: int = 41,
):
    baseline = torch.zeros_like(h)
    total_grad = torch.zeros_like(h)

    for alpha in torch.linspace(0.0, 1.0, steps, device=h.device):
        h_step = baseline + alpha * (h - baseline)
        h_step.requires_grad_(True)

        logits, _ = decode_from_hidden(model, h_step, cond, center_idx)
        target = logits.sum()

        grad = torch.autograd.grad(target, h_step, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()

    avg_grad = total_grad / steps
    ig = (h - baseline) * avg_grad
    token_ig = torch.sum(torch.abs(ig), dim=-1)
    base_ig = token_to_base_importance(token_ig, target_seq_len=seq_len)
    return base_ig


def summarize_attribution(df: pd.DataFrame, prefix: str, group_col: Optional[str], seq_len: int = 41) -> pd.DataFrame:
    pos_cols = [f"{prefix}_pos_{i}" for i in range(seq_len)]

    if group_col is None:
        row = {"group": "overall", "n_samples": len(df)}
        for c in pos_cols:
            row[c] = df[c].mean()
        return pd.DataFrame([row])

    rows = []
    for name, sub in df.groupby(group_col):
        row = {"group": name, "n_samples": len(sub)}
        for c in pos_cols:
            row[c] = sub[c].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


# ============================================================
# 2. main extraction
# ============================================================

def extract_attribution_dna_rna(
    model,
    dataset,
    batch_sampler_cls,
    batch_size: int,
    num_workers: int,
    device: str,
    OMICS2ID: Dict[str, int],
    seq_len: int = 41,
    ig_steps: int = 32,
):
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler_cls(dataset, batch_size=batch_size, shuffle=False),
        num_workers=num_workers,
    )

    model.eval()
    model.to(device)

    rows = []

    for batch in loader:
        omics_type_names = list(batch["omics_type_name"])
        species_names = list(batch["species_name"])
        task_names = list(batch["task_name_str"])
        sample_ids = list(batch["sample_id"])

        input_batch = {}
        for k, v in batch.items():
            input_batch[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        input_ids = input_batch["input_ids"]
        attention_mask = input_batch["attention_mask"]
        omics_id = input_batch["omics_id"]
        species_id = input_batch["species_id"]
        task_id = input_batch["task_id"]
        center_idx = input_batch["center_idx"]
        labels = input_batch["label"].detach().cpu().numpy()

        h, cond = compute_hidden_states_before_pooling(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            omics_id=omics_id,
            species_id=species_id,
            task_id=task_id,
            OMICS2ID=OMICS2ID,
        )

        saliency, logits, probs = compute_saliency(
            model=model,
            h=h,
            cond=cond,
            center_idx=center_idx,
            seq_len=seq_len,
        )

        ig = compute_integrated_gradients(
            model=model,
            h=h,
            cond=cond,
            center_idx=center_idx,
            steps=ig_steps,
            seq_len=seq_len,
        )

        saliency_np = saliency.detach().cpu().numpy()
        ig_np = ig.detach().cpu().numpy()
        logits_np = logits.cpu().numpy()
        probs_np = probs.cpu().numpy()

        for i in range(saliency_np.shape[0]):
            row = {
                "sample_id": sample_ids[i],
                "label": int(labels[i]),
                "prob": float(probs_np[i]),
                "logit": float(logits_np[i]),
                "omics_type": omics_type_names[i],
                "species_name": species_names[i],
                "task_name": task_names[i],
            }
            for p in range(seq_len):
                row[f"saliency_pos_{p}"] = float(saliency_np[i, p])
                row[f"ig_pos_{p}"] = float(ig_np[i, p])
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 3. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DNA/RNA attribution analysis for EpiLingo")
    parser.add_argument("--framework_path", type=str, default="EpilingoHierarchicalFramework.py")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./dna_rna_attribution_analysis")

    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "all"])
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--species_name", type=str, default=None)
    parser.add_argument("--max_per_group", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=41)
    parser.add_argument("--ig_steps", type=int, default=32)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    fw = load_framework_module(args.framework_path)

    ModelConfig = fw.ModelConfig
    load_all_datasets = fw.load_all_datasets
    build_vocab = fw.build_vocab
    EpiLingoDataset = fw.EpiLingoDataset
    OmicsGroupedBatchSampler = fw.OmicsGroupedBatchSampler
    EpiLingoHierarchical = fw.EpiLingoHierarchical
    AutoTokenizer = fw.AutoTokenizer
    OMICS2ID = fw.OMICS2ID

    dna_dir = os.path.join(args.data_root, "DNA")
    rna_dir = os.path.join(args.data_root, "RNA")
    ptm_dir = os.path.join(args.data_root, "PTM")

    train_samples, valid_samples = load_all_datasets(
        dna_dir=dna_dir,
        rna_dir=rna_dir,
        ptm_dir=ptm_dir,
    )

    if args.split == "train":
        selected_samples = train_samples
    elif args.split == "valid":
        selected_samples = valid_samples
    else:
        selected_samples = train_samples + valid_samples

    selected_samples = filter_samples(
        selected_samples,
        allowed_omics=["DNA", "RNA"],
        task_name=args.task_name,
        species_name=args.species_name,
    )

    if args.max_per_group is not None and args.max_per_group > 0:
        selected_samples = sample_balanced_subset(
            selected_samples,
            max_per_group=args.max_per_group,
            group_by=("task_name", "species", "label"),
            seed=args.seed,
        )

    print(f"Selected DNA/RNA samples: {len(selected_samples)}")
    if len(selected_samples) == 0:
        raise RuntimeError("No DNA/RNA samples selected. Check filters.")

    all_samples = train_samples + valid_samples
    species2id = build_vocab([s.species for s in all_samples])
    task2id = build_vocab([s.task_name for s in all_samples])

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    if "model_cfg" in ckpt:
        model_cfg = ModelConfig(**ckpt["model_cfg"])
    else:
        model_cfg = ModelConfig()

    model_cfg.num_species = len(species2id)
    model_cfg.num_tasks = len(task2id)

    dna_rna_tokenizer = AutoTokenizer.from_pretrained(model_cfg.dna_rna_backbone_name)
    protein_tokenizer = AutoTokenizer.from_pretrained(model_cfg.protein_backbone_name, do_lower_case=False)

    dataset = build_dataset(
        selected_samples,
        species2id,
        task2id,
        dna_rna_tokenizer,
        protein_tokenizer,
        model_cfg,
        EpiLingoDataset,
    )

    model = EpiLingoHierarchical(model_cfg)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    model.eval()

    attribution_df = extract_attribution_dna_rna(
        model=model,
        dataset=dataset,
        batch_sampler_cls=OmicsGroupedBatchSampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        OMICS2ID=OMICS2ID,
        seq_len=args.seq_len,
        ig_steps=args.ig_steps,
    )

    per_sample_path = os.path.join(args.output_dir, "per_sample_attribution.csv")
    attribution_df.to_csv(per_sample_path, index=False)

    # summaries
    saliency_overall = summarize_attribution(attribution_df, "saliency", None, seq_len=args.seq_len)
    saliency_by_label = summarize_attribution(attribution_df, "saliency", "label", seq_len=args.seq_len)
    saliency_by_task = summarize_attribution(attribution_df, "saliency", "task_name", seq_len=args.seq_len)
    saliency_by_species = summarize_attribution(attribution_df, "saliency", "species_name", seq_len=args.seq_len)
    saliency_by_omics = summarize_attribution(attribution_df, "saliency", "omics_type", seq_len=args.seq_len)

    ig_overall = summarize_attribution(attribution_df, "ig", None, seq_len=args.seq_len)
    ig_by_label = summarize_attribution(attribution_df, "ig", "label", seq_len=args.seq_len)
    ig_by_task = summarize_attribution(attribution_df, "ig", "task_name", seq_len=args.seq_len)
    ig_by_species = summarize_attribution(attribution_df, "ig", "species_name", seq_len=args.seq_len)
    ig_by_omics = summarize_attribution(attribution_df, "ig", "omics_type", seq_len=args.seq_len)

    saliency_overall.to_csv(os.path.join(args.output_dir, "saliency_overall.csv"), index=False)
    saliency_by_label.to_csv(os.path.join(args.output_dir, "saliency_by_label.csv"), index=False)
    saliency_by_task.to_csv(os.path.join(args.output_dir, "saliency_by_task.csv"), index=False)
    saliency_by_species.to_csv(os.path.join(args.output_dir, "saliency_by_species.csv"), index=False)
    saliency_by_omics.to_csv(os.path.join(args.output_dir, "saliency_by_omics.csv"), index=False)

    ig_overall.to_csv(os.path.join(args.output_dir, "ig_overall.csv"), index=False)
    ig_by_label.to_csv(os.path.join(args.output_dir, "ig_by_label.csv"), index=False)
    ig_by_task.to_csv(os.path.join(args.output_dir, "ig_by_task.csv"), index=False)
    ig_by_species.to_csv(os.path.join(args.output_dir, "ig_by_species.csv"), index=False)
    ig_by_omics.to_csv(os.path.join(args.output_dir, "ig_by_omics.csv"), index=False)

    summary_df = pd.DataFrame([{
        "n_samples": int(len(attribution_df)),
        "split": args.split,
        "task_name": args.task_name if args.task_name is not None else "",
        "species_name": args.species_name if args.species_name is not None else "",
        "seq_len": args.seq_len,
        "ig_steps": args.ig_steps,
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
    }])
    summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)

    print(f"Saved: {per_sample_path}")
    print(f"Saved summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()