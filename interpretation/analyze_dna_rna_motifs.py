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


def decode_from_hidden(model, h, cond, center_idx):
    seq_feat = model.pooler(h, center_idx)
    logits = model.decoder(seq_feat, cond)
    probs = torch.sigmoid(logits)
    return logits, probs


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


def token_to_base_importance(token_imp: torch.Tensor, target_seq_len: int = 41) -> torch.Tensor:
    """
    Map token-level importance back to 41 base positions.
    For DNABERT-6 with 41bp input:
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


def integrated_gradients_hidden(
    model,
    h: torch.Tensor,
    cond: torch.Tensor,
    center_idx: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """
    Integrated gradients on hidden states h (continuous space) before pooling.
    baseline = zero tensor
    Returns token-level attribution [B, L]
    """
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
    token_imp = torch.sum(torch.abs(ig), dim=-1)  # [B, L]
    return token_imp


def extract_top_fragment(seq: str, importance: np.ndarray, motif_len: int = 7) -> Dict[str, Any]:
    assert len(seq) == len(importance), f"Sequence length {len(seq)} != importance length {len(importance)}"

    half = motif_len // 2
    center_pos = int(np.argmax(importance))
    start = max(0, center_pos - half)
    end = min(len(seq), start + motif_len)
    start = max(0, end - motif_len)

    frag = seq[start:end]
    frag_imp = importance[start:end]

    return {
        "top_pos": center_pos,
        "fragment_start": start,
        "fragment_end": end,
        "fragment_seq": frag,
        "fragment_mean_importance": float(np.mean(frag_imp)),
        "fragment_max_importance": float(np.max(frag_imp)),
    }


# ============================================================
# 2. main extraction
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    dataset,
    batch_sampler_cls,
    batch_size: int,
    num_workers: int,
    device: str,
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

        out = model(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            omics_id=input_batch["omics_id"],
            species_id=input_batch["species_id"],
            task_id=input_batch["task_id"],
            center_idx=input_batch["center_idx"],
            return_features=True,
        )

        probs = out["probs"].detach().cpu().numpy()
        logits = out["logits"].detach().cpu().numpy()
        labels = input_batch["label"].detach().cpu().numpy()

        for i in range(len(probs)):
            rows.append({
                "sample_id": sample_ids[i],
                "label": int(labels[i]),
                "prob": float(probs[i]),
                "logit": float(logits[i]),
                "omics_type": omics_type_names[i],
                "species_name": species_names[i],
                "task_name": task_names[i],
            })
    return pd.DataFrame(rows)


def extract_dna_rna_motif_importance(
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

        token_imp = integrated_gradients_hidden(
            model=model,
            h=h,
            cond=cond,
            center_idx=center_idx,
            steps=ig_steps,
        )

        base_imp = token_to_base_importance(token_imp, target_seq_len=seq_len).detach().cpu().numpy()

        # probs/logits from actual hidden
        logits, probs = decode_from_hidden(model, h, cond, center_idx)
        probs_np = probs.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        # sequence from dataset samples
        # dataset.samples order matches DataLoader order because shuffle=False inside batch sampler
        # but easiest is to recover original sequence from dataset sample_id mapping
        for i in range(base_imp.shape[0]):
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
                row[f"pos_{p}"] = float(base_imp[i, p])
            rows.append(row)

    return pd.DataFrame(rows)


def create_sampleid_to_sequence(samples: List[Any]) -> Dict[str, str]:
    return {s.sample_id: s.sequence for s in samples if s.sample_id is not None}


# ============================================================
# 3. optional logo
# ============================================================

def try_plot_logo(sequence_file: str, output_png: str):
    try:
        import logomaker
        import matplotlib.pyplot as plt
        from collections import Counter

        seqs = []
        with open(sequence_file, "r") as f:
            for line in f:
                line = line.strip().upper()
                if line:
                    seqs.append(line)

        if len(seqs) == 0:
            return False

        motif_len = len(seqs[0])
        chars = ["A", "C", "G", "T"]
        mat = []
        for i in range(motif_len):
            counter = Counter([s[i] for s in seqs if len(s) == motif_len])
            row = {c: counter.get(c, 0) for c in chars}
            total = sum(row.values())
            if total > 0:
                row = {k: v / total for k, v in row.items()}
            mat.append(row)

        df = pd.DataFrame(mat)
        fig, ax = plt.subplots(figsize=(max(5, motif_len * 0.6), 3.5))
        logomaker.Logo(df, ax=ax)
        ax.set_xlabel("Position", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title("Sequence logo of high-contribution motifs", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


# ============================================================
# 4. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="High-contribution motif analysis for DNA/RNA EpiLingo models")
    parser.add_argument("--framework_path", type=str, default="EpilingoHierarchicalFramework.py")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./dna_rna_motif_analysis")

    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "all"])
    parser.add_argument("--task_name", type=str, default=None, help="Optional: restrict to one task, e.g. 6mA or m6A")
    parser.add_argument("--species_name", type=str, default=None, help="Optional: restrict to one species")
    parser.add_argument("--max_per_group", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--seq_len", type=int, default=41)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    parser.add_argument("--positive_only", action="store_true", help="Keep only label=1 high-confidence samples")
    parser.add_argument("--motif_len", type=int, default=7)
    parser.add_argument("--top_n_fragments", type=int, default=500, help="Keep top N fragments by fragment_mean_importance")

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

    # position-level attribution
    importance_df = extract_dna_rna_motif_importance(
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
    importance_path = os.path.join(args.output_dir, "per_sample_base_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    # high-confidence selection
    high_conf_df = importance_df[importance_df["prob"] >= args.confidence_threshold].copy()
    if args.positive_only:
        high_conf_df = high_conf_df[high_conf_df["label"] == 1].copy()

    high_conf_df = high_conf_df.sort_values("prob", ascending=False).reset_index(drop=True)
    high_conf_path = os.path.join(args.output_dir, "high_confidence_samples.csv")
    high_conf_df.to_csv(high_conf_path, index=False)

    print(f"High-confidence samples: {len(high_conf_df)}")

    # map sample_id -> sequence
    sampleid_to_seq = create_sampleid_to_sequence(selected_samples)

    pos_cols = [f"pos_{i}" for i in range(args.seq_len)]
    fragment_rows = []

    for _, row in high_conf_df.iterrows():
        sid = row["sample_id"]
        seq = sampleid_to_seq.get(sid, None)
        if seq is None:
            continue

        imp = row[pos_cols].values.astype(float)
        frag_info = extract_top_fragment(seq, imp, motif_len=args.motif_len)

        fragment_rows.append({
            "sample_id": sid,
            "label": int(row["label"]),
            "prob": float(row["prob"]),
            "logit": float(row["logit"]),
            "omics_type": row["omics_type"],
            "species_name": row["species_name"],
            "task_name": row["task_name"],
            **frag_info,
        })

    frag_df = pd.DataFrame(fragment_rows)
    if len(frag_df) > 0:
        frag_df = frag_df.sort_values(
            by=["fragment_mean_importance", "prob"],
            ascending=[False, False]
        ).reset_index(drop=True)

    if args.top_n_fragments is not None and args.top_n_fragments > 0 and len(frag_df) > args.top_n_fragments:
        frag_df = frag_df.head(args.top_n_fragments).copy()

    frag_path = os.path.join(args.output_dir, "top_fragments.tsv")
    frag_df.to_csv(frag_path, sep="\t", index=False)

    # save sequence-logo input
    logo_input_path = os.path.join(args.output_dir, "sequence_logo_input.txt")
    with open(logo_input_path, "w") as f:
        for seq in frag_df["fragment_seq"].tolist():
            f.write(seq + "\n")

    # summary
    summary_df = pd.DataFrame([{
        "n_selected_samples": int(len(selected_samples)),
        "n_high_confidence_samples": int(len(high_conf_df)),
        "n_fragments_saved": int(len(frag_df)),
        "split": args.split,
        "task_name": args.task_name if args.task_name is not None else "",
        "species_name": args.species_name if args.species_name is not None else "",
        "confidence_threshold": args.confidence_threshold,
        "positive_only": bool(args.positive_only),
        "motif_len": args.motif_len,
        "ig_steps": args.ig_steps,
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
    }])
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # optional logo
    logo_png = os.path.join(args.output_dir, "motif_logo.png")
    made_logo = try_plot_logo(logo_input_path, logo_png)

    print(f"Saved: {importance_path}")
    print(f"Saved: {high_conf_path}")
    print(f"Saved: {frag_path}")
    print(f"Saved: {logo_input_path}")
    print(f"Saved: {summary_path}")
    if made_logo:
        print(f"Saved: {logo_png}")
    else:
        print("Logo plot not created. Install logomaker if you want automatic motif_logo.png.")


if __name__ == "__main__":
    main()