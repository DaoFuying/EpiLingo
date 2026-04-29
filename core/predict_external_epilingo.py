from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from EpilingoHierarchicalFramework4 import (
    EpiLingoDataset,
    EpiLingoHierarchical,
    ModelConfig,
    Sample,
    binary_metrics_from_probs,
    set_seed,
)


ALLOWED_OMICS = {"DNA", "RNA", "Protein"}


def read_external_dataset(filepath: str) -> pd.DataFrame:
    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}

    seq_col = None
    for c in ["seq", "sequence", "sequences", "text"]:
        if c in colmap:
            seq_col = colmap[c]
            break
    if seq_col is None:
        raise ValueError("External file must contain one of: seq, sequence, sequences, text")

    out = pd.DataFrame()
    out["seq"] = df[seq_col].astype(str).str.strip()

    label_col = None
    for c in ["label", "labels", "y", "class", "target"]:
        if c in colmap:
            label_col = colmap[c]
            break
    out["label"] = df[label_col] if label_col is not None else np.nan

    sample_id_col = colmap.get("sample_id")
    if sample_id_col is not None:
        out["sample_id"] = df[sample_id_col].astype(str)
    else:
        out["sample_id"] = [f"ext_{i}" for i in range(len(df))]

    # keep any additional metadata columns for later merge-back
    used_original_cols = {seq_col}
    if label_col is not None:
        used_original_cols.add(label_col)
    if sample_id_col is not None:
        used_original_cols.add(sample_id_col)

    extra_cols = [c for c in df.columns if c not in used_original_cols]
    for c in extra_cols:
        out[c] = df[c]

    out = out[out["seq"].notna() & (out["seq"] != "") & (out["seq"].str.lower() != "nan")].reset_index(drop=True)
    return out



def build_external_samples(
    df: pd.DataFrame,
    omics_type: str,
    species: str,
    task_name: str,
) -> List[Sample]:
    samples: List[Sample] = []
    for _, row in df.iterrows():
        label = row["label"]
        if pd.isna(label):
            label_int = 0  # placeholder for inference-only usage
        else:
            label_int = int(label)

        seq = str(row["seq"]).strip()
        samples.append(
            Sample(
                sequence=seq,
                label=label_int,
                omics_type=omics_type,
                species=species,
                task_name=task_name,
                center_position=len(seq) // 2,
                sample_id=str(row["sample_id"]),
            )
        )
    return samples


@torch.no_grad()
def predict_external(
    model: EpiLingoHierarchical,
    loader: DataLoader,
    device: str,
) -> pd.DataFrame:
    model.eval()
    model.to(device)

    rows: List[Dict[str, Any]] = []

    for batch in loader:
        sample_ids = list(batch["sample_id"])
        omics_type_names = list(batch["omics_type_name"])
        species_names = list(batch["species_name"])
        task_name_strs = list(batch["task_name_str"])

        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        output = model(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            omics_id=moved["omics_id"],
            species_id=moved["species_id"],
            task_id=moved["task_id"],
            center_idx=moved["center_idx"],
            return_features=False,
        )

        logits = output["logits"].detach().cpu().numpy()
        probs = output["probs"].detach().cpu().numpy()
        labels = moved["label"].detach().cpu().numpy()

        for i in range(len(probs)):
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "omics_type": omics_type_names[i],
                    "species_name": species_names[i],
                    "task_name": task_name_strs[i],
                    "label": float(labels[i]),
                    "logit": float(logits[i]),
                    "prob": float(probs[i]),
                    "pred_label_0.5": int(probs[i] >= 0.5),
                }
            )

    return pd.DataFrame(rows)



def summarize_metrics(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    if "label" not in df.columns or df["label"].isna().all():
        return pd.DataFrame()

    eval_df = df.dropna(subset=["label"]).copy()
    if eval_df.empty:
        return pd.DataFrame()

    eval_df["label"] = eval_df["label"].astype(int)

    groups = [("overall", eval_df)] if group_col is None else list(eval_df.groupby(group_col, dropna=False))

    rows = []
    for gname, sub_df in groups:
        labels = sub_df["label"].values.astype(int)
        probs = sub_df["prob"].values.astype(float)

        row: Dict[str, Any] = {
            "group": gname,
            "n_samples": int(len(sub_df)),
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
        }

        if len(np.unique(labels)) < 2:
            row.update({
                "ACC": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "MCC": np.nan,
                "AUROC": np.nan,
                "AUPRC": np.nan,
            })
        else:
            m = binary_metrics_from_probs(probs, labels)
            try:
                auprc = float(average_precision_score(labels, probs))
            except Exception:
                auprc = np.nan
            row.update(m)
            row["AUPRC"] = auprc

        rows.append(row)

    return pd.DataFrame(rows)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict external data with a trained EpiLingo model")
    parser.add_argument("--external_file", required=True, help="Path to external CSV/TSV file")
    parser.add_argument("--checkpoint_path", default="best_model.pt", help="Path to best_model.pt")
    parser.add_argument("--metadata_path", default="training_metadata.pt", help="Path to training_metadata.pt")
    parser.add_argument("--output_dir", default="external_prediction_results", help="Directory to save outputs")

    parser.add_argument("--omics_type", required=True, choices=sorted(ALLOWED_OMICS), help="DNA, RNA, or Protein")
    parser.add_argument("--species", required=True, help="Species name exactly matching training metadata, e.g. Arabidopsis_thaliana")
    parser.add_argument("--task_name", required=True, help="Task name exactly matching training metadata, e.g. 6mA")

    parser.add_argument("--group_col", default=None, help="Optional column in external_file for grouped metrics, e.g. stage")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.external_file):
        raise FileNotFoundError(f"external_file not found: {args.external_file}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"checkpoint_path not found: {args.checkpoint_path}")
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"metadata_path not found: {args.metadata_path}")

    metadata = torch.load(args.metadata_path, map_location="cpu")
    species2id = metadata["species2id"]
    task2id = metadata["task2id"]
    model_cfg_dict = metadata.get("model_cfg", {})

    if args.species not in species2id:
        raise KeyError(
            f"Species '{args.species}' not found in training metadata. Available species: {sorted(species2id.keys())}"
        )
    if args.task_name not in task2id:
        raise KeyError(
            f"Task '{args.task_name}' not found in training metadata. Available tasks: {sorted(task2id.keys())}"
        )

    model_cfg = ModelConfig(**model_cfg_dict) if model_cfg_dict else ModelConfig()
    model_cfg.num_species = len(species2id)
    model_cfg.num_tasks = len(task2id)

    dna_rna_tokenizer = AutoTokenizer.from_pretrained(model_cfg.dna_rna_backbone_name)
    protein_tokenizer = AutoTokenizer.from_pretrained(model_cfg.protein_backbone_name, do_lower_case=False)

    ext_df = read_external_dataset(args.external_file)
    ext_samples = build_external_samples(
        ext_df,
        omics_type=args.omics_type,
        species=args.species,
        task_name=args.task_name,
    )

    ext_dataset = EpiLingoDataset(
        samples=ext_samples,
        species2id=species2id,
        task2id=task2id,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        cfg=model_cfg,
    )

    ext_loader = DataLoader(
        ext_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = EpiLingoHierarchical(model_cfg)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    pred_df = predict_external(model, ext_loader, device=args.device)
    merged_df = ext_df.merge(pred_df, on="sample_id", how="left")

    pred_file = os.path.join(args.output_dir, "external_predictions.csv")
    merged_df.to_csv(pred_file, index=False)

    print(f"Saved predictions: {pred_file}")
    print("Preview:")
    print(merged_df.head().to_string(index=False))

    # metrics only if labels were truly provided in the original file
    label_was_provided = "label" in ext_df.columns and ext_df["label"].notna().any()
    if label_was_provided:
        overall_df = summarize_metrics(merged_df, group_col=None)
        overall_path = os.path.join(args.output_dir, "external_overall_metrics.csv")
        overall_df.to_csv(overall_path, index=False)
        print(f"Saved overall metrics: {overall_path}")
        print(overall_df.to_string(index=False))

        if args.group_col is not None:
            if args.group_col not in merged_df.columns:
                raise KeyError(f"group_col '{args.group_col}' not found in external file columns: {list(merged_df.columns)}")
            grouped_df = summarize_metrics(merged_df, group_col=args.group_col)
            grouped_path = os.path.join(args.output_dir, f"external_metrics_by_{args.group_col}.csv")
            grouped_df.to_csv(grouped_path, index=False)
            print(f"Saved grouped metrics: {grouped_path}")
            print(grouped_df.to_string(index=False))
    else:
        print("No valid label column detected in external file. Only prediction probabilities were saved.")


if __name__ == "__main__":
    main()
