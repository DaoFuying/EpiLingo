from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import davies_bouldin_score, silhouette_score
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


def save_numpy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


# ============================================================
# 2. stage feature extraction
# ============================================================

@torch.no_grad()
def extract_stage_features(
    model,
    dataset,
    batch_sampler_cls,
    batch_size: int,
    num_workers: int,
    device: str = "cuda",
    OMICS2ID: Dict[str, int] = None,
):
    """
    Extract pooled representations at 5 stages:
      1. backbone_only
      2. projection
      3. omics_adapter
      4. species_adapter
      5. task_adapter
    """
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler_cls(dataset, batch_size=batch_size, shuffle=False),
        num_workers=num_workers,
    )

    model.eval()
    model.to(device)

    stage_names = [
        "backbone_only",
        "projection",
        "omics_adapter",
        "species_adapter",
        "task_adapter",
    ]
    stage_feature_lists = {k: [] for k in stage_names}
    meta_rows = []

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

        unique = torch.unique(omics_id)
        if len(unique) != 1:
            raise ValueError("Each batch must contain exactly one omics type.")

        oid = int(unique[0].item())
        is_dna_rna = oid in [OMICS2ID["DNA"], OMICS2ID["RNA"]]

        # backbone output
        if is_dna_rna:
            outputs = model.dna_rna_backbone(input_ids=input_ids, attention_mask=attention_mask)
            h_backbone = outputs.last_hidden_state
        else:
            outputs = model.protein_backbone(input_ids=input_ids, attention_mask=attention_mask)
            h_backbone = outputs.last_hidden_state

        # shared projection
        if is_dna_rna:
            h_proj = model.dna_proj(h_backbone)
        else:
            h_proj = model.protein_proj(h_backbone)

        # embeddings
        e_omics = model.omics_embedding(omics_id)
        e_species = model.species_embedding(species_id)
        e_task = model.task_embedding(task_id)
        cond = torch.cat([e_omics, e_species, e_task], dim=-1)

        # omics adapter
        h_omics = model.omics_adapter(h_proj, e_omics)

        # species adapter
        h_species = model.species_adapter(h_omics, e_species)

        # task adapter
        h_task = model.task_adapter(h_species, e_task)

        # note: pooled stage feature for each stage
        # backbone_only needs branch-specific pooling because dim differs before projection
        if is_dna_rna:
            if hasattr(model, "pooler"):
                # if pooler expects projected dim, backbone dim may mismatch
                # use center pooling for raw backbone stage to keep it stable
                stage_backbone = h_backbone[torch.arange(h_backbone.size(0), device=h_backbone.device), center_idx]
            else:
                stage_backbone = h_backbone[torch.arange(h_backbone.size(0), device=h_backbone.device), center_idx]
        else:
            stage_backbone = h_backbone[torch.arange(h_backbone.size(0), device=h_backbone.device), center_idx]

        # after projection and subsequent stages, use the same pooling rule as model
        if model.cfg.use_asymmetric_context:
            stage_projection = model.pooler(h_proj, center_idx)
            stage_omics = model.pooler(h_omics, center_idx)
            stage_species = model.pooler(h_species, center_idx)
            stage_task = model.pooler(h_task, center_idx)
        else:
            stage_projection = model.pooler(h_proj, center_idx)
            stage_omics = model.pooler(h_omics, center_idx)
            stage_species = model.pooler(h_species, center_idx)
            stage_task = model.pooler(h_task, center_idx)

        stage_feature_lists["backbone_only"].append(stage_backbone.detach().cpu().numpy())
        stage_feature_lists["projection"].append(stage_projection.detach().cpu().numpy())
        stage_feature_lists["omics_adapter"].append(stage_omics.detach().cpu().numpy())
        stage_feature_lists["species_adapter"].append(stage_species.detach().cpu().numpy())
        stage_feature_lists["task_adapter"].append(stage_task.detach().cpu().numpy())

        for i in range(len(labels)):
            meta_rows.append({
                "sample_id": sample_ids[i],
                "label": int(labels[i]),
                "omics_type": omics_type_names[i],
                "species_name": species_names[i],
                "task_name": task_names[i],
            })

    stage_features = {
        k: np.concatenate(v, axis=0) for k, v in stage_feature_lists.items()
    }
    meta_df = pd.DataFrame(meta_rows)
    return stage_features, meta_df


# ============================================================
# 3. metrics
# ============================================================

def compute_centroid_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    inter_class_distance:
      mean pairwise Euclidean distance between class centroids

    intra_class_compactness:
      mean Euclidean distance of samples to their own centroid
    """
    y = np.asarray(y)
    classes = np.unique(y)

    if len(classes) < 2:
        return {
            "inter_class_distance": np.nan,
            "intra_class_compactness": np.nan,
            "silhouette_score": np.nan,
            "davies_bouldin_index": np.nan,
            "n_classes": int(len(classes)),
        }

    centroids = []
    compactness_list = []

    for c in classes:
        Xc = X[y == c]
        centroid = Xc.mean(axis=0)
        centroids.append(centroid)

        d = np.sqrt(((Xc - centroid) ** 2).sum(axis=1))
        compactness_list.append(d.mean())

    centroids = np.stack(centroids, axis=0)

    # inter-class centroid distance
    pair_dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            d = np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
            pair_dists.append(d)
    inter_class_distance = float(np.mean(pair_dists)) if len(pair_dists) > 0 else np.nan

    intra_class_compactness = float(np.mean(compactness_list))

    # silhouette and DB index
    try:
        sil = float(silhouette_score(X, y))
    except Exception:
        sil = np.nan

    try:
        dbi = float(davies_bouldin_score(X, y))
    except Exception:
        dbi = np.nan

    return {
        "inter_class_distance": inter_class_distance,
        "intra_class_compactness": intra_class_compactness,
        "silhouette_score": sil,
        "davies_bouldin_index": dbi,
        "n_classes": int(len(classes)),
    }


def evaluate_all_stages(stage_features: Dict[str, np.ndarray], meta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    label_cols = ["omics_type", "species_name", "task_name"]

    for stage_name, X in stage_features.items():
        for label_col in label_cols:
            y = meta_df[label_col].values
            metrics = compute_centroid_metrics(X, y)

            rows.append({
                "stage": stage_name,
                "label_type": label_col,
                "n_samples": int(X.shape[0]),
                "feature_dim": int(X.shape[1]),
                **metrics,
            })

    return pd.DataFrame(rows)


# ============================================================
# 4. main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze hierarchical conditional adaptation stages in EpiLingo")
    parser.add_argument("--framework_path", type=str, default="EpilingoHierarchicalFramework.py")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./hierarchical_adaptation_analysis")

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "all"],
        help="Which split to use for analysis."
    )
    parser.add_argument(
        "--omics",
        type=str,
        default="all",
        choices=["all", "dna_rna", "protein", "dna", "rna"],
        help="Which omics subset to analyze."
    )
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--species_name", type=str, default=None)
    parser.add_argument("--max_per_group", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    stage_feature_dir = os.path.join(args.output_dir, "stage_features")
    os.makedirs(stage_feature_dir, exist_ok=True)

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

    omics_map = {
        "all": None,
        "dna_rna": ["DNA", "RNA"],
        "protein": ["Protein"],
        "dna": ["DNA"],
        "rna": ["RNA"],
    }

    selected_samples = filter_samples(
        selected_samples,
        allowed_omics=omics_map[args.omics],
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

    print(f"Selected samples: {len(selected_samples)}")
    if len(selected_samples) == 0:
        raise RuntimeError("No samples selected. Check your filters.")

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

    stage_features, meta_df = extract_stage_features(
        model=model,
        dataset=dataset,
        batch_sampler_cls=OmicsGroupedBatchSampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        OMICS2ID=OMICS2ID,
    )

    # save features
    for stage_name, X in stage_features.items():
        save_numpy(os.path.join(stage_feature_dir, f"{stage_name}_features.npy"), X)

    meta_path = os.path.join(stage_feature_dir, "metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    # evaluate metrics
    metrics_df = evaluate_all_stages(stage_features, meta_df)
    metrics_csv = os.path.join(args.output_dir, "stage_metrics.csv")
    #metrics_xlsx = os.path.join(args.output_dir, "stage_metrics.xlsx")
    summary_csv = os.path.join(args.output_dir, "summary.csv")

    metrics_df.to_csv(metrics_csv, index=False)
    #metrics_df.to_excel(metrics_xlsx, index=False)

    summary_df = pd.DataFrame([{
        "n_samples": int(len(meta_df)),
        "split": args.split,
        "omics": args.omics,
        "task_name": args.task_name if args.task_name is not None else "",
        "species_name": args.species_name if args.species_name is not None else "",
        "max_per_group": args.max_per_group,
        "checkpoint_path": os.path.abspath(args.checkpoint_path),
    }])
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved: {metrics_csv}")
    #print(f"Saved: {metrics_xlsx}")
    print(f"Saved: {summary_csv}")
    print(f"Saved stage features to: {stage_feature_dir}")
    print("\nStage metric summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()