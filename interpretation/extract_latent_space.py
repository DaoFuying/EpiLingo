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


def load_framework_module(framework_path: str):
    framework_path = os.path.abspath(framework_path)
    module_name = Path(framework_path).stem

    spec = importlib.util.spec_from_file_location(module_name, framework_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    """
    Balanced subsampling from a list of Sample objects.
    Default grouping: task_name × species × label
    """
    rng = random.Random(seed)
    groups: Dict[tuple, List[Any]] = {}

    for s in samples:
        key = tuple(getattr(s, g) for g in group_by)
        groups.setdefault(key, []).append(s)

    subset = []
    for key, items in groups.items():
        if len(items) <= max_per_group:
            subset.extend(items)
        else:
            subset.extend(rng.sample(items, max_per_group))
    return subset


def filter_samples_by_omics(samples: List[Any], allowed_omics: Optional[List[str]] = None) -> List[Any]:
    if allowed_omics is None or len(allowed_omics) == 0:
        return samples
    return [s for s in samples if s.omics_type in allowed_omics]


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


@torch.no_grad()
def extract_features(
    model,
    dataset,
    batch_sampler_cls,
    batch_size: int,
    num_workers: int,
    device: str = "cuda",
):
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler_cls(dataset, batch_size=batch_size, shuffle=False),
        num_workers=num_workers,
    )

    model.eval()
    model.to(device)

    feature_list = []
    meta_rows = []

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

        if "seq_feat" not in out:
            raise RuntimeError("Model forward(return_features=True) did not return 'seq_feat'.")

        feats = out["seq_feat"].detach().cpu().numpy()
        probs = out["probs"].detach().cpu().numpy()
        logits = out["logits"].detach().cpu().numpy()
        labels = input_batch["label"].detach().cpu().numpy()

        feature_list.append(feats)

        for i in range(len(feats)):
            meta_rows.append({
                "sample_id": sample_ids[i],
                "label": int(labels[i]),
                "prob": float(probs[i]),
                "logit": float(logits[i]),
                "omics_type": omics_type_names[i],
                "species_name": species_names[i],
                "task_name": task_names[i],
            })

    X = np.concatenate(feature_list, axis=0)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df


def run_pca(X: np.ndarray, n_components: int = 2, seed: int = 42):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)
    return Z, pca


def run_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0, seed: int = 42):
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    Z = tsne.fit_transform(X)
    return Z, tsne


def run_umap(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42):
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    Z = reducer.fit_transform(X)
    return Z, reducer


def attach_embedding(meta_df: pd.DataFrame, Z: np.ndarray, prefix: str) -> pd.DataFrame:
    df = meta_df.copy()
    df[f"{prefix}1"] = Z[:, 0]
    df[f"{prefix}2"] = Z[:, 1]
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract latent space features from EpiLingo")
    parser.add_argument("--framework_path", type=str, default="EpilingoHierarchicalFramework.py")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./latent_space_results")

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "all"],
        help="Which split to extract features from."
    )
    parser.add_argument(
        "--omics",
        type=str,
        default="all",
        choices=["all", "dna_rna", "protein", "dna", "rna"],
        help="Which omics subset to use."
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=100,
        help="Max samples per task×species×label group; set <=0 to disable subsampling."
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_umap", action="store_true")
    parser.add_argument("--run_tsne", action="store_true")
    parser.add_argument("--run_pca", action="store_true")

    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    fw = load_framework_module(args.framework_path)

    # Reuse framework components
    ModelConfig = fw.ModelConfig
    load_all_datasets = fw.load_all_datasets
    build_vocab = fw.build_vocab
    EpiLingoDataset = fw.EpiLingoDataset
    OmicsGroupedBatchSampler = fw.OmicsGroupedBatchSampler
    EpiLingoHierarchical = fw.EpiLingoHierarchical
    AutoTokenizer = fw.AutoTokenizer

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

    if args.omics == "dna_rna":
        selected_samples = filter_samples_by_omics(selected_samples, ["DNA", "RNA"])
    elif args.omics == "protein":
        selected_samples = filter_samples_by_omics(selected_samples, ["Protein"])
    elif args.omics == "dna":
        selected_samples = filter_samples_by_omics(selected_samples, ["DNA"])
    elif args.omics == "rna":
        selected_samples = filter_samples_by_omics(selected_samples, ["RNA"])

    if args.max_per_group is not None and args.max_per_group > 0:
        selected_samples = sample_balanced_subset(
            selected_samples,
            max_per_group=args.max_per_group,
            group_by=("task_name", "species", "label"),
            seed=args.seed,
        )

    print(f"Selected samples: {len(selected_samples)}")

    # Build global vocab from all samples, consistent with training
    all_samples = train_samples + valid_samples
    all_species = [s.species for s in all_samples]
    all_tasks = [s.task_name for s in all_samples]

    species2id = build_vocab(all_species)
    task2id = build_vocab(all_tasks)

    # Load checkpoint
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

    # support both plain state_dict and wrapped checkpoint
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)

    X, meta_df = extract_features(
        model=model,
        dataset=dataset,
        batch_sampler_cls=OmicsGroupedBatchSampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    # Save raw features and metadata
    feature_path = os.path.join(args.output_dir, "features.npy")
    meta_path = os.path.join(args.output_dir, "metadata.csv")
    np.save(feature_path, X)
    meta_df.to_csv(meta_path, index=False)

    print(f"Saved: {feature_path}")
    print(f"Saved: {meta_path}")

    saved_any_embedding = False

    if args.run_pca:
        Z_pca, _ = run_pca(X, n_components=2, seed=args.seed)
        df_pca = attach_embedding(meta_df, Z_pca, prefix="PCA")
        pca_path = os.path.join(args.output_dir, "pca.csv")
        df_pca.to_csv(pca_path, index=False)
        print(f"Saved: {pca_path}")
        saved_any_embedding = True

    if args.run_tsne:
        Z_tsne, _ = run_tsne(X, n_components=2, perplexity=args.tsne_perplexity, seed=args.seed)
        df_tsne = attach_embedding(meta_df, Z_tsne, prefix="TSNE")
        tsne_path = os.path.join(args.output_dir, "tsne.csv")
        df_tsne.to_csv(tsne_path, index=False)
        print(f"Saved: {tsne_path}")
        saved_any_embedding = True

    if args.run_umap:
        Z_umap, _ = run_umap(
            X,
            n_components=2,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            seed=args.seed,
        )
        df_umap = attach_embedding(meta_df, Z_umap, prefix="UMAP")
        umap_path = os.path.join(args.output_dir, "umap.csv")
        df_umap.to_csv(umap_path, index=False)
        print(f"Saved: {umap_path}")
        saved_any_embedding = True

    if not saved_any_embedding:
        print("No dimensionality reduction was requested. Use --run_umap and/or --run_tsne and/or --run_pca.")


if __name__ == "__main__":
    main()