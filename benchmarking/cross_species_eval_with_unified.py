from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_framework_module(framework_path: str):
    framework_path = os.path.abspath(framework_path)
    module_name = Path(framework_path).stem

    spec = importlib.util.spec_from_file_location(module_name, framework_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    # Important for dataclass/import stability
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def dataset_label(sample) -> str:
    return f"{sample.omics_type}|{sample.task_name}|{sample.species}"


def collect_dataset_map(samples: List[Any], allowed_omics: List[str]) -> Dict[str, List[Any]]:
    """
    Group samples by dataset label = omics|task|species
    """
    dataset_map: Dict[str, List[Any]] = {}
    for s in samples:
        if s.omics_type not in allowed_omics:
            continue
        key = dataset_label(s)
        dataset_map.setdefault(key, []).append(s)
    return dataset_map


def intersection_dataset_labels(train_map: Dict[str, List[Any]], valid_map: Dict[str, List[Any]]) -> List[str]:
    return sorted(set(train_map.keys()) & set(valid_map.keys()))


def build_dataset_from_samples(
    samples: List[Any],
    species2id: Dict[str, int],
    task2id: Dict[str, int],
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


def evaluate_on_dataset(
    trainer,
    dataset_samples,
    species2id,
    task2id,
    dna_rna_tokenizer,
    protein_tokenizer,
    model_cfg,
    EpiLingoDataset,
    OmicsGroupedBatchSampler,
    eval_batch_size,
    num_workers,
):
    dataset = build_dataset_from_samples(
        dataset_samples,
        species2id,
        task2id,
        dna_rna_tokenizer,
        protein_tokenizer,
        model_cfg,
        EpiLingoDataset,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=OmicsGroupedBatchSampler(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
        ),
        num_workers=num_workers,
    )
    return trainer.evaluate(loader)["overall"]


def save_metric_matrices(
    df: pd.DataFrame,
    prefix: str,
    out_dir: str,
    metrics: List[str] = ["AUROC", "F1", "MCC"],
):
    os.makedirs(out_dir, exist_ok=True)
    matrices = {}

    if df.empty:
        for metric in metrics:
            matrices[metric] = pd.DataFrame()
        return matrices

    row_order = sorted(df["test_dataset"].unique().tolist())
    col_order = sorted(df["model_name"].unique().tolist())

    for metric in metrics:
        mat = df.pivot(index="test_dataset", columns="model_name", values=metric)
        mat = mat.reindex(index=row_order, columns=col_order)
        mat.to_csv(os.path.join(out_dir, f"{prefix}_{metric}_matrix.csv"))
        matrices[metric] = mat

    return matrices


def run_group_evaluation(
    group_name: str,
    allowed_omics: List[str],
    train_samples: List[Any],
    valid_samples: List[Any],
    species2id: Dict[str, int],
    task2id: Dict[str, int],
    dna_rna_tokenizer,
    protein_tokenizer,
    model_cfg,
    train_cfg,
    EpiLingoDataset,
    OmicsGroupedBatchSampler,
    EpiLingoHierarchical,
    EpiLingoTrainer,
    output_dir: str,
):
    """
    group_name: "dna_rna" or "protein"
    allowed_omics: e.g. ["DNA", "RNA"] or ["Protein"]
    """

    train_map = collect_dataset_map(train_samples, allowed_omics)
    valid_map = collect_dataset_map(valid_samples, allowed_omics)

    dataset_labels = intersection_dataset_labels(train_map, valid_map)
    if len(dataset_labels) == 0:
        raise RuntimeError(f"No usable datasets found for group {group_name}")

    print(f"\n=== Group: {group_name} ===")
    print(f"Datasets ({len(dataset_labels)}):")
    for ds in dataset_labels:
        print(f" - {ds}")

    all_results = []

    # --------------------------------------------------
    # 1. Single-dataset models
    # --------------------------------------------------
    for train_label in dataset_labels:
        print(f"\n--- Training single-dataset model: {train_label} ---")

        train_subset = train_map[train_label]
        valid_subset_same = valid_map[train_label]

        train_dataset = build_dataset_from_samples(
            train_subset,
            species2id,
            task2id,
            dna_rna_tokenizer,
            protein_tokenizer,
            model_cfg,
            EpiLingoDataset,
        )
        valid_dataset_same = build_dataset_from_samples(
            valid_subset_same,
            species2id,
            task2id,
            dna_rna_tokenizer,
            protein_tokenizer,
            model_cfg,
            EpiLingoDataset,
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
        valid_loader_same = DataLoader(
            valid_dataset_same,
            batch_sampler=OmicsGroupedBatchSampler(
                valid_dataset_same,
                batch_size=train_cfg.eval_batch_size,
                shuffle=False,
            ),
            num_workers=train_cfg.num_workers,
        )

        model = EpiLingoHierarchical(model_cfg)
        trainer = EpiLingoTrainer(model, train_cfg, model_cfg)
        trainer.fit(train_loader, valid_loader_same)

        # test on all datasets in the same group
        for test_label in dataset_labels:
            test_subset = valid_map[test_label]

            metrics = evaluate_on_dataset(
                trainer=trainer,
                dataset_samples=test_subset,
                species2id=species2id,
                task2id=task2id,
                dna_rna_tokenizer=dna_rna_tokenizer,
                protein_tokenizer=protein_tokenizer,
                model_cfg=model_cfg,
                EpiLingoDataset=EpiLingoDataset,
                OmicsGroupedBatchSampler=OmicsGroupedBatchSampler,
                eval_batch_size=train_cfg.eval_batch_size,
                num_workers=train_cfg.num_workers,
            )

            row = {
                "group": group_name,
                "model_name": train_label,
                "model_type": "single_dataset_model",
                "test_dataset": test_label,
                "n_test_samples": len(test_subset),
                "AUROC": metrics.get("AUROC", float("nan")),
                "AUPRC": metrics.get("AUPRC", float("nan")),
                "ACC": metrics.get("ACC", float("nan")),
                "F1": metrics.get("F1", float("nan")),
                "MCC": metrics.get("MCC", float("nan")),
                "Precision": metrics.get("Precision", float("nan")),
                "Recall": metrics.get("Recall", float("nan")),
                "eval_loss": metrics.get("eval_loss", float("nan")),
            }
            all_results.append(row)

            print(
                f"Model={train_label} | Test={test_label} | "
                f"AUROC={row['AUROC']:.4f}, F1={row['F1']:.4f}, MCC={row['MCC']:.4f}"
            )

    # --------------------------------------------------
    # 2. Unified model
    # --------------------------------------------------
    unified_model_name = f"unified_{group_name}_model"
    print(f"\n--- Training unified model: {unified_model_name} ---")

    unified_train_samples = []
    unified_valid_samples = []
    for ds in dataset_labels:
        unified_train_samples.extend(train_map[ds])
        unified_valid_samples.extend(valid_map[ds])

    unified_train_dataset = build_dataset_from_samples(
        unified_train_samples,
        species2id,
        task2id,
        dna_rna_tokenizer,
        protein_tokenizer,
        model_cfg,
        EpiLingoDataset,
    )
    unified_valid_dataset = build_dataset_from_samples(
        unified_valid_samples,
        species2id,
        task2id,
        dna_rna_tokenizer,
        protein_tokenizer,
        model_cfg,
        EpiLingoDataset,
    )

    unified_train_loader = DataLoader(
        unified_train_dataset,
        batch_sampler=OmicsGroupedBatchSampler(
            unified_train_dataset,
            batch_size=train_cfg.train_batch_size,
            shuffle=True,
        ),
        num_workers=train_cfg.num_workers,
    )
    unified_valid_loader = DataLoader(
        unified_valid_dataset,
        batch_sampler=OmicsGroupedBatchSampler(
            unified_valid_dataset,
            batch_size=train_cfg.eval_batch_size,
            shuffle=False,
        ),
        num_workers=train_cfg.num_workers,
    )

    unified_model = EpiLingoHierarchical(model_cfg)
    unified_trainer = EpiLingoTrainer(unified_model, train_cfg, model_cfg)
    unified_trainer.fit(unified_train_loader, unified_valid_loader)

    # save unified model checkpoint
    unified_ckpt_path = os.path.join(output_dir, f"{group_name}_unified_model.pt")
    torch.save({
        "model_state_dict": unified_trainer.model.state_dict(),
        "species2id": species2id,
        "task2id": task2id,
        "model_cfg": dict(model_cfg.__dict__),
        "train_cfg": dict(train_cfg.__dict__),
        "group_name": group_name,
        "dataset_labels": dataset_labels,
    }, unified_ckpt_path)

    print(f"Saved unified model checkpoint: {unified_ckpt_path}")

    # collect unified model results separately
    unified_rows = []

    for test_label in dataset_labels:
        test_subset = valid_map[test_label]

        metrics = evaluate_on_dataset(
            trainer=unified_trainer,
            dataset_samples=test_subset,
            species2id=species2id,
            task2id=task2id,
            dna_rna_tokenizer=dna_rna_tokenizer,
            protein_tokenizer=protein_tokenizer,
            model_cfg=model_cfg,
            EpiLingoDataset=EpiLingoDataset,
            OmicsGroupedBatchSampler=OmicsGroupedBatchSampler,
            eval_batch_size=train_cfg.eval_batch_size,
            num_workers=train_cfg.num_workers,
        )

        row = {
            "group": group_name,
            "model_name": unified_model_name,
            "model_type": "unified_model",
            "test_dataset": test_label,
            "n_test_samples": len(test_subset),
            "AUROC": metrics.get("AUROC", float("nan")),
            "AUPRC": metrics.get("AUPRC", float("nan")),
            "ACC": metrics.get("ACC", float("nan")),
            "F1": metrics.get("F1", float("nan")),
            "MCC": metrics.get("MCC", float("nan")),
            "Precision": metrics.get("Precision", float("nan")),
            "Recall": metrics.get("Recall", float("nan")),
            "eval_loss": metrics.get("eval_loss", float("nan")),
        }

        all_results.append(row)
        unified_rows.append(row)

        print(
            f"Model={unified_model_name} | Test={test_label} | "
            f"AUROC={row['AUROC']:.4f}, F1={row['F1']:.4f}, MCC={row['MCC']:.4f}"
        )

    # save unified model validation results
    unified_metrics_df = pd.DataFrame(unified_rows)
    unified_metrics_path = os.path.join(output_dir, f"{group_name}_unified_model_metrics.csv")
    unified_metrics_df.to_csv(unified_metrics_path, index=False)
    print(f"Saved unified model metrics: {unified_metrics_path}")

    return pd.DataFrame(all_results), dataset_labels

def main():
    parser = argparse.ArgumentParser(description="Cross-species / cross-dataset evaluation with unified model")
    parser.add_argument("--framework_path", type=str, default="EpilingoHierarchicalFramework4.py")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./cross_species_with_unified_results")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fw = load_framework_module(args.framework_path)

    # Reuse classes/utilities from current framework
    ModelConfig = fw.ModelConfig
    TrainConfig = fw.TrainConfig
    set_seed = fw.set_seed
    load_all_datasets = fw.load_all_datasets
    build_vocab = fw.build_vocab
    EpiLingoDataset = fw.EpiLingoDataset
    OmicsGroupedBatchSampler = fw.OmicsGroupedBatchSampler
    EpiLingoHierarchical = fw.EpiLingoHierarchical
    EpiLingoTrainer = fw.EpiLingoTrainer
    AutoTokenizer = fw.AutoTokenizer

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # optional overrides
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

    print(f"Loaded train samples: {len(train_samples)}")
    print(f"Loaded valid samples: {len(valid_samples)}")

    all_samples = train_samples + valid_samples
    all_species = [s.species for s in all_samples]
    all_tasks = [s.task_name for s in all_samples]

    species2id = build_vocab(all_species)
    task2id = build_vocab(all_tasks)
    model_cfg.num_species = len(species2id)
    model_cfg.num_tasks = len(task2id)

    print(f"Species2id: {species2id}")
    print(f"Task2id: {task2id}")
    print(f"Device: {train_cfg.device}")

    dna_rna_tokenizer = AutoTokenizer.from_pretrained(model_cfg.dna_rna_backbone_name)
    protein_tokenizer = AutoTokenizer.from_pretrained(model_cfg.protein_backbone_name, do_lower_case=False)

    # --------------------------------------------------
    # DNA/RNA group
    # --------------------------------------------------

    dna_rna_df, dna_rna_labels = run_group_evaluation(
        group_name="dna_rna",
        allowed_omics=["DNA", "RNA"],
        train_samples=train_samples,
        valid_samples=valid_samples,
        species2id=species2id,
        task2id=task2id,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        EpiLingoDataset=EpiLingoDataset,
        OmicsGroupedBatchSampler=OmicsGroupedBatchSampler,
        EpiLingoHierarchical=EpiLingoHierarchical,
        EpiLingoTrainer=EpiLingoTrainer,
        output_dir=args.output_dir,
    )

    # --------------------------------------------------
    # Protein group
    # --------------------------------------------------
    protein_df, protein_labels = run_group_evaluation(
        group_name="protein",
        allowed_omics=["Protein"],
        train_samples=train_samples,
        valid_samples=valid_samples,
        species2id=species2id,
        task2id=task2id,
        dna_rna_tokenizer=dna_rna_tokenizer,
        protein_tokenizer=protein_tokenizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        EpiLingoDataset=EpiLingoDataset,
        OmicsGroupedBatchSampler=OmicsGroupedBatchSampler,
        EpiLingoHierarchical=EpiLingoHierarchical,
        EpiLingoTrainer=EpiLingoTrainer,
        output_dir=args.output_dir,
    )

    # Save long tables
    dna_rna_long_path = os.path.join(args.output_dir, "dna_rna_long_results.csv")
    protein_long_path = os.path.join(args.output_dir, "protein_long_results.csv")
    dna_rna_df.to_csv(dna_rna_long_path, index=False)
    protein_df.to_csv(protein_long_path, index=False)

    # Save matrices
    dna_rna_mats = save_metric_matrices(
        dna_rna_df,
        prefix="dna_rna",
        out_dir=args.output_dir,
        metrics=["AUROC", "F1", "MCC"],
    )
    protein_mats = save_metric_matrices(
        protein_df,
        prefix="protein",
        out_dir=args.output_dir,
        metrics=["AUROC", "F1", "MCC"],
    )

    # Save Excel
    # excel_path = os.path.join(args.output_dir, "cross_group_matrices.xlsx")
    # with pd.ExcelWriter(excel_path) as writer:
    #     dna_rna_df.to_excel(writer, sheet_name="dna_rna_long", index=False)
    #     protein_df.to_excel(writer, sheet_name="protein_long", index=False)

    #     for metric, mat in dna_rna_mats.items():
    #         mat.to_excel(writer, sheet_name=f"dna_rna_{metric}")

    #     for metric, mat in protein_mats.items():
    #         mat.to_excel(writer, sheet_name=f"protein_{metric}")

    ##########
    # Save CSV files
    dna_rna_df.to_csv(os.path.join(args.output_dir, "dna_rna_long.csv"),index=False)

    protein_df.to_csv(os.path.join(args.output_dir, "protein_long.csv"),index=False)

    for metric, mat in dna_rna_mats.items():
        mat.to_csv(
            os.path.join(args.output_dir, f"dna_rna_{metric}.csv"),
            index=True
        )

    for metric, mat in protein_mats.items():
        mat.to_csv(
            os.path.join(args.output_dir, f"protein_{metric}.csv"),
            index=True
        )
#########
    print("\nSaved files:")
    print(f" - {dna_rna_long_path}")
    print(f" - {protein_long_path}")
    print(f" - {os.path.join(args.output_dir, 'dna_rna_AUROC_matrix.csv')}")
    print(f" - {os.path.join(args.output_dir, 'dna_rna_F1_matrix.csv')}")
    print(f" - {os.path.join(args.output_dir, 'dna_rna_MCC_matrix.csv')}")
    print(f" - {os.path.join(args.output_dir, 'protein_AUROC_matrix.csv')}")
    print(f" - {os.path.join(args.output_dir, 'protein_F1_matrix.csv')}")
    print(f" - {os.path.join(args.output_dir, 'protein_MCC_matrix.csv')}")
    #print(f" - {excel_path}")


if __name__ == "__main__":
    main()