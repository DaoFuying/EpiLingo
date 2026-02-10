import os
import argparse
import random
import time
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd

from dna_lm_models import (
    DNAGPTForSequenceClassification,
    DNALLAMAForSequenceClassification,
    DNAT5ForSequenceClassification,
)


# ======================
#  Utilities
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DNADataset(Dataset):
    """
    Simple TSV dataset.
    Assumes at least two columns: sequence, label
    If column names are not 'sequence' / 'label', uses first two columns.
    """

    def __init__(self, tsv_path, tokenizer, max_length=41):
        df = pd.read_csv(tsv_path, sep="\t")

        # sequence column
        if "sequence" in df.columns:
            seq_col = "sequence"
        else:
            seq_col = df.columns[0]

        # label column
        if "label" in df.columns:
            label_col = "label"
        else:
            label_col = df.columns[1]

        self.sequences = df[seq_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


def build_model(
    arch: str,
    vocab_size: int,
    max_position_embeddings: int,
    model_name_or_path: str,
    num_labels: int = 2,
):
    """
    Build model according to architecture name.
    - dnabert: load pretrained DNABERT via AutoModelForSequenceClassification
    - gpt / llama / t5: use custom DNA models with config matched to DNABERT
    """

    arch = arch.lower()
    bert_config = AutoConfig.from_pretrained(model_name_or_path)

    # Extract reference dimensions from DNABERT
    d_model = getattr(bert_config, "hidden_size", 256)
    n_layer = getattr(bert_config, "num_hidden_layers", 6)
    n_head = getattr(bert_config, "num_attention_heads", 8)
    d_ff = getattr(bert_config, "intermediate_size", d_model * 4)

    if arch == "dnabert":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )
    elif arch == "gpt":
        model = DNAGPTForSequenceClassification(
            vocab_size=vocab_size,
            num_labels=num_labels,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            max_position_embeddings=max_position_embeddings,
            dropout=0.1,
        )
    elif arch == "llama":
        model = DNALLAMAForSequenceClassification(
            vocab_size=vocab_size,
            num_labels=num_labels,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            intermediate_size=d_ff,
            max_position_embeddings=max_position_embeddings,
            dropout=0.1,
        )
    elif arch == "t5":
        model = DNAT5ForSequenceClassification(
            vocab_size=vocab_size,
            num_labels=num_labels,
            d_model=d_model,
            d_ff=d_ff,
            n_layer=n_layer,
            n_head=n_head,
            max_position_embeddings=max_position_embeddings,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model '{arch}' has {num_params/1e6:.2f}M parameters")

    return model


def _extract_logits_and_loss(outputs, labels):
    """
    Handle both HF ModelOutput and dict-style custom outputs.
    Always returns (logits, loss).
    """
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        loss = outputs.get("loss", None)
        if loss is None:
            # compute loss here if not provided
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
    else:
        logits = outputs.logits
        loss = outputs.loss
    return logits, loss


def evaluate(model, dataloader, device, return_raw=False):
    """
    If return_raw=True, also return (labels, probs) arrays for ROC plotting.
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits, _ = _extract_logits_and_loss(outputs, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # prob of positive class

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    if return_raw:
        return auc, acc, f1, all_labels, all_probs
    else:
        return auc, acc, f1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) tokenizer (shared across all architectures)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    vocab_size = len(tokenizer)
    print("[INFO] Vocab size:", vocab_size)

    # 2) model
    model = build_model(
        arch=args.arch,
        vocab_size=vocab_size,
        max_position_embeddings=args.max_seq_length,
        model_name_or_path=args.model_name_or_path,
        num_labels=args.num_labels,
    ).to(device)
    print(f"[INFO] Using architecture: {args.arch}")

    # 3) datasets
    train_path = os.path.join(args.data_dir, "train.tsv")
    dev_path = os.path.join(args.data_dir, "dev.tsv")

    train_dataset = DNADataset(
        train_path, tokenizer, max_length=args.max_seq_length
    )
    dev_dataset = DNADataset(
        dev_path, tokenizer, max_length=args.max_seq_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4) optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate
    )

    t_total = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * t_total),
        num_training_steps=t_total,
    )

    global_step = 0
    best_dev_auc = -1.0
    best_state_dict = None
    os.makedirs(args.output_dir, exist_ok=True)

    # store per-epoch metrics for plotting AUC curves
    epoch_records = []

    print("[INFO] Start training")
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            _, loss = _extract_logits_and_loss(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                print(
                    f"[Epoch {epoch+1}/{args.num_train_epochs}] "
                    f"step {step+1}/{len(train_loader)}, "
                    f"global_step {global_step}, loss = {avg_loss:.4f}"
                )

        # Evaluate at end of epoch
        dev_auc, dev_acc, dev_f1 = evaluate(model, dev_loader, device)
        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch+1}] Dev AUC: {dev_auc:.4f}, "
            f"Dev ACC: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}, "
            f"time: {elapsed:.1f}s"
        )

        epoch_records.append(
            {
                "arch": args.arch,
                "epoch": epoch + 1,
                "global_step": global_step,
                "dev_auc": dev_auc,
                "dev_acc": dev_acc,
                "dev_f1": dev_f1,
            }
        )

        # Save best (in memory)
        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"[INFO] New best AUC = {best_dev_auc:.4f} at epoch {epoch+1}")

    print(f"[INFO] Training finished. Best Dev AUC = {best_dev_auc:.4f}")

    # Save per-epoch metrics (for AUC vs epoch plots)
    metrics_path = os.path.join(args.output_dir, "dev_metrics_per_epoch.csv")
    df_metrics = pd.DataFrame(epoch_records)
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[INFO] Per-epoch dev metrics saved to {metrics_path}")

    # Load best state before final evaluation for ROC data
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)

    # Final evaluation with raw outputs for ROC curve plotting
    best_auc, best_acc, best_f1, labels, probs = evaluate(
        model, dev_loader, device, return_raw=True
    )

    # Save ROC data
    roc_path = os.path.join(args.output_dir, "dev_roc_data.csv")
    df_roc = pd.DataFrame(
        {
            "label": labels,
            "prob": probs,
        }
    )
    df_roc.to_csv(roc_path, index=False)
    print(f"[INFO] ROC data (labels & probs) saved to {roc_path}")

    # Save final/best summary
    summary_path = os.path.join(args.output_dir, "dev_best_summary.csv")
    df_summary = pd.DataFrame(
        [
            {
                "arch": args.arch,
                "best_dev_auc": best_auc,
                "best_dev_acc": best_acc,
                "best_dev_f1": best_f1,
                "num_epochs": args.num_train_epochs,
            }
        ]
    )
    df_summary.to_csv(summary_path, index=False)
    print(f"[INFO] Best dev summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    # core settings
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["gpt", "llama", "t5", "dnabert"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "For dnabert: path to DNABERT checkpoint. "
            "For gpt/llama/t5: path containing tokenizer & vocab (DNABERT tokenizer)."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.tsv and dev.tsv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and metrics.",
    )

    # training hyperparameters
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=41)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
