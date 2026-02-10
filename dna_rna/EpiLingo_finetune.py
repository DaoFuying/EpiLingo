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

# LoRA (PEFT)
from peft import LoraConfig, get_peft_model


# ======================
#  Utilities
# ======================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return total, trainable, pct


class DNADataset(Dataset):
    """
    Simple TSV dataset.
    Assumes at least two columns: sequence, label
    If column names are not 'sequence' / 'label', uses first two columns.
    """

    def __init__(self, tsv_path, tokenizer, max_length=41):
        df = pd.read_csv(tsv_path, sep="\t")

        seq_col = "sequence" if "sequence" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else df.columns[1]

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


def _extract_logits_and_loss(outputs, labels):
    """
    Handle HF ModelOutput and dict-style outputs.
    Always returns (logits, loss).
    """
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        loss = outputs.get("loss", None)
        if loss is None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
    else:
        logits = outputs.logits
        loss = outputs.loss
    return logits, loss


def evaluate(model, dataloader, device, return_raw=False):
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
            probs = torch.softmax(logits, dim=-1)[:, 1]

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
    return auc, acc, f1


def infer_lora_target_modules(model):
    """
    Try to infer reasonable LoRA target module names for BERT-like attention projections.
    We prioritize Q/V projections. Different backbones may name them differently.

    Returns: list[str]
    """
    # Common candidates across HF implementations
    candidates_priority = [
        # BERT / RoBERTa (often)
        "query", "value",
        # some models use q_proj/v_proj
        "q_proj", "v_proj",
        # some use Wq/Wv
        "Wq", "Wv",
    ]

    existing = set()
    for name, module in model.named_modules():
        # we only want leaf Linear modules typically
        if isinstance(module, torch.nn.Linear):
            last = name.split(".")[-1]
            if last in candidates_priority:
                existing.add(last)

    # Prefer Q/V pairs if present
    if "query" in existing and "value" in existing:
        return ["query", "value"]
    if "q_proj" in existing and "v_proj" in existing:
        return ["q_proj", "v_proj"]
    if "Wq" in existing and "Wv" in existing:
        return ["Wq", "Wv"]

    # Fallback: if only one found, use it; otherwise raise for visibility
    if len(existing) > 0:
        return sorted(list(existing))

    # If none found, user may need to inspect names
    raise RuntimeError(
        "Cannot infer LoRA target_modules automatically. "
        "Please print some model.named_modules() keys and set --lora_target_modules manually."
    )

def freeze_backbone_for_linear_probe(model):
    """
    Freeze all backbone parameters, keep classifier trainable.
    Works for AutoModelForSequenceClassification (BERT-like).
    """
    for name, param in model.named_parameters():
        # classifier 通常叫 classifier 或 score
        if "classifier" in name or "score" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def build_dnabert_model(args):
    """
    Build DNABERT model and optionally wrap with LoRA.
    """
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     args.model_name_or_path,
    #     num_labels=args.num_labels,
    #     trust_remote_code=True,
    # )

    cfg = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=False
    )
    

    cfg.num_labels = args.num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=cfg,
        trust_remote_code=False
    )

    if args.linear_probe:
        freeze_backbone_for_linear_probe(model)
        print("[INFO] Linear probe enabled: backbone frozen, classifier trainable only")

    if args.use_lora:
        if args.lora_target_modules is None or len(args.lora_target_modules) == 0:
            target_modules = infer_lora_target_modules(model)
        else:
            target_modules = args.lora_target_modules

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        # (可选) 打印可训练参数概览
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        print(f"[INFO] LoRA enabled. target_modules = {target_modules}")

    total, trainable, pct = count_params(model)
    print(f"[INFO] Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M ({pct:.2f}%)")

    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=False
    )
    print("[INFO] Vocab size:", len(tokenizer))

    model = build_dnabert_model(args).to(device)

    train_path = os.path.join(args.data_dir, "train.tsv")
    dev_path = os.path.join(args.data_dir, "dev.tsv")

    train_dataset = DNADataset(train_path, tokenizer, max_length=args.max_seq_length)
    dev_dataset = DNADataset(dev_path, tokenizer, max_length=args.max_seq_length)

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

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

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

        dev_auc, dev_acc, dev_f1 = evaluate(model, dev_loader, device)
        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch+1}] Dev AUC: {dev_auc:.4f}, "
            f"Dev ACC: {dev_acc:.4f}, Dev F1: {dev_f1:.4f}, "
            f"time: {elapsed:.1f}s"
        )

        epoch_records.append(
            {
                "arch": "dnabert_lora" if args.use_lora else "dnabert_full",
                "epoch": epoch + 1,
                "global_step": global_step,
                "dev_auc": dev_auc,
                "dev_acc": dev_acc,
                "dev_f1": dev_f1,
            }
        )

        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"[INFO] New best AUC = {best_dev_auc:.4f} at epoch {epoch+1}")

    print(f"[INFO] Training finished. Best Dev AUC = {best_dev_auc:.4f}")

    metrics_path = os.path.join(args.output_dir, "dev_metrics_per_epoch.csv")
    pd.DataFrame(epoch_records).to_csv(metrics_path, index=False)
    print(f"[INFO] Per-epoch dev metrics saved to {metrics_path}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)

    best_auc, best_acc, best_f1, labels, probs = evaluate(
        model, dev_loader, device, return_raw=True
    )

    roc_path = os.path.join(args.output_dir, "dev_roc_data.csv")
    pd.DataFrame({"label": labels, "prob": probs}).to_csv(roc_path, index=False)
    print(f"[INFO] ROC data saved to {roc_path}")

    total, trainable, pct = count_params(model)
    summary_path = os.path.join(args.output_dir, "dev_best_summary.csv")
    pd.DataFrame(
        [{
            "model": "dnabert+LoRA" if args.use_lora else "dnabert(full-ft)",
            "best_dev_auc": best_auc,
            "best_dev_acc": best_acc,
            "best_dev_f1": best_f1,
            "num_epochs": args.num_train_epochs,
            "total_params_M": total / 1e6,
            "trainable_params_M": trainable / 1e6,
            "trainable_pct": pct,
            "lora_r": args.lora_r if args.use_lora else "",
            "lora_alpha": args.lora_alpha if args.use_lora else "",
            "lora_dropout": args.lora_dropout if args.use_lora else "",
            "lora_target_modules": ",".join(args.lora_target_modules) if (args.use_lora and args.lora_target_modules) else "auto",
        }]
    ).to_csv(summary_path, index=False)
    print(f"[INFO] Best dev summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

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

    # LoRA switches & hyperparams
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated module names for LoRA, e.g. query,value or q_proj,v_proj. Default: auto infer.",
    )
    parser.add_argument(
        "--linear_probe",
        action="store_true",
        help="Freeze backbone and train classifier only (linear probing)."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
