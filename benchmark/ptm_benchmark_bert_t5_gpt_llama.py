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


def normalize_protein_sequence(seq: str) -> str:
    """
    Protein LM safe normalization:
    - uppercase
    - remove spaces
    - keep letters only
    - map U/Z/O/B -> X
    """
    if not isinstance(seq, str):
        seq = str(seq)
    seq = seq.strip().upper()
    seq = "".join(seq.split())
    seq = "".join([c for c in seq if c.isalpha()])
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    return seq


def protein_to_spaced(seq: str) -> str:
    """
    ProtBert-style tokenization expects space-separated amino acids:
    "MTEYK" -> "M T E Y K"
    """
    seq = normalize_protein_sequence(seq)
    return " ".join(list(seq))


class ProteinTSVDataset(Dataset):
    """
    TSV dataset: columns [sequence, label] or named columns 'sequence'/'label'
    (与你 DNA 的 train.tsv/dev.tsv 格式一致)
    """

    def __init__(self, tsv_path, tokenizer, max_length=512):
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
        seq = protein_to_spaced(self.sequences[idx])
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
    dropout: float = 0.5,
):
    """
    Protein comparison under fair setting:
    - bert: load pretrained protein BERT (e.g., Rostlab/prot_bert_bfd)
    - gpt/llama/t5: use your custom models with matched dims from ref_config
    """
    arch = arch.lower()
    ref_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    d_model = getattr(ref_config, "hidden_size", 768)
    n_layer = getattr(ref_config, "num_hidden_layers", 12)
    n_head = getattr(ref_config, "num_attention_heads", 12)
    d_ff = getattr(ref_config, "intermediate_size", d_model * 4)

    if arch == "bert":
        # enforce dropout=0.5
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        # common dropout fields for BERT-like models
        if hasattr(cfg, "hidden_dropout_prob"):
            cfg.hidden_dropout_prob = dropout
        if hasattr(cfg, "attention_probs_dropout_prob"):
            cfg.attention_probs_dropout_prob = dropout
        if hasattr(cfg, "classifier_dropout"):
            cfg.classifier_dropout = dropout
        cfg.num_labels = num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=cfg,
            trust_remote_code=True,
        )

        # allow longer seq (e.g. 1024) if requested
        if hasattr(model.config, "max_position_embeddings") and max_position_embeddings > model.config.max_position_embeddings:
            try:
                model.resize_position_embeddings(max_position_embeddings)
                print(f"[INFO] Resized position embeddings to {max_position_embeddings}")
            except Exception as e:
                print(f"[WARN] Failed to resize position embeddings: {e}. Will rely on truncation.")

    elif arch == "gpt":
        model = DNAGPTForSequenceClassification(
            vocab_size=vocab_size,
            num_labels=num_labels,
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
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
            dropout=dropout,
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
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model '{arch}' has {num_params/1e6:.2f}M parameters")
    return model


def _extract_logits_and_loss(outputs, labels):
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
    all_labels, all_probs = [], []

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


def set_requires_grad_for_backbone(model, arch: str, requires_grad: bool):
    """
    Freeze/unfreeze pretrained backbone.
    For HF BERT-style models, backbone typically under model.base_model or model.bert.
    """
    if arch.lower() != "bert":
        return False

    backbone = None
    if hasattr(model, "base_model"):
        backbone = model.base_model
    elif hasattr(model, "bert"):
        backbone = model.bert

    if backbone is None:
        print("[WARN] Cannot locate BERT backbone to freeze/unfreeze.")
        return False

    for p in backbone.parameters():
        p.requires_grad = requires_grad

    # keep classifier trainable
    for n, p in model.named_parameters():
        if "classifier" in n or "score" in n:
            p.requires_grad = True

    return True


def make_dataloaders(args, tokenizer, seq_len, batch_size):
    train_path = os.path.join(args.data_dir, "train.tsv")
    dev_path = os.path.join(args.data_dir, "dev.tsv")

    train_dataset = ProteinTSVDataset(train_path, tokenizer, max_length=seq_len)
    dev_dataset = ProteinTSVDataset(dev_path, tokenizer, max_length=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, dev_loader


def train_one_stage(
    model,
    arch: str,
    device,
    tokenizer,
    args,
    seq_len: int,
    batch_size: int,
    max_epochs: int,
    lr: float,
    begin_frozen: bool,
    stage_name: str,
):
    """
    Implements:
    - ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-5)
    monitored metric: dev AUC

    NOTE:
    - EarlyStopping is DISABLED.
    - We still keep "best weights on dev AUC" and restore them at the end of stage.
    """

    # dataloaders for this seq_len
    train_loader, dev_loader = make_dataloaders(args, tokenizer, seq_len, batch_size)

    # freeze if requested
    froze_ok = False
    if begin_frozen:
        froze_ok = set_requires_grad_for_backbone(model, arch, requires_grad=False)
        if not froze_ok:
            print(
                f"[WARN] begin_with_frozen_pretrained_layers requested but arch={arch} has no obvious pretrained backbone. Skip freezing."
            )

    # optimizer over trainable params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)

    # ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.plateau_patience,
        factor=args.plateau_factor,
        min_lr=args.plateau_min_lr,
        # verbose=True,  # keep commented if your torch doesn't support it
    )

    best_auc = -1.0
    best_state = None

    records = []
    global_step = 0

    print(
        f"\n[INFO] ===== Stage: {stage_name} | seq_len={seq_len} | batch={batch_size} | lr={lr} | frozen={begin_frozen} ====="
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            _, loss = _extract_logits_and_loss(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.logging_steps == 0:
                print(
                    f"[{stage_name}] epoch {epoch}/{max_epochs}, step {step}/{len(train_loader)}, loss={epoch_loss/step:.4f}"
                )

        # end-of-epoch eval
        dev_auc, dev_acc, dev_f1 = evaluate(model, dev_loader, device)
        scheduler.step(dev_auc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{stage_name}] Epoch {epoch}: Dev AUC={dev_auc:.4f} ACC={dev_acc:.4f} F1={dev_f1:.4f} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        records.append(
            {
                "stage": stage_name,
                "epoch": epoch,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "lr": lr_now,
                "dev_auc": dev_auc,
                "dev_acc": dev_acc,
                "dev_f1": dev_f1,
            }
        )

        # keep best weights (but DO NOT early stop)
        if dev_auc > best_auc:
            best_auc = dev_auc
            best_state = copy.deepcopy(model.state_dict())

    # restore best weights of this stage (recommended for stability)
    if best_state is not None:
        model.load_state_dict(best_state)

    # unfreeze after frozen stage ends (so later stage can train full model)
    if begin_frozen and froze_ok:
        set_requires_grad_for_backbone(model, arch, requires_grad=True)

    return model, best_auc, records


def evaluate_by_len(model, arch, device, tokenizer, args, start_seq_len=512, start_batch_size=32):
    """
    Minimal evaluate_by_len compatible with your call:
    results, confusion_matrix, a, b = evaluate_by_len(...)
    Here:
      - results: dict of metrics by seq_len
      - confusion_matrix: None (you can add if needed)
      - a,b: returned labels/probs for ROC plotting at start_seq_len
    """
    _, dev_loader = make_dataloaders(args, tokenizer, start_seq_len, start_batch_size)
    auc, acc, f1, labels, probs = evaluate(model, dev_loader, device, return_raw=True)
    results = {start_seq_len: {"auc": auc, "acc": acc, "f1": f1}}
    confusion_matrix = None
    return results, confusion_matrix, labels, probs


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print("[INFO] Vocab size:", vocab_size)

    # build model with capacity supporting final_seq_len
    max_pos = max(args.seq_len, args.final_seq_len)
    model = build_model(
        arch=args.arch,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        model_name_or_path=args.model_name_or_path,
        num_labels=args.num_labels,
        dropout=args.dropout,
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: frozen pretrained layers (if bert), high lr
    model, best1, rec1 = train_one_stage(
        model=model,
        arch=args.arch,
        device=device,
        tokenizer=tokenizer,
        args=args,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs_per_stage,
        lr=args.lr_with_frozen_pretrained_layers,
        begin_frozen=args.begin_with_frozen_pretrained_layers,
        stage_name="frozen_stage",
    )

    # Stage 2: unfrozen full finetune, lr=1e-4
    model, best2, rec2 = train_one_stage(
        model=model,
        arch=args.arch,
        device=device,
        tokenizer=tokenizer,
        args=args,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs_per_stage,
        lr=args.lr,
        begin_frozen=False,
        stage_name="finetune_stage",
    )

    # Final stage: 1 epoch with longer seq_len (e.g., 1024) and final_lr
    model, best3, rec3 = train_one_stage(
        model=model,
        arch=args.arch,
        device=device,
        tokenizer=tokenizer,
        args=args,
        seq_len=args.final_seq_len,
        batch_size=args.batch_size,
        max_epochs=args.n_final_epochs,
        lr=args.final_lr,
        begin_frozen=False,
        stage_name="final_stage",
    )

    # save metrics
    all_records = rec1 + rec2 + rec3
    metrics_path = os.path.join(args.output_dir, "dev_metrics_per_epoch.csv")
    pd.DataFrame(all_records).to_csv(metrics_path, index=False)
    print(f"[INFO] Saved per-epoch dev metrics to {metrics_path}")

    # final eval (dev) by len (start_seq_len=512, start_batch_size=32)
    results, confusion_matrix, labels, probs = evaluate_by_len(
        model, args.arch, device, tokenizer, args,
        start_seq_len=args.seq_len,
        start_batch_size=args.batch_size,
    )

    roc_path = os.path.join(args.output_dir, "dev_roc_data.csv")
    pd.DataFrame({"label": labels, "prob": probs}).to_csv(roc_path, index=False)
    print(f"[INFO] Saved ROC data to {roc_path}")

    summary_path = os.path.join(args.output_dir, "dev_best_summary.csv")
    # best summary uses best over stages (roughly)
    best_auc = max(best1, best2, best3)
    pd.DataFrame([{
        "arch": args.arch,
        "model_name_or_path": args.model_name_or_path,
        "best_dev_auc": best_auc,
        "seq_len": args.seq_len,
        "final_seq_len": args.final_seq_len,
        "dropout": args.dropout,
        "lr": args.lr,
        "lr_with_frozen_pretrained_layers": args.lr_with_frozen_pretrained_layers,
        "final_lr": args.final_lr,
        "max_epochs_per_stage": args.max_epochs_per_stage,
        "n_final_epochs": args.n_final_epochs,
    }]).to_csv(summary_path, index=False)
    print(f"[INFO] Saved best summary to {summary_path}")

    print("[INFO] Final results:", results)
    return results, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, required=True, choices=["bert", "gpt", "llama", "t5"])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="contains train.tsv and dev.tsv")
    parser.add_argument("--output_dir", type=str, required=True)

    # match your required settings
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--seq_len", type=int, default=64)          # seq_len = 512
    parser.add_argument("--final_seq_len", type=int, default=64)   # final_seq_len = 1024

    parser.add_argument("--batch_size", type=int, default=32)        # batch_size = 32

    parser.add_argument("--max_epochs_per_stage", type=int, default=40)  # max_epochs_per_stage = 40
    parser.add_argument("--lr", type=float, default=1e-4)                # lr = 1e-4
    parser.add_argument("--begin_with_frozen_pretrained_layers", action="store_true", default=True)
    parser.add_argument("--lr_with_frozen_pretrained_layers", type=float, default=1e-2)  # lr_with_frozen_pretrained_layers = 1e-2
    parser.add_argument("--n_final_epochs", type=int, default=1)         # n_final_epochs = 1
    parser.add_argument("--final_lr", type=float, default=1e-5)          # final_lr = 1e-5

    # callbacks equivalent
    parser.add_argument("--plateau_patience", type=int, default=1)
    parser.add_argument("--plateau_factor", type=float, default=0.25)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-5)

    #parser.add_argument("--early_stop_patience", type=int, default=2)

    # misc
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
