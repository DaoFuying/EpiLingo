#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A unified fine-tuning framework: 
Comparing different pre-trained models under identical data, training procedures, and evaluation conditions：
- DNABERT-6
- Nucleotide Transformer
- PlantRNA-FM

data format:
data/
  train.tsv   # header: sequence<TAB>label
  dev.tsv     # header: sequence<TAB>label

示例运行：
export DATA_PATH=data
export OUTPUT_PATH=outputs/dnabert6

python benchmark_plant_llm_comparison.py \
  --backbone dnabert6 \
  --data_dir $DATA_PATH \
  --output_dir $OUTPUT_PATH \
  --max_seq_length 41 \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --logging_steps 100
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from transformers import BertForSequenceClassification

# -----------------------
# 1. 读 tsv
# -----------------------
def load_tsv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} 必须包含 'sequence' 和 'label' 两列")
    df["label"] = df["label"].astype(int)
    return df



class SeqClsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        # 必须是一个“有固定长度的列表/数组”
        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Trainer 就是靠这个来判断有多少个 step
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        enc["labels"] = label
        return {k: torch.tensor(v) for k, v in enc.items()}


# -----------------------
# 2. 统一评估指标：AUC / ACC / F1
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.array(labels)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    # 二分类 AUC
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")

    return {"accuracy": acc, "f1": f1, "auc": auc}


# -----------------------
# 3. backbone 注册表
# -----------------------
BACKBONES = {
    # ① DNABERT-6：你现在的 baseline（6-mer 预训练）
    "dnabert6": "zhihan1996/DNA_bert_6",
    "dnabert5": "zhihan1996/DNA_bert_5",
    "dnabert4": "zhihan1996/DNA_bert_4",
    "dnabert3": "zhihan1996/DNA_bert_3",
    "plant_nt_6mer": "zhangtaolab/plant-nucleotide-transformer-6mer",

    # ② Nucleotide Transformer v2 – 100M multi-species
    "nt_100m": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",

    # ③ Nucleotide Transformer v2 – 500M human
    #"nt_500m": "InstaDeepAI/nucleotide-transformer-v2-500m-human",
    "plantrna_fm": "yangheng/PlantRNA-FM",
    "agront": "InstaDeepAI/agro-nucleotide-transformer-1b",
    "plantcaduceus_l32": "kuleshov-group/PlantCaduceus_l32",
}


# -----------------------
# 4. 主函数
# -----------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, required=True,
                        help=f"选择预训练模型: {list(BACKBONES.keys())}")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=41)

    parser.add_argument("--per_gpu_train_batch_size", type=int, default=64)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.backbone not in BACKBONES:
        raise ValueError(f"未知 backbone: {args.backbone}，可选: {list(BACKBONES.keys())}")
    model_name_or_path = BACKBONES[args.backbone]
    print(f"使用 backbone = {args.backbone} ({model_name_or_path})")

    # 1) load tokenizer & model
    # DNABERT-6 自带 6-mer tokenizer; NT 有自己的 bp tokenizer

    if args.backbone in ["dnabert6", "dnabert5", "dnabert4","dnabert3"]:
        # DNABERT-6 使用自己的 BertForSequenceClassification 实现
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            trust_remote_code=True,
        )
    elif args.backbone == "plantcaduceus_l32":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,num_labels=2,trust_remote_code=True,use_safetensors=True)

    else:
        # 其他 backbone（NT, PDLLM, megaDNA 等）用 AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            trust_remote_code=True,
        )


    # 2) load data
    train_df = load_tsv(os.path.join(args.data_dir, "train.tsv"))
    dev_df = load_tsv(os.path.join(args.data_dir, "dev.tsv"))

    train_dataset = SeqClsDataset(train_df, tokenizer, args.max_seq_length)
    dev_dataset   = SeqClsDataset(dev_df, tokenizer, args.max_seq_length)


    # 3) TrainingArguments：和 DNABERT 框架一样，每 epoch eval
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 4) 训练 + 记录时间
    print("***** Start training *****")
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    print(f"Training done. Time = {train_time/60:.2f} min")

    metrics = train_result.metrics
    metrics["train_runtime_sec"] = train_time
    trainer.save_model(args.output_dir)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 5) 最终在 dev 上评估一次，保存 eval_results.json
    print("***** Evaluate on dev *****")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # print("Done.")
    # ---- Save raw predictions for dev.tsv ----
    print("Saving prediction file for external plotting ...")

    dev_predictions = trainer.predict(dev_dataset)
    logits = dev_predictions.predictions
    labels = dev_predictions.label_ids

    # Convert logits → probabilities
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    # Predicted class
    preds = probs.argmax(axis=-1)

    # Save into CSV
    out_df = pd.DataFrame({
        #"sequence": dev_df["sequence"],
        "label": labels,
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "pred": preds,
    })

    save_path = os.path.join(args.output_dir, "dev_predictions.csv")
    out_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
