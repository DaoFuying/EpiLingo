#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Training output directory, such as outputs/dnabert6_fullft")
    ap.add_argument("--input_tsv", required=True, help="The TSV to be predicted must contain at least a sequence column")
    ap.add_argument("--output_csv", required=True, help="prediction outputs csv file")
    ap.add_argument("--max_seq_length", type=int, default=41)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    if "sequence" not in df.columns:
        raise ValueError("input_tsv must include sequence column")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    probs_1 = []
    preds = []

    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            batch = df["sequence"].iloc[i:i+args.batch_size].tolist()
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=args.max_seq_length,
                return_tensors="pt",
            )
            enc = {k: v.to(args.device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            probs_1.extend(p[:, 1].tolist())
            preds.extend(np.argmax(p, axis=-1).tolist())

    out = pd.DataFrame({
        "prob_1": probs_1,
        "pred": preds,
    })


    out.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

if __name__ == "__main__":
    main()
