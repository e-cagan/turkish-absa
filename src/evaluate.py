"""Evaluate the fine-tuned ABSA model against the human-verified gold set.

Reports the diagnostics that the aggregate val number hides:
  * per-class precision/recall/F1  (is `notr` the weak spot?)
  * per-aspect detect/sentiment F1 (which rare aspects underperform -> need data?)
  * a 4x4 confusion matrix         (what gets confused with what)

Falls back to the WEAK gold_candidates if data/gold.jsonl is absent, with a loud
warning -- only the human-verified file gives an honest number.
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import ASPECT_PROBE, LABEL2ID, LABELS, PRESENT_IDS, load_labeled


def build_eval_pairs(rows: list[dict], cfg: dict):
    """Expand reviews into (review, aspect) pairs, keeping the aspect identity."""
    aspects = list(cfg["aspects"].keys())
    texts, probes, labels, asp_of = [], [], [], []
    for r in rows:
        present = {a["aspect"]: a["sentiment"] for a in r["aspects"]}
        for asp in aspects:
            texts.append(r["text"])
            probes.append(ASPECT_PROBE[asp])
            labels.append(LABEL2ID[present.get(asp, "yok")])
            asp_of.append(asp)
    return texts, probes, np.array(labels), np.array(asp_of)


@torch.no_grad()
def predict(texts, probes, tokenizer, model, max_length, device, bs=32):
    preds = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(
            texts[i : i + bs], probes[i : i + bs],
            truncation="only_first", max_length=max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main(cfg: dict) -> None:
    out_dir = cfg["train"]["out_dir"]
    gold = Path("data/gold.jsonl")
    if gold.exists():
        rows, src = load_labeled(str(gold)), "gold.jsonl (human-verified)"
    else:
        rows = load_labeled("data/gold_candidates.jsonl")
        src = "gold_candidates.jsonl (WEAK -- verify -> data/gold.jsonl for an honest number)"
    print(f"eval set: {len(rows)} reviews from {src}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device).eval()

    texts, probes, y_true, asp_of = build_eval_pairs(rows, cfg)
    y_pred = predict(texts, probes, tokenizer, model, cfg["train"]["max_length"], device)

    print("=== PER-CLASS ===")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3, zero_division=0))

    print("=== HEADLINE ===")
    print(f"  macro_f1   : {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"  present_f1 : {f1_score(y_true, y_pred, labels=PRESENT_IDS, average='macro', zero_division=0):.3f}")
    print(f"  detect_f1  : {f1_score(y_true != 0, y_pred != 0, zero_division=0):.3f}")

    print("\n=== PER-ASPECT ===")
    print(f"  {'aspect':22s} {'support':>7s} {'detect_f1':>10s} {'present_f1':>11s}")
    for asp in cfg["aspects"]:
        m = asp_of == asp
        yt, yp = y_true[m], y_pred[m]
        support = int((yt != 0).sum())
        d = f1_score(yt != 0, yp != 0, zero_division=0)
        p = f1_score(yt, yp, labels=PRESENT_IDS, average="macro", zero_division=0)
        print(f"  {asp:22s} {support:7d} {d:10.3f} {p:11.3f}")

    print("\n=== CONFUSION (rows=true, cols=pred) ===")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    print("         " + " ".join(f"{l:>8s}" for l in LABELS))
    for i, l in enumerate(LABELS):
        print(f"  {l:>6s} " + " ".join(f"{cm[i][j]:8d}" for j in range(len(LABELS))))


if __name__ == "__main__":
    with open("config.yaml") as f:
        main(yaml.safe_load(f))