"""Fine-tune BERTurk for Turkish ABSA as sentence-pair classification.

Formulation
-----------
Each review is expanded into (review, aspect) pairs over the CLOSED taxonomy.
The model predicts one of 4 labels per pair:

    yok | negatif | notr | pozitif

"yok" means the aspect is not discussed (closed-world negative: any aspect the
weak labeler did not emit for a review is treated as absent). This single head
does aspect detection AND sentiment, and crucially POOLS sentiment signal across
aspects -- which is what rescues the rare aspects and the scarce `notr` class.

A fixed, seeded slice of reviews is held out as a GOLD candidate set and written
to disk for manual verification; it is excluded from training so evaluate.py can
report an honest number against human-checked labels.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

LABELS = ["yok", "negatif", "notr", "pozitif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
PRESENT_IDS = [LABEL2ID[l] for l in ("negatif", "notr", "pozitif")]

# Natural-language probe phrase per aspect, fed as the 2nd segment of the pair.
ASPECT_PROBE = {
    "urun_kalitesi": "ürün kalitesi ve etkisi",
    "fiyat": "fiyat",
    "kargo_teslimat": "kargo ve teslimat",
    "ambalaj": "paketleme ve ambalaj",
    "koku": "koku",
    "kalicilik": "kalıcılık ve dayanıklılık",
    "musteri_hizmetleri": "müşteri hizmetleri",
}


def load_labeled(path: str) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [r for r in rows if not r["error"]]


def split_reviews(rows: list[dict], cfg: dict) -> tuple[list, list, list]:
    """Pin the gold holdout to the verified ids in data/gold.jsonl and exclude
    them from train/val, so the eval set is stable and leak-free across reruns.
    Falls back to a size-independent split only on the very first run."""
    rng = np.random.default_rng(cfg["data"]["seed"])
    gold_file = Path("data/gold.jsonl")
    if gold_file.exists():
        gold_ids = {json.loads(l)["id"] for l in open(gold_file, encoding="utf-8")}
        pool = [r for r in rows if r["id"] not in gold_ids]
        gold = [r for r in rows if r["id"] in gold_ids]
        idx = rng.permutation(len(pool))
        val_n = int(len(pool) * cfg["train"]["val_frac"])
        val = [pool[i] for i in idx[:val_n]]
        train = [pool[i] for i in idx[val_n:]]
        return train, val, gold
    # first run only: no verified gold yet
    idx = rng.permutation(len(rows))
    g = cfg["train"]["gold_n"]
    v = int((len(rows) - g) * cfg["train"]["val_frac"])
    return ([rows[i] for i in idx[g + v:]],
            [rows[i] for i in idx[g:g + v]],
            [rows[i] for i in idx[:g]])


def build_pairs(rows: list[dict], cfg: dict) -> list[dict]:
    """Expand each review into one (review, aspect) pair per taxonomy aspect."""
    aspects = list(cfg["aspects"].keys())
    out = []
    for r in rows:
        present = {a["aspect"]: a["sentiment"] for a in r["aspects"]}
        for asp in aspects:
            label = present.get(asp, "yok")
            out.append(
                {
                    "text": r["text"],
                    "probe": ASPECT_PROBE[asp],
                    "label": LABEL2ID[label],
                }
            )
    return out


def make_dataset(pairs: list[dict], tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_list(pairs)

    def tok(batch):
        enc = tokenizer(
            batch["text"], batch["probe"],
            truncation="only_first", max_length=max_length, padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    return ds.map(tok, batched=True, remove_columns=ds.column_names)


def class_weights(pairs: list[dict]) -> torch.Tensor:
    counts = np.bincount([p["label"] for p in pairs], minlength=len(LABELS))
    w = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "present_f1": f1_score(labels, preds, labels=PRESENT_IDS, average="macro"),
        "detect_f1": f1_score(labels != 0, preds != 0),  # yok vs present
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=self.weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def main(cfg: dict) -> None:
    tc = cfg["train"]
    rows = load_labeled(cfg["labeling"]["out_path"])
    train_rows, val_rows, gold_rows = split_reviews(rows, cfg)

    gold_path = Path(cfg["data"]["splits_dir"]).parent / "gold_candidates.jsonl"
    if not Path("data/gold.jsonl").exists():
        with open(gold_path, "w", encoding="utf-8") as f:
            for r in gold_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"gold candidates -> {gold_path} (verify -> data/gold.jsonl)")
    else:
        print(f"gold pinned to data/gold.jsonl ({len(gold_rows)} reviews, excluded from train)")

    train_pairs = build_pairs(train_rows, cfg)
    val_pairs = build_pairs(val_rows, cfg)
    print(f"pairs   -> train {len(train_pairs)} | val {len(val_pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"], num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=LABEL2ID,
    )

    train_ds = make_dataset(train_pairs, tokenizer, tc["max_length"])
    val_ds = make_dataset(val_pairs, tokenizer, tc["max_length"])

    args = TrainingArguments(
        output_dir=tc["out_dir"],
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"] * 2,
        gradient_accumulation_steps=tc["grad_accum"],
        learning_rate=float(tc["lr"]),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        weights=class_weights(train_pairs),
    )
    trainer.train()

    trainer.save_model(tc["out_dir"])
    tokenizer.save_pretrained(tc["out_dir"])
    print(f"\nval metrics: {trainer.evaluate()}")
    print(f"model saved -> {tc['out_dir']}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        main(yaml.safe_load(f))