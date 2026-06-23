"""Data loading and preparation for the Turkish ABSA project.

Source : maydogan/Turkish_SentimentAnalysis_TRSAv1 (HF Hub)
Schema : id (int64) | score (str: Positive/Neutral/Negative) | review (str, 3-1990 chars)
Splits : single `train` split, 150k rows (balanced 50k / 50k / 50k)

NOTE: `score` is a DOCUMENT-level label and is demonstrably noisy
(e.g. "Tam zamanında teslimat teşekkürler" is labeled Negative). We keep it
only as a weak baseline signal under the name `doc_sentiment`. Aspect-level
labels are produced separately in label.py.
"""

import re
import unicodedata
from pathlib import Path

import polars as pl
from datasets import load_dataset

# Source uses capitalized English labels; normalize to lowercase.
_DOC_LABEL_MAP = {"Positive": "positive", "Neutral": "neutral", "Negative": "negative"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")


def load_raw(cfg: dict) -> pl.DataFrame:
    """Load TRSAv1 from the HF Hub and return a normalized polars frame.

    Returns columns:
        id            : original row id
        text          : raw review text (untouched; cleaning happens later)
        doc_sentiment : normalized document-level label (weak signal only)
    """
    ds = load_dataset(cfg["data"]["hf_id"], split="train")
    try:
        df = ds.to_polars()  # datasets with native polars support
    except AttributeError:
        df = pl.from_arrow(ds.data.table)  # fallback: zero-copy arrow -> polars

    df = df.rename({"review": "text", "score": "doc_sentiment"})
    df = df.with_columns(pl.col("doc_sentiment").replace(_DOC_LABEL_MAP))
    return df.select(["id", "text", "doc_sentiment"])


def clean_text(s: str) -> str:
    """Light, ABSA-safe normalization.

    We deliberately do NOT lowercase: the target model (BERTurk *cased*) relies
    on casing, and Turkish lowercasing is a known trap ("İSTANBUL".lower() in a
    non-Turkish locale breaks the dotted/dotless i). We also preserve negation
    words, punctuation, and emojis, all of which carry aspect-level sentiment
    ("fena değil", "👍"). Only URLs and redundant whitespace are removed.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _URL.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def filter_for_absa(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Drop reviews too short to plausibly contain an aspect.

    Single-word or emoji-only reviews ("Kötü", "👍👍") expose at most one aspect
    and mostly add noise to ABSA, so we require >= min_chars characters on the
    cleaned text.
    """
    return df.filter(pl.col("text").str.len_chars() >= cfg["data"]["min_chars"])


def make_splits(df: pl.DataFrame, cfg: dict) -> dict[str, pl.DataFrame]:
    """Stratified train/val/test split, seeded for reproducibility.

    Stratifies on doc_sentiment so the weak baseline stays balanced across
    splits. Aspect labels are attached later, per split, in label.py.
    """
    seed = cfg["data"]["seed"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    parts: dict[str, list[pl.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, group in df.group_by("doc_sentiment", maintain_order=True):
        g = group.sample(fraction=1.0, shuffle=True, seed=seed)
        n = g.height
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        parts["test"].append(g.slice(0, n_test))
        parts["val"].append(g.slice(n_test, n_val))
        parts["train"].append(g.slice(n_test + n_val))

    return {
        name: pl.concat(frames).sample(fraction=1.0, shuffle=True, seed=seed)
        for name, frames in parts.items()
    }


def prepare(cfg: dict) -> dict[str, pl.DataFrame]:
    """End-to-end: load -> clean -> filter -> split -> persist as parquet."""
    df = load_raw(cfg)
    df = df.with_columns(pl.col("text").map_elements(clean_text, return_dtype=pl.String))
    df = filter_for_absa(df, cfg)
    splits = make_splits(df, cfg)

    out = Path(cfg["data"]["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.write_parquet(out / f"{name}.parquet")
        print(f"{name:5s}: {frame.height:>6d} rows")
    return splits


if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        prepare(yaml.safe_load(f))