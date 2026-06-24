"""LLM weak-labeling for Turkish ABSA.

Reads the train split produced by data.py, samples `sample_n` reviews, and asks
an LLM to extract, for each review, the aspects from a CLOSED taxonomy that are
explicitly discussed, each with a sentiment label and a short evidence span.

Design goals:
  * Provider-agnostic  : Anthropic API or a local Ollama model (config-driven).
  * Robust parsing     : tolerates markdown fences / trailing prose / bad labels.
  * Resumable          : appends to JSONL keyed by id; reruns skip finished ids.
  * Concurrent         : API calls are I/O bound, so we fan out with a thread pool.

Output: one JSON object per line in `out_path`:
    {"id", "text", "doc_sentiment", "aspects": [{"aspect","sentiment","span"}], "error"}

The evidence `span` is kept so a small slice can be human-verified later to build
a gold evaluation set (see evaluate.py).
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
_FEWSHOT = [
    (
        "Kremin kokusu harika ve cilde çok iyi geliyor ama kargo tam bir "
        "rezaletti, 10 günde geldi. Fiyatına da değer doğrusu.",
        {
            "aspects": [
                {"aspect": "koku", "sentiment": "pozitif", "span": "kokusu harika"},
                {"aspect": "urun_kalitesi", "sentiment": "pozitif", "span": "cilde çok iyi geliyor"},
                {"aspect": "kargo_teslimat", "sentiment": "negatif", "span": "kargo tam bir rezaletti, 10 günde geldi"},
                {"aspect": "fiyat", "sentiment": "pozitif", "span": "Fiyatına da değer"},
            ]
        },
    ),
    (
        "İdare eder, çok da özel değil. Paketleme normaldi.",
        {
            "aspects": [
                {"aspect": "urun_kalitesi", "sentiment": "notr", "span": "İdare eder, çok da özel değil"},
                {"aspect": "ambalaj", "sentiment": "notr", "span": "Paketleme normaldi"},
            ]
        },
    ),
]


def build_system(cfg: dict) -> str:
    """Assemble the system prompt: role, closed taxonomy, rules, and examples."""
    taxonomy = "\n".join(f"  - {name}: {desc}" for name, desc in cfg["aspects"].items())
    labels = ", ".join(cfg["sentiment_labels"])
    examples = "\n\n".join(
        f"Yorum: {txt}\nÇıktı: {json.dumps(out, ensure_ascii=False)}"
        for txt, out in _FEWSHOT
    )
    return f"""Sen Türkçe e-ticaret yorumlarını etiketleyen uzman bir veri etiketleyicisisin.

Görevin: Verilen yorumda AÇIKÇA değinilen aspect'leri aşağıdaki KAPALI listeden seç ve her birine bir duygu etiketi ata.

Aspect listesi (yalnızca bunları kullan):
{taxonomy}

Duygu etiketleri (yalnızca bunlardan biri): {labels}

Kurallar:
  - Sadece yorumda gerçekten bahsedilen aspect'leri döndür. Bahsedilmeyeni UYDURMA.
  - Olumsuzlamaya dikkat et ("fena değil" olumsuz DEĞİLDİR; "hiç beğenmedim" olumsuzdur).
  - Her aspect için 'span' alanına o yargıyı destekleyen kısa metin parçasını yorumdan AYNEN kopyala.
  - Aynı aspect birden çok geçiyorsa tek bir kayıtta birleştir.
  - Hiçbir aspect yoksa boş liste döndür.
  - SADECE geçerli JSON döndür. Markdown, açıklama, ekstra metin YOK.

Çıktı şeması: {{"aspects": [{{"aspect": "...", "sentiment": "...", "span": "..."}}]}}

Örnekler:
{examples}"""


def build_user(text: str) -> str:
    return f"Yorum: {text}\nÇıktı:"


# --------------------------------------------------------------------------- #
# LLM backends
# --------------------------------------------------------------------------- #
def _call_anthropic(system: str, user: str, cfg: dict) -> str:
    from anthropic import Anthropic

    client = _get_client(cfg)
    # Prefill the assistant turn with "{" to force a JSON object start.
    resp = client.messages.create(
        model=cfg["labeling"]["model"],
        max_tokens=512,
        system=system,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": "{"},
        ],
    )
    return "{" + resp.content[0].text


def _call_ollama(system: str, user: str, cfg: dict) -> str:
    import ollama

    resp = ollama.chat(
        model=cfg["labeling"]["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format="json",  # Ollama constrains output to valid JSON
    )
    return resp["message"]["content"]


# Single shared Anthropic client (thread-safe via httpx connection pool).
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _get_client(cfg: dict):
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from anthropic import Anthropic

            _CLIENT = Anthropic()
    return _CLIENT


def call_llm(system: str, user: str, cfg: dict) -> str:
    provider = cfg["labeling"]["provider"]
    if provider == "anthropic":
        return _call_anthropic(system, user, cfg)
    if provider == "ollama":
        return _call_ollama(system, user, cfg)
    raise ValueError(f"Unknown provider: {provider}")


# --------------------------------------------------------------------------- #
# Parsing & validation
# --------------------------------------------------------------------------- #
def parse_response(raw: str, cfg: dict) -> list[dict] | None:
    """Extract and validate the aspect list. Returns None on unrecoverable failure
    (so the caller can retry); returns a (possibly empty) list on success."""
    valid_aspects = set(cfg["aspects"].keys())
    valid_labels = set(cfg["sentiment_labels"])

    # Isolate the JSON object even if wrapped in fences or trailing prose.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None

    items = obj.get("aspects", []) if isinstance(obj, dict) else []
    cleaned, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            continue
        aspect = str(it.get("aspect", "")).strip()
        sentiment = str(it.get("sentiment", "")).strip().lower()
        if aspect not in valid_aspects or sentiment not in valid_labels:
            continue  # drop hallucinated aspects / invalid labels
        if aspect in seen:
            continue  # keep first occurrence
        seen.add(aspect)
        cleaned.append({"aspect": aspect, "sentiment": sentiment, "span": str(it.get("span", ""))})
    return cleaned


def annotate_one(row: dict, system: str, cfg: dict) -> dict:
    """Label a single review with retries. Always returns a record."""
    user = build_user(row["text"])
    for attempt in range(cfg["labeling"]["max_retries"] + 1):
        try:
            raw = call_llm(system, user, cfg)
            parsed = parse_response(raw, cfg)
            if parsed is not None:
                return {**_base(row), "aspects": parsed, "error": False}
        except Exception:  # network / rate limit / SDK errors -> back off and retry
            time.sleep(1.5 * (attempt + 1))
    return {**_base(row), "aspects": [], "error": True}


def _base(row: dict) -> dict:
    return {"id": row["id"], "text": row["text"], "doc_sentiment": row["doc_sentiment"]}


# --------------------------------------------------------------------------- #
# Orchestration (sampled, resumable, concurrent)
# --------------------------------------------------------------------------- #
def _done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _sample(train: pl.DataFrame, lc: dict, seed: int) -> pl.DataFrame:
    """Stable random head (so growing sample_n stays a superset) + an optional
    keyword-boosted draw that oversamples the data-starved aspects."""
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=seed)
    base = shuffled.head(lc["sample_n"])
    kws = lc.get("boost_keywords")
    if not kws:
        return base
    pattern = "(?i)" + "|".join(kws)
    boost = shuffled.filter(pl.col("text").str.contains(pattern)).head(lc.get("boost_n", 1500))
    return pl.concat([base, boost]).unique(subset="id", keep="first")


def run(cfg: dict) -> None:
    lc = cfg["labeling"]
    out_path = Path(lc["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = pl.read_parquet(Path(cfg["data"]["splits_dir"]) / "train.parquet")
    sample = _sample(train, lc, cfg["data"]["seed"])

    done = _done_ids(out_path)
    todo = [r for r in sample.to_dicts() if r["id"] not in done]
    print(f"to label: {len(todo)} (already done: {len(done)})")
    if not todo:
        return

    system = build_system(cfg)
    write_lock = threading.Lock()
    n_err = 0

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=lc["workers"]) as ex:
        futures = {ex.submit(annotate_one, row, system, cfg): row for row in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            n_err += int(rec["error"])
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} done, {n_err} errors")

    print(f"finished: {len(todo)} labeled, {n_err} errors -> {out_path}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        run(yaml.safe_load(f))