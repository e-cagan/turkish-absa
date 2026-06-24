"""FastAPI service for Turkish ABSA inference.

For each review we run one (review, aspect) pair per taxonomy aspect through the
fine-tuned BERTurk model in a single batched forward pass, drop the aspects
predicted `yok`, and return the present aspects with sentiment and a softmax
confidence. This mirrors the training/eval formulation exactly.

Run:  uv run python src/serve.py        (or: uvicorn src.serve:app --reload)
"""

from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import json
import yaml
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import ASPECT_PROBE, LABELS

CFG = yaml.safe_load(open("config.yaml"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    out_dir = CFG["serve"]["model_id"]
    STATE["tok"] = AutoTokenizer.from_pretrained(out_dir)
    STATE["model"] = AutoModelForSequenceClassification.from_pretrained(out_dir).to(DEVICE).eval()
    STATE["aspects"] = list(CFG["aspects"].keys())
    STATE["max_length"] = CFG["train"]["max_length"]
    yield
    STATE.clear()


app = FastAPI(title="Turkish ABSA", lifespan=lifespan)


class Review(BaseModel):
    text: str


class AspectOut(BaseModel):
    aspect: str
    sentiment: str
    confidence: float


class Prediction(BaseModel):
    text: str
    aspects: list[AspectOut]


@torch.no_grad()
def analyze(text: str) -> list[AspectOut]:
    aspects = STATE["aspects"]
    probes = [ASPECT_PROBE[a] for a in aspects]
    enc = STATE["tok"](
        [text] * len(aspects), probes,
        truncation="only_first", max_length=STATE["max_length"],
        padding=True, return_tensors="pt",
    ).to(DEVICE)
    probs = F.softmax(STATE["model"](**enc).logits, dim=-1)
    conf, pred = probs.max(dim=-1)

    out = []
    for asp, p, c in zip(aspects, pred.tolist(), conf.tolist()):
        label = LABELS[p]
        if label != "yok":  # only emit aspects the review actually discusses
            out.append(AspectOut(aspect=asp, sentiment=label, confidence=round(c, 3)))
    return out


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict", response_model=Prediction)
def predict(review: Review):
    result = Prediction(text=review.text, aspects=analyze(review.text))
    # Log request + response as one copy-pasteable JSON line
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), flush=True)
    return result


_DIST = Path("frontend/dist")
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))