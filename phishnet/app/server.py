"""PhishNet platform: real-time API + web UI for the CNN-LSTM phishing URL detector.

    uvicorn app.server:app --port 8000
    (PHISHNET_MODEL selects another checkpoint; default models/cnn_lstm.pt)
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference import Predictor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("PHISHNET_MODEL", ROOT / "models" / "cnn_lstm.pt"))
REPORTS = ROOT / "reports"
STATIC = Path(__file__).parent / "static"
DB_PATH = ROOT / "data" / "history.db"
state: dict = {}
db_lock = threading.Lock()


def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, url TEXT, label TEXT,
        probability REAL, source TEXT)""")
    return conn


def log_scans(rows, source):
    with db_lock:
        state["db"].executemany("INSERT INTO scans (ts, url, label, probability, source) VALUES (?,?,?,?,?)",
                                [(time.time(), r["url"][:2048], r["label"], r["probability"], source) for r in rows])
        state["db"].commit()


def load_predictor():
    if MODEL_PATH.exists():
        state["predictor"] = Predictor(str(MODEL_PATH))
        state["model_mtime"] = MODEL_PATH.stat().st_mtime


@asynccontextmanager
async def lifespan(_app):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["db"] = db()
    load_predictor()
    yield
    state["db"].close()
    state.clear()


app = FastAPI(title="PhishNet API", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
REPORTS.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS), name="reports")


def predictor() -> Predictor:
    # Pick up a freshly trained checkpoint without restarting the server.
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime != state.get("model_mtime"):
        load_predictor()
    if "predictor" not in state:
        raise HTTPException(503, f"No trained model at {MODEL_PATH}. Run: python -m src.train")
    return state["predictor"]


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=4096)
    explain: bool = True


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=5000)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    p = state.get("predictor")
    return {"status": "ok" if p else "no_model", "model": MODEL_PATH.name,
            **({k: v for k, v in p.info.items() if k != "test_metrics"} if p else {})}


@app.post("/api/predict")
def predict(req: URLRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "empty URL")
    result = predictor().predict(url, explain=req.explain)
    log_scans([result], "single")
    return result


@app.post("/api/batch")
def batch(req: BatchRequest):
    p = predictor()
    urls = [u.strip() for u in req.urls if u.strip()]
    if not urls:
        raise HTTPException(400, "no URLs")
    t0 = time.perf_counter()
    probs = p.predict_proba(urls)
    rows = [{"url": u, "probability": pr, "label": p.label(pr)} for u, pr in zip(urls, probs)]
    log_scans(rows, "batch")
    return {"results": rows, "threshold": p.threshold,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/api/history")
def history(limit: int = 50):
    with db_lock:
        cur = state["db"].execute(
            "SELECT ts, url, label, probability, source FROM scans ORDER BY id DESC LIMIT ?", (min(limit, 500),))
        rows = [dict(zip(("ts", "url", "label", "probability", "source"), r)) for r in cur.fetchall()]
        total, phish = state["db"].execute(
            "SELECT COUNT(*), COALESCE(SUM(label = 'phishing'), 0) FROM scans").fetchone()
    return {"total": total, "phishing": phish, "recent": rows}


@app.delete("/api/history")
def clear_history():
    with db_lock:
        state["db"].execute("DELETE FROM scans")
        state["db"].commit()
    return {"ok": True}


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/model")
def model_report(run: str = "cnn_lstm"):
    data = _read(REPORTS / run / "metrics.json")
    if data is None:
        raise HTTPException(404, f"No report for run '{run}' yet - train it first.")
    data["images"] = [f"/reports/{run}/{n}" for n in ("training_curves.png", "confusion_matrix.png",
                                                      "roc_pr_curves.png") if (REPORTS / run / n).exists()]
    return data


@app.get("/api/compare")
def compare():
    runs = []
    for path in sorted(REPORTS.glob("*/metrics.json")):
        d = _read(path)
        if d and "test" in d:
            runs.append({"run": d["run"], "arch": d.get("arch"), "test": d["test"],
                         "robustness": d.get("robustness"), "train_seconds": d.get("train_seconds")})
    return runs


@app.get("/api/dataset")
def dataset():
    data = _read(REPORTS / "dataset.json")
    if data is None:
        raise HTTPException(404, "Run: python -m src.dataset_stats")
    return data
