"""PhishGuard real-time API + web UI.

    uvicorn app.server:app --port 8000
    (set PHISHGUARD_MODEL to use a different checkpoint)
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference import Predictor

MODEL_PATH = os.environ.get("PHISHGUARD_MODEL", "models/phishguard_hybrid.pt")
STATIC = Path(__file__).parent / "static"
state = {}


@asynccontextmanager
async def lifespan(_app):
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Train one first: python -m src.train --data ...")
    state["predictor"] = Predictor(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="PhishGuard API", version="1.0", lifespan=lifespan)
# Allows the browser extension (chrome-extension:// origin) to call the API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)
    explain: bool = True


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=1000)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, **state["predictor"].info}


@app.post("/api/predict")
def predict(req: URLRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "empty URL")
    return state["predictor"].predict(url, explain=req.explain)


@app.post("/api/batch")
def batch(req: BatchRequest):
    p = state["predictor"]
    probs = p.predict_proba(req.urls)
    return [{"url": u, "probability": pr, "label": "phishing" if pr >= p.threshold else "legitimate"}
            for u, pr in zip(req.urls, probs)]
