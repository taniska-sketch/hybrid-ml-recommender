from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware

from src.recommend import (
    METADATA,
    fallback_top_popular,
    ensure_seed_or_fallback,
    recommend_cb,
    recommend_cf,
    recommend_hybrid
)

app = FastAPI(
    title="Hybrid Song Recommender API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class HybridRequest(BaseModel):
    track_id: str
    user_id: Optional[str] = None
    top_n: int = 10
    w_cb: float = 0.4
    w_cf: float = 0.6

@app.get("/health")
def health():
    return {"status": "ok", "tracks": len(METADATA)}

@app.get("/popular")
def popular(top_n: int = 10):
    return fallback_top_popular(top_n)

@app.get("/recommend/cb")
def rec_cb(track_id: str, top_n: int = 10):
    fb = ensure_seed_or_fallback(track_id, top_n)
    return fb if fb else recommend_cb(track_id, top_n)

@app.get("/recommend/cf")
def rec_cf(track_id: str, top_n: int = 10):
    return recommend_cf(track_id, top_n)

@app.post("/recommend/hybrid")
def rec_hybrid(payload: HybridRequest):
    fb = ensure_seed_or_fallback(payload.track_id, payload.top_n)
    return fb if fb else recommend_hybrid(
        payload.track_id, payload.user_id,
        payload.top_n, payload.w_cb, payload.w_cf
    )
