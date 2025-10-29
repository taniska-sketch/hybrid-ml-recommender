from fastapi import FastAPI, Query
from src.recommend import recommend_cf, get_cf_ids
from typing import Optional

app = FastAPI(
    title="CF Recommender API",
    version="1.2.0",
    description="Collaborative Filtering-based Music Recommendation"
)


@app.get("/")
def root():
    return {
        "message": "✅ CF Recommender is Live!",
        "docs": "/docs",
        "tracks": "/tracks?limit=20",
        "collab_example": "/recommend/collab?track_id=<valid_cf_id>&top_n=10",
        "debug_cf_ids": "/debug/cf_ids?limit=10"
    }


@app.get("/tracks")
def tracks(limit: int = 20):
    return get_cf_ids(limit=limit)


@app.get("/debug/cf_ids")
def debug_cf_ids(limit: int = 20):
    return get_cf_ids(limit=limit)


@app.get("/recommend/collab")
def collab(track_id: str = Query(...), top_n: int = 10):
    return recommend_cf(track_id, top_n)
