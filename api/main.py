# api/main.py

from fastapi import FastAPI, Query
from src.recommend import (
    METADATA,
    recommend_hybrid,
    fallback_top_popular
)

app = FastAPI(
    title="Hybrid ML Recommender API",
    version="1.0.0",
    description="Hybrid Collaborative + Content-based Music Recommendation"
)

@app.get("/")
def root():
    return {
        "message": "✅ Hybrid Recommender is Live!",
        "docs": "/docs",
        "example": "/tracks?limit=20",
        "hybrid": "/recommend/hybrid?track_id=6dyku3NZZukkS8yhzWG9TU"
    }

@app.get("/health")
def health():
    return {"status": "✅ OK - Service healthy"}

@app.get("/tracks")
def get_tracks(limit: int = Query(20, gt=0, lt=200)):
    df = METADATA.head(limit)
    return df.to_dict(orient="records")

@app.get("/recommend/hybrid")
def hybrid_api(track_id: str, user_id: str | None = None, top_n: int = 10):
    results = recommend_hybrid(track_id, user_id, top_n)
    if not results:
        return fallback_top_popular(top_n)
    
    return {
        "input_track_id": track_id,
        "results": results
    }
