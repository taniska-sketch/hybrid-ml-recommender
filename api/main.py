from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.recommend import (
    fetch_metadata,
    recommend_cb,
    recommend_cf,
    recommend_hybrid
)

app = FastAPI(
    title="Hybrid ML Music Recommender API",
    version="2.0.0",
)


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "✅ Hybrid Recommender is Live!",
        "hybrid_demo": "/recommend/hybrid?track_id=1X8uhUgBKmotpvHrsS7fEe&top_n=10",
        "tracks": "/tracks?limit=10"
    }


@app.get("/tracks", tags=["Tracks"])
def tracks(limit: int = 20):
    return fetch_metadata(limit)


@app.get("/recommend/content", tags=["Content-Based"])
def content(track_id: str, top_n: int = 10):
    try:
        return recommend_cb(track_id, top_n)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/recommend/collab", tags=["Collaborative Filtering"])
def collab(track_id: str, top_n: int = 10):
    try:
        return recommend_cf(track_id, top_n)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/recommend/hybrid", tags=["Hybrid"])
def hybrid(track_id: str, top_n: int = 10):
    try:
        return recommend_hybrid(track_id, top_n)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
