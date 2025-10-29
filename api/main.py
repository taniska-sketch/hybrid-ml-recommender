# api/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.recommend import recommend_cf, fetch_metadata

app = FastAPI(
    title="Collaborative Filtering Music Recommender",
    version="2.0.0",
)

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "✅ CF Recommender is Live!",
        "docs": "/docs",
        "tracks": "/tracks?limit=20",
        "collab_example": "/recommend/collab?track_id=1X8uhUgBKmotpvHrsS7fEe&top_n=10"
    }

@app.get("/tracks", tags=["Tracks"])
def get_tracks(limit: int = 20):
    return fetch_metadata(limit)

@app.get("/recommend/collab", tags=["Collaborative Filtering"])
def collab(track_id: str, top_n: int = 10):
    try:
        recs = recommend_cf(track_id, top_n)
        if recs is None:
            return JSONResponse(status_code=404, content={"detail": "Track not found"})
        return recs
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
