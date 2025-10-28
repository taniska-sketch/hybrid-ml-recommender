from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.recommend import (
    recommend_hybrid,
    recommend_content,
    recommend_collab,
    get_tracks_metadata
)

app = FastAPI(
    title="Hybrid ML Music Recommender API",
    version="1.0.0",
)

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "✅ Hybrid Recommender is Live!",
        "docs": "/docs",
        "example_metadata": "/tracks?limit=20",
        "example_hybrid": "/recommend/hybrid?track_id=6dyku3NZZukkS8yhzWG9TU",
        "example_content": "/recommend/content?track_id=6dyku3NZZukkS8yhzWG9TU",
        "example_collab": "/recommend/collab?track_id=6dyku3NZZukkS8yhzWG9TU",
    }

@app.get("/tracks", tags=["Tracks"])
def fetch_tracks(limit: int = 10):
    return get_tracks_metadata(limit)

@app.get("/recommend/hybrid", tags=["Hybrid Recommendation"])
def hybrid(track_id: str, top_n: int = 10):
    try:
        recs = recommend_hybrid(track_id, top_n)
        if recs is None:
            return JSONResponse(status_code=404, content={"detail": "Track not found"})
        return recs
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/recommend/content", tags=["Content-Based Recommendation"])
def content(track_id: str, top_n: int = 10):
    try:
        recs = recommend_content(track_id, top_n)
        if recs is None:
            return JSONResponse(status_code=404, content={"detail": "Track not found"})
        return recs
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/recommend/collab", tags=["Collaborative Filtering Recommendation"])
def collab(track_id: str, top_n: int = 10):
    try:
        recs = recommend_collab(track_id, top_n)
        if recs is None:
            return JSONResponse(status_code=404, content={"detail": "Track not found"})
        return recs
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
