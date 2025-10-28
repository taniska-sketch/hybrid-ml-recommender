from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.recommend import recommend_hybrid

app = FastAPI(
    title="Hybrid Music Recommender API",
    description="Content + Collaborative Filtering based hybrid music recommender system",
    version="1.0"
)

@app.get("/")
async def root():
    return {
        "message": "✅ Hybrid Music Recommender API is running successfully!",
        "endpoints": {
            "docs": "/docs",
            "hybrid_recommendation": "/recommend/hybrid?track_id=<TRACK_ID>&user_id=<USER_ID(optional)>&top_n=10"
        },
        "example_track_id": "6dyku3NZZukkS8yhzWG9TU"
    }

@app.get("/health")
async def health():
    return {"status": "✅ OK - Service healthy"}

# ✅ Recommendation Request Body
class RecommendRequest(BaseModel):
    track_id: str
    user_id: Optional[str] = None
    top_n: int = 10
    w_cb: float = 0.4
    w_cf: float = 0.6

@app.post("/recommend/hybrid")
async def hybrid_recommend_api(req: RecommendRequest):
    result = recommend_hybrid(
        req.track_id,
        user_id=req.user_id,
        top_n=req.top_n,
        w_cb=req.w_cb,
        w_cf=req.w_cf
    )
    if not result:
        return {"error": "Track not found or no recommendations available"}
    return result

