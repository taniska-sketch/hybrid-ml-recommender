# src/recommend.py
import os
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = "models"
CF_MODEL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
META_FILE = "data/songs_metadata_for_api_fixed.csv"

print("🔄 Loading Collaborative Filtering model...")

# Load Item-Item similarity matrix
item_sim = joblib.load(CF_MODEL)
print("✅ CF similarity matrix loaded:", item_sim.shape)

# Load metadata
meta = pd.read_csv(META_FILE)
meta.set_index("track_id", inplace=True)
print("✅ Metadata loaded:", meta.shape)


def recommend_cf(track_id: str, top_n: int = 10):
    if track_id not in meta.index:
        print("⚠ Track ID not found in metadata")
        return None

    idx = meta.index.tolist().index(track_id)
    scores = item_sim[idx]

    top_indices = np.argsort(scores)[::-1][1:top_n + 1]
    top_ids = meta.index[top_indices]

    recs = []
    for tid, score in zip(top_ids, scores[top_indices]):
        info = meta.loc[tid].to_dict()
        info["track_id"] = tid
        info["score"] = float(score)
        info["method"] = "collab"
        recs.append(info)

    return recs


def fetch_metadata(limit: int = 20):
    return meta.head(limit).reset_index().to_dict(orient="records")
