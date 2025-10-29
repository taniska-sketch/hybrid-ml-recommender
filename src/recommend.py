# src/recommend.py

import os
import pandas as pd
import numpy as np
import joblib

CF_MODEL = None
METADATA = None

def _load_cf_model():
    global CF_MODEL
    if CF_MODEL is None:
        print("🔄 Loading Collaborative Filtering model...")
        try:
            CF_MODEL = joblib.load("models/item_similarity_cf_matrix.pkl")

            # Ensure it's a DataFrame
            if not isinstance(CF_MODEL, pd.DataFrame):
                CF_MODEL = pd.DataFrame(CF_MODEL)

            print(f"✅ CF similarity matrix loaded: {CF_MODEL.shape}")
        except Exception as e:
            print("❌ Failed to load CF model:", e)
            raise
    return CF_MODEL


def _load_metadata():
    global METADATA
    if METADATA is None:
        print("🔄 Loading metadata...")
        try:
            METADATA = pd.read_csv("data/songs_metadata_for_api.csv")

            # Ensure correct index for joining
            if "track_id" not in METADATA.columns:
                raise ValueError("❌ track_id not in metadata!")

            METADATA.set_index("track_id", inplace=True)

            print(f"✅ Metadata loaded: {METADATA.shape}")

        except Exception as e:
            print("❌ Failed to load metadata:", e)
            raise

    return METADATA


def _align_metadata(cf):
    meta = _load_metadata()

    # Keep only track_ids in CF model
    cf_ids = list(cf.index)
    meta = meta[meta.index.isin(cf_ids)]

    # Reorder to match CF
    meta = meta.loc[cf_ids]

    print(f"✅ Metadata aligned to CF size: {meta.shape}")

    return meta


def recommend_cf(track_id: str, top_n: int = 10):
    cf = _load_cf_model()

    if track_id not in cf.index:
        return {"error": f"Track ID not found in CF model: {track_id}"}

    meta = _align_metadata(cf)

    # Retrieve similarity scores
    sim_scores = cf.loc[track_id].values

    # Sort by score
    sorted_idx = np.argsort(sim_scores)[::-1]

    results = []
    count = 0

    for idx in sorted_idx:
        similar_track = cf.index[idx]
        if similar_track == track_id:
            continue  # Skip the same track

        if similar_track in meta.index:
            info = meta.loc[similar_track]
            results.append({
                "track_id": similar_track,
                "track_name": str(info.get("track_name", "Unknown")),
                "artist_name": str(info.get("artist_name", "Unknown Artist")),
                "popularity": int(info.get("popularity")) if pd.notna(info.get("popularity")) else None,
                "score": float(sim_scores[idx]),  # ✅ Convert to Python float
                "method": "collab"
            })
            count += 1

        if count >= top_n:
            break

    return results
