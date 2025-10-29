import os
import numpy as np
import pandas as pd
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_DIR, "songs_metadata_for_api.csv")
SCALED_CSV = os.path.join(DATA_DIR, "scaled_feature_sample.csv")

CB_PKL = os.path.join(MODEL_DIR, "content_similarity.pkl")
CF_PKL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
UIM_PKL = os.path.join(MODEL_DIR, "user_item_matrix.pkl")

METADATA = pd.read_csv(META_CSV)
METADATA["track_id"] = METADATA["track_id"].astype(str)

if "popularity" not in METADATA.columns:
    pop = METADATA["artist_name"].value_counts().to_dict()
    METADATA["popularity"] = METADATA["artist_name"].map(lambda x: pop.get(x, 1))


# ✅ Lazy load content-based similarity
def load_cb():
    df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
    track_ids = df_scaled.index.tolist()
    sim = joblib.load(CB_PKL)
    index = {tid: i for i, tid in enumerate(track_ids)}
    return sim, track_ids, index


# ✅ Lazy load CF similarity
def load_cf():
    if not os.path.exists(CF_PKL) or not os.path.exists(UIM_PKL):
        return None, None, {}
    sim = joblib.load(CF_PKL)
    uim = joblib.load(UIM_PKL)
    uim.columns = uim.columns.astype(str)
    track_ids = list(uim.columns)
    index = {tid: i for i, tid in enumerate(track_ids)}
    return sim, track_ids, index


def fallback_popular(top_n=10):
    df = METADATA.sort_values("popularity", ascending=False).head(top_n)
    return df.to_dict(orient="records")


# ✅ RECOMMENDATION FUNCTIONS (the names API needs)

def recommend_content(track_id: str, top_n: int = 10):
    CB_SIM, CB_IDS, CB_INDEX = load_cb()
    if track_id not in CB_INDEX:
        return fallback_popular(top_n)

    i = CB_INDEX[track_id]
    sims = CB_SIM[i]
    order = np.argsort(-sims)[1:top_n + 1]

    results = []
    for j in order:
        tid = CB_IDS[j]
        info = METADATA[METADATA["track_id"] == tid]
        if not info.empty:
            row = info.iloc[0]
            results.append({
                "track_id": tid,
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "score": float(sims[j])
            })
    return results


def recommend_collab(track_id: str, top_n: int = 10):
    CF_SIM, CF_IDS, CF_INDEX = load_cf()
    if CF_SIM is None or track_id not in CF_INDEX:
        return fallback_popular(top_n)

    i = CF_INDEX[track_id]
    sims = CF_SIM[i]
    order = np.argsort(-sims)[1:top_n + 1]

    results = []
    for j in order:
        tid = CF_IDS[j]
        info = METADATA[METADATA["track_id"] == tid]
        if not info.empty:
            row = info.iloc[0]
            results.append({
                "track_id": tid,
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "score": float(sims[j])
            })
    return results


def recommend_hybrid(track_id: str, top_n: int = 10, w_cb=0.4, w_cf=0.6):
    cb = recommend_content(track_id, top_n * 2)
    cf = recommend_collab(track_id, top_n * 2)

    merged = {}
    for x in cb:
        merged[x["track_id"]] = merged.get(x["track_id"], 0) + w_cb * x["score"]
    for x in cf:
        merged[x["track_id"]] = merged.get(x["track_id"], 0) + w_cf * x["score"]

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for tid, score in ranked:
        row = METADATA[METADATA["track_id"] == tid]
        if not row.empty:
            r = row.iloc[0]
            results.append({
                "track_id": tid,
                "track_name": r["track_name"],
                "artist_name": r["artist_name"],
                "score": round(score, 4)
            })
    return results
