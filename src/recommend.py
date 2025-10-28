# src/recommend.py

import os
import numpy as np
import pandas as pd
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"

META_CSV = os.path.join(DATA_DIR, "songs_metadata_for_api.csv")
SCALED_CSV = os.path.join(DATA_DIR, "scaled_feature_sample.csv")
CB_PKL = os.path.join(MODEL_DIR, "content_similarity.pkl")
CF_PKL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
UIM_PKL = os.path.join(MODEL_DIR, "user_item_matrix.pkl")

# ✅ Load metadata only (small)
METADATA = pd.read_csv(META_CSV)
METADATA["track_id"] = METADATA["track_id"].astype(str)

if "popularity" not in METADATA.columns:
    pop = METADATA["artist_name"].value_counts().to_dict()
    METADATA["popularity"] = METADATA["artist_name"].map(lambda a: pop.get(a, 1))

# ✅ Globals to load later
CB_SIM = None
CF_SIM = None
CB_INDEX = {}
CF_INDEX = {}

def lazy_load_cb():
    """
    Loads content-based similarity memory only when needed
    """
    global CB_SIM, CB_INDEX
    if CB_SIM is None:
        df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
        track_ids = df_scaled.index.tolist()
        CB_SIM = joblib.load(CB_PKL)
        CB_INDEX = {tid: i for i, tid in enumerate(track_ids)}

def lazy_load_cf():
    """
    Loads CF similarity only when needed
    """
    global CF_SIM, CF_INDEX
    if CF_SIM is None and os.path.exists(CF_PKL):
        CF_SIM = joblib.load(CF_PKL)
        uim = joblib.load(UIM_PKL)
        uim.columns = uim.columns.astype(str)
        CF_INDEX = {tid: i for i, tid in enumerate(uim.columns)}

def fallback_top_popular(top_n=10):
    df = METADATA.sort_values("popularity", ascending=False).head(top_n)
    return df.to_dict(orient="records")

def ensure_seed_or_fallback(track_id, top_n=10):
    if track_id not in METADATA["track_id"].values:
        return fallback_top_popular(top_n)
    return None

def recommend_cb(track_id, top_n=10):
    lazy_load_cb()
    if track_id not in CB_INDEX:
        return []
    i = CB_INDEX[track_id]
    sims = CB_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [(list(CB_INDEX.keys())[j], float(sims[j])) for j in order]

def recommend_cf(track_id, top_n=10):
    lazy_load_cf()
    if CF_SIM is None or track_id not in CF_INDEX:
        return []
    i = CF_INDEX[track_id]
    sims = CF_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [(list(CF_INDEX.keys())[j], float(sims[j])) for j in order]

def build_metadata_response(results, top_n=10):
    output = []
    for tid, score in results[:top_n]:
        r = METADATA[METADATA["track_id"] == tid].iloc[0]
        output.append({
            "track_id": tid,
            "track_name": r["track_name"],
            "artist_name": r["artist_name"],
            "popularity": int(r.get("popularity", 0)),
            "score": round(score, 4)
        })
    return output

def recommend_hybrid(track_id, user_id=None, top_n=10, w_cb=0.4, w_cf=0.6):
    cb = recommend_cb(track_id, top_n*2)
    cf = recommend_cf(track_id, top_n*2)

    if not cb and not cf:
        return fallback_top_popular(top_n)

    combined = {}
    for tid, score in cb:
        combined[tid] = combined.get(tid, 0) + w_cb * score
    for tid, score in cf:
        combined[tid] = combined.get(tid, 0) + w_cf * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return build_metadata_response(ranked, top_n)
