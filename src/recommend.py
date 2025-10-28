import os
import numpy as np
import pandas as pd
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"

# ✅ Load metadata
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

# ✅ Safe load (only if file exists)
CB_SIM, CB_INDEX, CB_TRACK_IDS = None, {}, []
if os.path.exists(SCALED_CSV) and os.path.exists(CB_PKL):
    df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
    CB_TRACK_IDS = df_scaled.index.tolist()
    CB_SIM = joblib.load(CB_PKL)
    CB_INDEX = {tid: i for i, tid in enumerate(CB_TRACK_IDS)}

CF_SIM, CF_INDEX = None, {}
if os.path.exists(CF_PKL) and os.path.exists(UIM_PKL):
    CF_SIM = joblib.load(CF_PKL)
    uim = joblib.load(UIM_PKL)
    uim.columns = uim.columns.astype(str)
    CF_TRACK_IDS = list(uim.columns)
    CF_INDEX = {tid: i for i, tid in enumerate(CF_TRACK_IDS)}

def fallback_popular(top_n=10):
    df = METADATA.sort_values("popularity", ascending=False).head(top_n)
    return df.to_dict(orient="records")

def recommend_cb(track_id, top_n=10):
    if CB_SIM is None or track_id not in CB_INDEX:
        return []
    i = CB_INDEX[track_id]
    sims = CB_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [(CB_TRACK_IDS[j], float(sims[j])) for j in order]

def recommend_cf(track_id, top_n=10):
    if CF_SIM is None or track_id not in CF_INDEX:
        return []
    i = CF_INDEX[track_id]
    sims = CF_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [(CF_TRACK_IDS[j], float(sims[j])) for j in order]

def recommend_hybrid(track_id, user_id=None, top_n=10, w_cb=0.4, w_cf=0.6):
    cb = recommend_cb(track_id, top_n * 3)
    cf = recommend_cf(track_id, top_n * 3)

    if not cb and not cf:
        return fallback_popular(top_n)

    combined = {}
    for tid, score in cb:
        combined[tid] = combined.get(tid, 0) + w_cb * score
    for tid, score in cf:
        combined[tid] = combined.get(tid, 0) + w_cf * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    result = []
    for tid, score in ranked[:top_n]:
        row = METADATA[METADATA["track_id"] == tid]
        if not row.empty:
            r = row.iloc[0]
            result.append({
                "track_id": tid,
                "track_name": r["track_name"],
                "artist_name": r["artist_name"],
                "popularity": int(r["popularity"]),
                "hybrid_score": round(score, 3)
            })
    return result or fallback_popular(top_n)
