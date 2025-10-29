import os
import numpy as np
import pandas as pd
import joblib

# --- Base Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

META_CSV = os.path.join(DATA_DIR, "songs_metadata_for_api.csv")
SCALED_CSV = os.path.join(DATA_DIR, "scaled_feature_sample.csv")

CB_PKL = os.path.join(MODEL_DIR, "content_similarity.pkl")
CF_PKL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
UIM_PKL = os.path.join(MODEL_DIR, "user_item_matrix.pkl")

# --- Load Metadata ---
METADATA = pd.read_csv(META_CSV)
METADATA["track_id"] = METADATA["track_id"].astype(str)

# popularity fallback if missing
if "popularity" not in METADATA.columns:
    pop = METADATA["artist_name"].value_counts().to_dict()
    METADATA["popularity"] = METADATA["artist_name"].map(lambda x: pop.get(x, 1))

# --- Lazy loaded globals ---
CB_SIM = None
CF_SIM = None
CB_INDEX = {}
CF_INDEX = {}
CB_TRACK_IDS = []
CF_TRACK_IDS = []


# -------------------------------------------------
# ✅ Lazy Loading Functions (Fix Out-Of-Memory)
# -------------------------------------------------
def load_cb_similarity():
    global CB_SIM, CB_INDEX, CB_TRACK_IDS
    if CB_SIM is None:
        df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
        CB_TRACK_IDS = df_scaled.index.tolist()
        CB_SIM = joblib.load(CB_PKL)
        CB_INDEX = {tid: i for i, tid in enumerate(CB_TRACK_IDS)}


def load_cf_similarity():
    global CF_SIM, CF_INDEX, CF_TRACK_IDS
    if CF_SIM is None and os.path.exists(CF_PKL) and os.path.exists(UIM_PKL):
        CF_SIM = joblib.load(CF_PKL)
        uim = joblib.load(UIM_PKL)
        uim.columns = uim.columns.astype(str)
        CF_TRACK_IDS = list(uim.columns)
        CF_INDEX = {tid: i for i, tid in enumerate(CF_TRACK_IDS)}


# -------------------------------------------------
# ✅ Utility Builders
# -------------------------------------------------
def build_track_info(tid, score=None):
    row = METADATA[METADATA["track_id"] == tid]
    if row.empty:
        return None
    r = row.iloc[0]
    result = {
        "track_id": tid,
        "track_name": r["track_name"],
        "artist_name": r["artist_name"],
        "popularity": int(r.get("popularity", 0))
    }
    if score is not None:
        result["score"] = round(score, 4)
    return result


# -------------------------------------------------
# ✅ Individual Recommenders
# -------------------------------------------------
def recommend_content(track_id, top_n=10):
    load_cb_similarity()
    if track_id not in CB_INDEX:
        return []
    i = CB_INDEX[track_id]
    sims = CB_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [build_track_info(CB_TRACK_IDS[j], float(sims[j])) for j in order]


def recommend_cf(track_id, top_n=10):
    load_cf_similarity()
    if CF_SIM is None or track_id not in CF_INDEX:
        return []
    i = CF_INDEX[track_id]
    sims = CF_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [build_track_info(CF_TRACK_IDS[j], float(sims[j])) for j in order]


# -------------------------------------------------
# ✅ Hybrid Recommender — FIXED ✅
# -------------------------------------------------
def recommend_hybrid(track_id, top_n=10, w_cb=0.4, w_cf=0.6):
    cb = {r["track_id"]: r["score"] for r in recommend_content(track_id, top_n * 3)}
    cf = {r["track_id"]: r["score"] for r in recommend_cf(track_id, top_n * 3)}

    if not cb and not cf:
        return []

    # Combine both models safely ✅
    combined = {}
    for tid, score in cb.items():
        combined[tid] = combined.get(tid, 0) + w_cb * score
    for tid, score in cf.items():
        combined[tid] = combined.get(tid, 0) + w_cf * score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [build_track_info(tid, score) for tid, score in ranked[:top_n]]
