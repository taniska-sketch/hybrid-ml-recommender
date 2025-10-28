import os
import numpy as np
import pandas as pd
import joblib

# BASE DIRECTORIES
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# FILE PATHS
META_CSV = os.path.join(DATA_DIR, "songs_metadata_for_api.csv")
SCALED_CSV = os.path.join(DATA_DIR, "scaled_feature_sample.csv")
CB_PKL = os.path.join(MODEL_DIR, "content_similarity.pkl")
CF_PKL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
UIM_PKL = os.path.join(MODEL_DIR, "user_item_matrix.pkl")

# GLOBALS
METADATA = pd.DataFrame()
CB_SIM = None
CB_INDEX = {}
CF_SIM = None
USER_ITEM_MATRIX = None
CF_INDEX = {}

def initialize():
    global METADATA, CB_SIM, CB_INDEX, CF_SIM, USER_ITEM_MATRIX, CF_INDEX

    print("🔄 Initializing Recommender...")

    METADATA = pd.read_csv(META_CSV)
    METADATA["track_id"] = METADATA["track_id"].astype(str)

    df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
    CB_SIM = joblib.load(CB_PKL)
    CB_INDEX = {tid: i for i, tid in enumerate(df_scaled.index)}

    try:
        CF_SIM = joblib.load(CF_PKL)
        USER_ITEM_MATRIX = joblib.load(UIM_PKL)
        USER_ITEM_MATRIX.columns = USER_ITEM_MATRIX.columns.astype(str)
        CF_INDEX = {tid: i for i, tid in enumerate(USER_ITEM_MATRIX.columns)}
        print("✅ CF Loaded")
    except:
        CF_SIM = None
        USER_ITEM_MATRIX = None
        CF_INDEX = {}
        print("⚠️ CF model not loaded, only CB will be active")

    if "popularity" not in METADATA.columns:
        METADATA["popularity"] = METADATA["artist_name"].map(
            METADATA["artist_name"].value_counts().to_dict()
        )

    print("✅ METADATA:", len(METADATA))
    print("✅ Content tracks:", len(CB_INDEX))
    print("✅ CF tracks:", len(CF_INDEX))

def fallback_top_popular(top_n=10):
    df = METADATA.sort_values("popularity", ascending=False).head(top_n)
    return df.to_dict(orient="records")

def ensure_seed_or_fallback(track_id, top_n):
    if track_id not in CB_INDEX and track_id not in CF_INDEX:
        return fallback_top_popular(top_n)
    return None

def recommend_cb(track_id, top_n=10):
    if track_id not in CB_INDEX:
        return []
    idx = CB_INDEX[track_id]
    sims = CB_SIM[idx]
    top_ids = np.argsort(-sims)[1:top_n+1]
    return [(list(CB_INDEX.keys())[i], float(sims[i])) for i in top_ids]

def recommend_cf(track_id, top_n=10):
    if CF_SIM is None or track_id not in CF_INDEX:
        return []
    idx = CF_INDEX[track_id]
    sims = CF_SIM[idx]
    top_ids = np.argsort(-sims)[1:top_n+1]
    return [(USER_ITEM_MATRIX.columns[i], float(sims[i])) for i in top_ids]

def recommend_hybrid(seed_track_id, user_id=None, top_n=10, w_cb=0.4, w_cf=0.6):
    cb = dict(recommend_cb(seed_track_id, top_n * 2))
    cf = dict(recommend_cf(seed_track_id, top_n * 2))

    all_ids = set(cb.keys()).union(set(cf.keys()))
    result = []

    for tid in all_ids:
        score = w_cb * cb.get(tid, 0) + w_cf * cf.get(tid, 0)
        row = METADATA[METADATA["track_id"] == tid].iloc[0].to_dict()
        row["hybrid_score"] = round(score, 4)
        result.append(row)

    return sorted(result, key=lambda x: -x["hybrid_score"])[:top_n]

# INIT
try:
    initialize()
except Exception as e:
    print("❌ Initialization FAILED:", e)
