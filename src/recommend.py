# src/recommend.py

import os
import numpy as np
import pandas as pd
import joblib
import gdown
import threading

# ----------------------------
# DIRECTORIES ✅
# ----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_DIR, "songs_metadata_for_api.csv")
SCALED_CSV = os.path.join(DATA_DIR, "scaled_feature_sample.csv")

CB_PKL = os.path.join(MODEL_DIR, "content_similarity.pkl")
CF_PKL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
UIM_PKL = os.path.join(MODEL_DIR, "user_item_matrix.pkl")

# ----------------------------
# GOOGLE DRIVE MODEL IDS ✅
# ----------------------------
FILE_IDS = {
    "content_similarity.pkl": "145t8B6RdV9GXJNDSEkvF5BXhOA4qq7XE",
    "item_similarity_cf_matrix.pkl": "1pmWdY5DCpUDp4--ej0EM912pgdohdzPS",
    "user_item_matrix.pkl": "1ldM64nwdj4hNSmwnBxpbIHEr0VR7BsWg"
}

def download_models_if_missing():
    for file_name, file_id in FILE_IDS.items():
        file_path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"⬇️ Downloading {file_name} ...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"✅ Already Exists: {file_name}")


# ----------------------------
# LAZY LOAD MODELS ✅
# ----------------------------
MODEL_LOCK = threading.Lock()
LOADED_MODELS = False

# Storage placeholders
METADATA = None
df_scaled = None
CB_SIM = None
CF_SIM = None

CB_TRACK_IDS = []
CF_TRACK_IDS = []
CB_INDEX = {}
CF_INDEX = {}

def load_models():
    global LOADED_MODELS, METADATA, df_scaled, CB_SIM, CF_SIM

    if LOADED_MODELS:
        return

    with MODEL_LOCK:
        if LOADED_MODELS:
            return

        print("📌 Loading models on demand...")
        download_models_if_missing()

        # Load metadata
        METADATA = pd.read_csv(META_CSV)
        METADATA["track_id"] = METADATA["track_id"].astype(str)

        # Popularity fallback if missing
        if "popularity" not in METADATA.columns:
            pop = METADATA["artist_name"].value_counts().to_dict()
            METADATA["popularity"] = METADATA["artist_name"].map(lambda x: pop.get(x, 1))

        # Load Content-based similarity
        if os.path.exists(SCALED_CSV) and os.path.exists(CB_PKL):
            df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
            CB_TRACK_IDS.extend(df_scaled.index.tolist())
            CB_SIM = joblib.load(CB_PKL)
            CB_INDEX.update({tid: i for i, tid in enumerate(CB_TRACK_IDS)})

        # Load CF similarity if exists
        if os.path.exists(CF_PKL) and os.path.exists(UIM_PKL):
            CF_SIM = joblib.load(CF_PKL)
            uim = joblib.load(UIM_PKL)
            uim.columns = uim.columns.astype(str)
            CF_TRACK_IDS.extend(list(uim.columns))
            CF_INDEX.update({tid: i for i, tid in enumerate(CF_TRACK_IDS)})

        LOADED_MODELS = True
        print("✅ Models loaded successfully!")


# ----------------------------
# FALLBACKS ✅
# ----------------------------
def fallback_top_popular(top_n=10):
    load_models()
    df = METADATA.sort_values("popularity", ascending=False).head(top_n)
    return df.to_dict(orient="records")


def ensure_seed_or_fallback(track_id, top_n=10):
    load_models()
    if track_id not in METADATA["track_id"].values:
        print("⚠️ Invalid track → using popular fallback")
        return fallback_top_popular(top_n)
    return None


# ----------------------------
# RECOMMENDERS ✅
# ----------------------------
def recommend_cb(track_id, top_n=10):
    load_models()
    if track_id not in CB_INDEX:
        return []
    i = CB_INDEX[track_id]
    sims = joblib.load(CB_PKL)[i]  # ✅ Load only needed row
    order = np.argsort(-sims)[1:top_n+1]
    return [(CB_TRACK_IDS[j], float(sims[j])) for j in order]


def recommend_cf(track_id, top_n=10):
    load_models()
    if CF_SIM is None or track_id not in CF_INDEX:
        return []
    i = CF_INDEX[track_id]
    sims = CF_SIM[i]
    order = np.argsort(-sims)[1:top_n+1]
    return [(CF_TRACK_IDS[j], float(sims[j])) for j in order]


def recommend_hybrid(track_id, user_id=None, top_n=10, w_cb=0.4, w_cf=0.6):
    load_models()

    fallback = ensure_seed_or_fallback(track_id, top_n)
    if fallback: return fallback

    cb = recommend_cb(track_id, top_n*2)
    cf = recommend_cf(track_id, top_n*2)

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
                "popularity": int(r.get("popularity", 0)),
                "hybrid_score": round(score, 4)
            })

    return result

