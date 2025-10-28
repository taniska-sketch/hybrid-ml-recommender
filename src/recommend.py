# src/recommend.py
import os
import numpy as np
import pandas as pd
import joblib
import gdown

# ------------------------------------------
# GOOGLE DRIVE MODEL DOWNLOAD ✅
# ------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FILE_IDS = {
    "content_similarity.pkl": "145t8B6RdV9GXJNDSEkvF5BXhOA4qq7XE",
    "item_similarity_cf_matrix.pkl": "1pmWdY5DCpUDp4--ej0EM912pgdohdzPS",
    "user_item_matrix.pkl": "1ldM64nwdj4hNSmwnBxpbIHEr0VR7BsWg"
}

def download_models_if_missing():
    for file_name, file_id in FILE_IDS.items():
        file_path = os.path.join(MODEL_DIR, file_name)
        if not os.path.exists(file_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"⬇️ Downloading {file_name} ...")
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"✅ Already Exists: {file_name}")

download_models_if_missing()

# ------------------------------------------
# LOCAL DATA FILES ✅ (Upload to GitHub)
# ------------------------------------------
META_CSV = os.path.join(DATA_DIR, "songs_metadata_for_api.csv")
SCALED_CSV = os.path.join(DATA_DIR, "scaled_feature_sample.csv")

if not os.path.exists(META_CSV) or not os.path.exists(SCALED_CSV):
    raise FileNotFoundError("❌ CSV data missing! Upload to /data folder in GitHub")

METADATA = pd.read_csv(META_CSV)
METADATA["track_id"] = METADATA["track_id"].astype(str)

# Popularity fallback if needed
if "popularity" not in METADATA.columns:
    pop = METADATA["artist_name"].value_counts().to_dict()
    METADATA["popularity"] = METADATA["artist_name"].map(lambda x: pop.get(x, 1))

# ------------------------------------------
# Load CB Similarity ✅
# ------------------------------------------
df_scaled = pd.read_csv(SCALED_CSV, index_col="track_id")
CB_TRACK_IDS = df_scaled.index.tolist()
CB_SIM = joblib.load(os.path.join(MODEL_DIR, "content_similarity.pkl"))
CB_INDEX = {tid: i for i, tid in enumerate(CB_TRACK_IDS)}

# ------------------------------------------
# Load CF Similarity ✅
# ------------------------------------------
CF_SIM = None
CF_INDEX = {}
CF_PKL = os.path.join(MODEL_DIR, "item_similarity_cf_matrix.pkl")
UIM_PKL = os.path.join(MODEL_DIR, "user_item_matrix.pkl")

if os.path.exists(CF_PKL) and os.path.exists(UIM_PKL):
    CF_SIM = joblib.load(CF_PKL)
    uim = joblib.load(UIM_PKL)
    uim.columns = uim.columns.astype(str)
    CF_TRACK_IDS = list(uim.columns)
    CF_INDEX = {tid: i for i, tid in enumerate(CF_TRACK_IDS)}

# ------------------------------------------
# RECOMMENDERS ✅
# ------------------------------------------
def fallback_top_popular(top_n=10):
    df = METADATA.sort_values("popularity", ascending=False).head(top_n)
    return df.to_dict(orient="records")

def ensure_seed_or_fallback(track_id, top_n=10):
    if track_id not in METADATA["track_id"].values:
        print("⚠️ Seed missing → Popular fallback")
        return fallback_top_popular(top_n)
    return None

def recommend_cb(track_id, top_n=10):
    if track_id not in CB_INDEX:
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
    cb = recommend_cb(track_id, top_n*2)
    cf = recommend_cf(track_id, top_n*2)

    combined = {}
    for tid, score in cb:
        combined[tid] = combined.get(tid, 0.0) + w_cb * score
    for tid, score in cf:
        combined[tid] = combined.get(tid, 0.0) + w_cf * score

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
