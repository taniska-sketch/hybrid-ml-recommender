import os
import numpy as np
import pandas as pd
import pickle
import requests
import gdown
import time

MODEL_DIR = "models/"
META_PATH = "data/songs_metadata_for_api.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Correct Google Drive file IDs & filenames
DRIVE_FILES = {
    "cb": ("145t8B6RdV9GXJNDSEkvF5BXhOA4qq7XE", "content_similarity.pkl"),
    "cf": ("1ldM64nwdj4hNSmwnBxpbIHEr0VR7BsWg", "item_similarity_cf_matrix.pkl"),
    "uim": ("1pmWdY5DCpUDp4--ej0EM912pgdohdzPS", "user_item_matrix.pkl"),
}


def _requests_drive_download(file_id: str, out_path: str):
    """Google Drive confirm token handling"""
    print(f"🌍 HTTP fallback download: {out_path}")
    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if "download_warning" in k:
            token = v

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)


def ensure_models_present():
    """Download missing ML files from Google Drive"""
    for key, (file_id, filename) in DRIVE_FILES.items():
        path = MODEL_DIR + filename
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue

        print(f"📥 Downloading model: {filename}...")
        try:
            gdown.download(id=file_id, output=path, quiet=False, use_cookies=False)
        except Exception:
            _requests_drive_download(file_id, path)

    print("✅ All models downloaded and ready!")


_CB_MODEL = None
_CF_MODEL = None
_UIM = None
_META = None


def _lazy_load_models():
    """Load models only when needed"""
    ensure_models_present()
    global _CB_MODEL, _CF_MODEL, _UIM, _META

    if _CB_MODEL is None:
        print("🔹 Loading CB model...")
        _CB_MODEL = np.load(MODEL_DIR + DRIVE_FILES["cb"][1], allow_pickle=True)

    if _CF_MODEL is None:
        print("🔸 Loading CF model...")
        _CF_MODEL = np.load(MODEL_DIR + DRIVE_FILES["cf"][1], allow_pickle=True)

    if _UIM is None:
        print("🧩 Loading User-Item Matrix...")
        with open(MODEL_DIR + DRIVE_FILES["uim"][1], "rb") as f:
            _UIM = pickle.load(f)

    if _META is None:
        print("📄 Loading metadata...")
        _META = pd.read_csv(META_PATH)
        _META.set_index("track_id", inplace=True)


def _idx(tid: str):
    if tid not in _META.index:
        return None
    return list(_META.index).index(tid)


def fetch_metadata(limit=20):
    _lazy_load_models()
    df = _META.reset_index().head(limit)
    df["popularity"] = df["popularity"].astype(float).fillna(0).astype(int)
    return df.to_dict(orient="records")


def _final_payload(indices, scores, method, top_n):
    items = []
    ids = list(_META.index)
    for i, s in zip(indices[:top_n], scores[:top_n]):
        row = _META.iloc[i]
        items.append({
            "track_id": ids[i],
            "track_name": row.get("track_name", "Unknown"),
            "artist_name": row.get("artist_name", "Unknown Artist"),
            "popularity": int(row.popularity) if pd.notna(row.popularity) else None,
            "score": float(s),
            "method": method
        })
    return items


def recommend_cb(track_id: str, top_n=10):
    _lazy_load_models()
    idx = _idx(track_id)
    if idx is None:
        return []
    scores = _CB_MODEL[idx]
    rank = np.argsort(scores)[::-1]
    return _final_payload(rank[1:], scores[rank][1:], "content", top_n)


def recommend_cf(track_id: str, top_n=10):
    _lazy_load_models()
    idx = _idx(track_id)
    if idx is None:
        return []
    scores = _CF_MODEL[idx]
    rank = np.argsort(scores)[::-1]
    return _final_payload(rank[1:], scores[rank][1:], "collab", top_n)


def recommend_hybrid(track_id: str, top_n=10):
    """Final hybrid: Score merge (no CF = fallback to CB)"""
    cb = recommend_cb(track_id, top_n * 3)
    cf = recommend_cf(track_id, top_n * 3)

    if not cb:
        return cf
    if not cf:
        return cb

    merged = {}
    for r in cb:
        merged[r["track_id"]] = r["score"]
    for r in cf:
        merged[r["track_id"]] = merged.get(r["track_id"], 0) + r["score"]

    sorted_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    indices = [list(_META.index).index(tid) for tid, _ in sorted_items]
    scores = [score for _, score in sorted_items]

    return _final_payload(indices, scores, "hybrid", top_n)
