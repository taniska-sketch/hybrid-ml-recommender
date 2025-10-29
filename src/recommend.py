import os
import numpy as np
import pandas as pd
import pickle
import gdown

MODEL_DIR = "models"
META_PATH = "data/songs_metadata_for_api.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive Model Files (ID, Filename)
DRIVE_FILES = {
    "cf": ("1ldM64nwdj4hNSmwnBxpbIHEr0VR7BsWg", "item_similarity_cf_matrix.pkl"),
    "cb": ("145t8B6RdV9GXJNDSEkvF5BXhOA4qq7XE", "content_similarity.pkl"),
    "uim": ("1pmWdY5DCpUDp4--ej0EM912pgdohdzPS", "user_item_matrix.pkl"),
}


def ensure_models_present():
    for key, (file_id, filename) in DRIVE_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"📥 Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)
    print("✅ Models ready")


_CB_MODEL = None
_CF_MODEL = None
_UIM = None
_META = None


def _lazy_load_models():
    global _CB_MODEL, _CF_MODEL, _UIM, _META

    ensure_models_present()

    if _CB_MODEL is None:
        print("Loading Content-Based model...")
        _CB_MODEL = np.load(os.path.join(MODEL_DIR, DRIVE_FILES["cb"][1]), allow_pickle=True)

    if _CF_MODEL is None:
        print("Loading CF model...")
        _CF_MODEL = np.load(os.path.join(MODEL_DIR, DRIVE_FILES["cf"][1]), allow_pickle=True)

    if _UIM is None:
        print("Loading User-Item Matrix...")
        with open(os.path.join(MODEL_DIR, DRIVE_FILES["uim"][1]), "rb") as f:
            _UIM = pickle.load(f)

    if _META is None:
        print("Loading metadata...")
        _META = pd.read_csv(META_PATH)
        _META.set_index("track_id", inplace=True)


def _idx(track_id: str):
    if track_id not in _META.index:
        return None
    return _META.index.tolist().index(track_id)


def fetch_metadata(limit=20):
    _lazy_load_models()
    return _META.reset_index().head(limit).to_dict(orient="records")


def _build_response(indices, scores, method, top_n):
    recs = []
    idx_list = _META.index.tolist()

    for i, score in zip(indices[:top_n], scores[:top_n]):
        tid = idx_list[i]
        row = _META.loc[tid]
        recs.append({
            "track_id": tid,
            "track_name": row.get("track_name", "Unknown"),
            "artist_name": row.get("artist_name", "Unknown Artist"),
            "popularity": int(row.popularity) if pd.notna(row.popularity) else None,
            "score": float(score),
            "method": method
        })
    return recs


def recommend_cb(track_id: str, top_n=10):
    _lazy_load_models()
    idx = _idx(track_id)
    if idx is None:
        return None
    scores = _CB_MODEL[idx]
    inds = np.argsort(scores)[::-1]
    return _build_response(inds[1:], scores[inds][1:], "content", top_n)


def recommend_cf(track_id: str, top_n=10):
    _lazy_load_models()
    idx = _idx(track_id)
    if idx is None:
        return None
    scores = _CF_MODEL[idx]
    inds = np.argsort(scores)[::-1]
    return _build_response(inds[1:], scores[inds][1:], "collab", top_n)


def recommend_hybrid(track_id: str, top_n=10):
    _lazy_load_models()

    try:
        cb = recommend_cb(track_id, top_n * 3)
        cf = recommend_cf(track_id, top_n * 3)

        if not cb or not cf:
            return cb or cf

        combo = {}

        for r in cb:
            combo[r['track_id']] = r['score']

        for r in cf:
            combo[r['track_id']] = combo.get(r['track_id'], 0) + r['score']

        sorted_items = sorted(combo.items(), key=lambda x: x[1], reverse=True)

        indices = [list(_META.index).index(tid) for tid, _ in sorted_items]
        scores = [score for _, score in sorted_items]

        return _build_response(indices, scores, "hybrid", top_n)

    except Exception:
        return recommend_cb(track_id, top_n)
