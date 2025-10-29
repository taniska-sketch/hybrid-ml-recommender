# src/recommend.py

import os
import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------
# Globals: Lazy-loaded shared state
# ----------------------------------------------------------
_CB = None
_CF = None
_META = None
_TRACK_INDEX = None
_N = None


# ----------------------------------------------------------
# Internal Utilities
# ----------------------------------------------------------
def _set_N(n: int):
    global _N
    _N = int(n)


def _ensure_cb():
    global _CB
    if _CB is None:
        print("Loading Content-Based model...")
        _CB = joblib.load(os.path.join("models", "content_similarity.pkl"))
        _CB = np.asarray(_CB)
        if _CB.ndim != 2 or _CB.shape[0] != _CB.shape[1]:
            raise ValueError(f"CB matrix must be square. Got {_CB.shape}")
        _set_N(_CB.shape[0])
        _ensure_metadata()
    return _CB


def _ensure_cf():
    global _CF
    if _CF is None:
        print("Loading Collaborative Filtering model...")
        _CF = joblib.load(os.path.join("models", "item_similarity_cf_matrix.pkl"))
        _CF = np.asarray(_CF)
        if _CF.ndim != 2 or _CF.shape[0] != _CF.shape[1]:
            raise ValueError(f"CF matrix must be square. Got {_CF.shape}")
        _set_N(_CF.shape[0])
        _ensure_metadata()
    return _CF


def _ensure_metadata():
    global _META, _TRACK_INDEX, _N
    if _META is not None:
        return _META

    csv_path = os.path.join("data", "songs_metadata_for_api.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Metadata file missing: " + csv_path)

    meta = pd.read_csv(csv_path)

    # Ensure required columns
    for col in ["track_id", "track_name", "artist_name", "popularity"]:
        if col not in meta.columns:
            meta[col] = None

    target = _N if _N is not None else len(meta)

    # Align to model length
    if len(meta) < target:
        missing = target - len(meta)
        filler = pd.DataFrame({
            "track_id": [f"MISSING_{i}" for i in range(missing)],
            "track_name": ["Unknown"] * missing,
            "artist_name": ["Unknown Artist"] * missing,
            "popularity": [None] * missing
        })
        meta = pd.concat([meta, filler], ignore_index=True)
    elif len(meta) > target:
        meta = meta.iloc[:target].reset_index(drop=True)

    _META = meta[["track_id", "track_name", "artist_name", "popularity"]].copy()
    _TRACK_INDEX = {str(tid): i for i, tid in enumerate(_META["track_id"].astype(str))}
    _set_N(len(_META))
    return _META


def _idx_for_track(track_id: str):
    if not track_id:
        return None
    meta = _ensure_metadata()
    global _TRACK_INDEX
    if len(_TRACK_INDEX) != len(meta):
        _TRACK_INDEX = {str(tid): i for i, tid in enumerate(meta["track_id"].astype(str))}
    return _TRACK_INDEX.get(str(track_id))


def _safe_scores_row(mat: np.ndarray, idx: int):
    if mat is None or not isinstance(mat, np.ndarray):
        return np.zeros((0,), dtype=np.float32)
    if idx is None or idx < 0 or idx >= mat.shape[0]:
        return np.zeros((mat.shape[0],), dtype=np.float32)
    row = np.nan_to_num(mat[idx], nan=0.0, posinf=0.0, neginf=0.0)
    return np.ravel(row)


def _top_n(scores: np.ndarray, n: int, exclude_idx: int = None):
    if not isinstance(scores, np.ndarray) or scores.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    s = scores.copy()
    if exclude_idx is not None and 0 <= exclude_idx < len(s):
        s[exclude_idx] = -1.0
    order = np.argsort(s)[::-1]
    if n > 0:
        order = order[:n]
    return order, s[order]


def _build_results(indices, scores, method):
    meta = _ensure_metadata()
    out = []
    idxs = np.asarray(indices).tolist()
    scrs = np.asarray(scores).tolist()

    for idx, score in zip(idxs, scrs):
        if not (0 <= idx < len(meta)):
            continue

        row = meta.iloc[idx]
        pop = row.get("popularity")
        pop = None if pd.isna(pop) else int(pop)

        out.append({
            "track_id": str(row.get("track_id")),
            "track_name": str(row.get("track_name")),
            "artist_name": str(row.get("artist_name")),
            "popularity": pop,
            "score": float(score),  # ✅ fixes JSON serialization issue
            "method": method
        })

    return out


# ----------------------------------------------------------
# ✅ PUBLIC API - Content Based
# ----------------------------------------------------------
def recommend_cb(track_id: str, top_n: int = 10):
    idx = _idx_for_track(track_id)
    if idx is None:
        return None
    cb = _ensure_cb()
    scores = _safe_scores_row(cb, idx)
    inds, scrs = _top_n(scores, top_n, exclude_idx=idx)
    return _build_results(inds, scrs, "content")


# ----------------------------------------------------------
# ✅ PUBLIC API - Collaborative Filtering
# ----------------------------------------------------------
def recommend_cf(track_id: str, top_n: int = 10):
    idx = _idx_for_track(track_id)
    if idx is None:
        return None
    cf = _ensure_cf()
    scores = _safe_scores_row(cf, idx)
    inds, scrs = _top_n(scores, top_n, exclude_idx=idx)
    return _build_results(inds, scrs, "collab")


# ----------------------------------------------------------
# ✅ PUBLIC API - Hybrid (Safe fallback to content only)
# ----------------------------------------------------------
def recommend_hybrid(track_id: str, top_n: int = 10):
    idx = _idx_for_track(track_id)
    if idx is None:
        return None

    cb = _ensure_cb()
    cb_scores = _safe_scores_row(cb, idx)

    try:
        cf = _ensure_cf()
        if cf.shape != cb.shape:
            raise ValueError("Matrix shape mismatch")

        cf_scores = _safe_scores_row(cf, idx)
        hybrid = (0.4 * cb_scores) + (0.6 * cf_scores)

        inds, scrs = _top_n(hybrid, top_n, exclude_idx=idx)
        return _build_results(inds, scrs, "hybrid_0.4cb_0.6cf")

    except Exception:
        inds, scrs = _top_n(cb_scores, top_n, exclude_idx=idx)
        return _build_results(inds, scrs, "hybrid_fallback_content")


# ----------------------------------------------------------
# ✅ PUBLIC API - Track listing
# ----------------------------------------------------------
def fetch_metadata(limit: int = 20):
    meta = _ensure_metadata()
    return meta.head(limit).to_dict(orient="records")
