import joblib
import pandas as pd

CF_MODEL_PATH = "models/item_similarity_cf_matrix.pkl"
META_PATH = "data/songs_metadata_for_api.csv"

print("🔄 Loading Collaborative Filtering model...")

try:
    cf = joblib.load(CF_MODEL_PATH)
    print("✅ CF matrix loaded successfully!")
except Exception as e:
    print("❌ ERROR loading CF model:", e)
    raise e


# ✅ Load metadata
try:
    meta = pd.read_csv(META_PATH)
    print("✅ Metadata loaded:", meta.shape)
except Exception as e:
    print("❌ ERROR loading metadata:", e)
    raise e


# ✅ Ensure CF is a DataFrame
if isinstance(cf, pd.DataFrame):
    cf_matrix = cf
else:
    # Assume square numpy array
    cf_matrix = pd.DataFrame(cf)

# ✅ Align metadata to CF shape
cf_size = cf_matrix.shape[0]
meta = meta.head(cf_size)

# ✅ Set track IDs as index & columns
cf_matrix.index = meta["track_id"]
cf_matrix.columns = meta["track_id"]


def recommend_cf(track_id: str, top_n: int = 10):
    if track_id not in cf_matrix.index:
        return {"error": f"Track ID not found in CF model: {track_id}"}

    sims = cf_matrix.loc[track_id]
    top = sims.sort_values(ascending=False)[1: top_n + 1]

    results = []
    for tid, score in top.items():
        row = meta[meta["track_id"] == tid].iloc[0]
        results.append({
            "track_id": tid,
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            "popularity": int(row["popularity"]),
            "score": float(score)
        })

    return results


def get_cf_ids(limit: int = 20):
    return meta["track_id"].head(limit).tolist()
