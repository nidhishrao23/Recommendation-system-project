#RECOMMENDATION SYSTEM PROJECT

reco_project/
├─ data/
│  ├─ u.data                # MovieLens 100k ratings (tab-separated)
│  └─ u.item                # Movie metadata (pipe-separated)
├─ src/
│  ├─ data_utils.py         # data loading & preprocessing
│  ├─ cf_model.py           # collaborative filtering (train/predict)
│  ├─ content_model.py      # content-based model (genres TF-IDF)
│  ├─ hybrid.py             # blending logic and ranking
│  ├─ train.py              # orchestrates training, saves models
│  ├─ recommend.py          # example CLI usage to get top-k
│  └─ app.py                # Flask API exposing recommend endpoint
├─ models/
│  ├─ svd_model.pkl
│  └─ tfidf_vectorizer.pkl
├─ requirements.txt
├─ README.md
└─ statement.md
numpy
pandas
scikit-learn
scipy
flask
joblib
surprise
gunicorn
import pandas as pd
from sklearn.model_selection import train_test_split

def load_ratings(path="data/u.data"):
    cols = ["user_id","item_id","rating","timestamp"]
    df = pd.read_csv(path, sep="\t", names=cols, engine="python")
    return df

def load_items(path="data/u.item"):
    # MovieLens u.item has pipe-separated fields; last 19 columns are genres flags
    cols = [
        "item_id","title","release_date","video_release_date","imdb_url"
    ] + [f"genre_{i}" for i in range(19)]
    df = pd.read_csv(path, sep="|", names=cols, encoding='latin-1', engine="python")
    # For content-based, create a 'genres' string
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    def genres_from_flags(row):
        # There is no genre names here, so we map indices to generic tags
        tags = [f"g{i}" for i, c in enumerate(genre_cols) if row[c] == 1]
        return " ".join(tags)
    df["genres"] = df.apply(genres_from_flags, axis=1)
    return df[["item_id","title","genres"]]

def train_test_split_ratings(df, test_size=0.2, random_state=42):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test
import joblib
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

MODEL_PATH = "models/svd_model.pkl"

def train_svd(train_df, n_factors=50, n_epochs=20, random_state=42):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(train_df[["user_id","item_id","rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
    algo.fit(trainset)
    joblib.dump(algo, MODEL_PATH)
    return algo

def load_svd(path=MODEL_PATH):
    return joblib.load(path)

def evaluate_svd(algo, test_df):
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(test_df[["user_id","item_id","rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    return rmse

def predict_user_item(algo, user_id, item_id):
    pred = algo.predict(str(user_id), str(item_id))
    return pred.est
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

VEC_PATH = "models/tfidf_vectorizer.pkl"
MATRIX_PATH = "models/tfidf_matrix.npy"

class ContentModel:
    def __init__(self, items_df=None):
        self.items_df = items_df
        self.vectorizer = None
        self.tfidf_matrix = None

    def train(self):
        corpus = self.items_df["genres"].fillna("").astype(str).tolist()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        joblib.dump(self.vectorizer, VEC_PATH)
        # Save sparse matrix as npz via joblib
        joblib.dump(self.tfidf_matrix, MATRIX_PATH)
        return self.tfidf_matrix

    def load(self):
        self.vectorizer = joblib.load(VEC_PATH)
        self.tfidf_matrix = joblib.load(MATRIX_PATH)

    def top_k_similar(self, item_id, k=10):
        idx = self.items_df[self.items_df.item_id==item_id].index[0]
        vec = self.tfidf_matrix[idx]
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_idx = (-sims).argsort()[1:k+1]
        return self.items_df.iloc[top_idx][["item_id","title"]].assign(score=sims[top_idx])
import numpy as np
import pandas as pd
from src.cf_model import load_svd, predict_user_item
from src.content_model import ContentModel

def score_candidates(user_id, candidate_items, svd_model, content_model, items_df,
                     alpha=0.7):
    """
    alpha: weight for CF; (1-alpha) for content
    candidate_items: list of item_ids
    returns DataFrame with item_id, title, blended_score
    """
    cf_scores = []
    for iid in candidate_items:
        try:
            cf = svd_model.predict(str(user_id), str(iid)).est
        except Exception:
            cf = 3.0
        cf_scores.append(cf)
    # Normalize CF scores
    cf_arr = np.array(cf_scores)
    if cf_arr.max() - cf_arr.min() > 0:
        cf_norm = (cf_arr - cf_arr.min()) / (cf_arr.max() - cf_arr.min())
    else:
        cf_norm = cf_arr / 5.0

    # Content similarity score relative to a "user profile" can be approximated by
    # averaging content vectors of items the user rated highly. But for simplicity,
    # use average similarity to all candidate items: use item-to-item sim with content_model.
    # Here we compute content_score as 0.5 (placeholder) or compute real cosine similarity if available.
    # For simplicity in this code, set content as zeros and rely mostly on CF; but you can
    # compute content similarities using content_model.tfidf_matrix.
    content_scores = []
    # compute content similarity to user's top items
    # (Assumes items_df index aligned with content_model.items_df)
    for iid in candidate_items:
        content_scores.append(0.0)
    content_arr = np.array(content_scores)
    # normalize
    if content_arr.max() - content_arr.min() > 0:
        content_norm = (content_arr - content_arr.min()) / (content_arr.max() - content_arr.min())
    else:
        content_norm = content_arr

    blended = alpha * cf_norm + (1-alpha) * content_norm
    out = pd.DataFrame({
        "item_id": candidate_items,
        "cf_score": cf_arr,
        "blend_score": blended
    })
    merged = out.merge(items_df[["item_id","title"]], on="item_id", how="left")
    return merged.sort_values("blend_score", ascending=False)
import os
import joblib
from src.data_utils import load_ratings, load_items, train_test_split_ratings
from src.cf_model import train_svd
from src.content_model import ContentModel

os.makedirs("models", exist_ok=True)

def main():
    ratings = load_ratings("data/u.data")
    items = load_items("data/u.item")
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    print("Training CF SVD...")
    svd = train_svd(train, n_factors=50, n_epochs=30)
    print("Training content model...")
    cm = ContentModel(items_df=items.reset_index(drop=True))
    cm.train()
    print("Saved models to models/ directory.")
    # simple evaluation
    try:
        from src.cf_model import evaluate_svd
        rmse = evaluate_svd(svd, test)
        print(f"CF RMSE: {rmse:.4f}")
    except Exception as e:
        print("Evaluation failed:", e)

if __name__ == "__main__":
    main()
import pandas as pd
from src.cf_model import load_svd
from src.content_model import ContentModel
from src.hybrid import score_candidates

def recommend_top_k(user_id, k=10, alpha=0.8):
    items = pd.read_csv("data/u.item", sep="|", names=None, encoding='latin-1', engine='python')
    items = items.iloc[:,0:2]
    items.columns = ["item_id","title"]
    svd = load_svd()
    cm = ContentModel()
    cm.items_df = pd.read_csv("data/u.item", sep="|", names=None, encoding='latin-1', engine='python').iloc[:,0:2]
    # candidate items: all items (filter items user already rated for production)
    candidate_items = items["item_id"].tolist()
    scored = score_candidates(user_id, candidate_items, svd, cm, items, alpha=alpha)
    return scored.head(k)

if __name__ == "__main__":
    import sys
    uid = int(sys.argv[1]) if len(sys.argv)>1 else 1
    print(recommend_top_k(uid, k=10).to_string(index=False))
from flask import Flask, request, jsonify
import pandas as pd
from src.cf_model import load_svd
from src.content_model import ContentModel
from src.hybrid import score_candidates

app = Flask(__name__)

items = pd.read_csv("data/u.item", sep="|", names=None, encoding='latin-1', engine='python').iloc[:,0:2]
items.columns = ["item_id","title"]
svd = None
cm = None

@app.before_first_request
def load_models():
    global svd, cm
    svd = load_svd()
    cm = ContentModel()
    cm.items_df = pd.read_csv("data/u.item", sep="|", names=None, encoding='latin-1', engine='python').iloc[:,0:2]
    cm.load()

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", default=1, type=int)
    k = request.args.get("k", default=10, type=int)
    alpha = float(request.args.get("alpha", 0.8))
    candidate_items = items["item_id"].tolist()
    scored = score_candidates(user_id, candidate_items, svd, cm, items, alpha=alpha)
    topk = scored.head(k)
    data = topk[["item_id","title","blend_score"]].to_dict(orient="records")
    return jsonify({"user_id": user_id, "recommendations": data})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
def build_user_profile(user_id, ratings_df, items_df, tfidf_matrix, min_rating=4.0):
    # get item indices the user rated >= min_rating
    liked = ratings_df[(ratings_df.user_id==user_id) & (ratings_df.rating>=min_rating)]
    if liked.empty:
        # fallback to global mean vector
        profile = tfidf_matrix.mean(axis=0)
        return profile
    indices = [items_df[items_df.item_id==iid].index[0] for iid in liked.item_id]
    profile = tfidf_matrix[indices].mean(axis=0)
    return profile