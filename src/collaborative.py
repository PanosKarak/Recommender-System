import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import pickle
from sklearn.metrics import ndcg_score

# 1. Load train/test
train_path = "../data/ml-32m/train.csv"
test_path = "../data/ml-32m/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f" Train loaded: {train_df.shape}")
print(f" Test loaded: {test_df.shape}")

# 2. Map IDs to indices
user_ids = train_df['userId'].unique()
item_ids = train_df['movieId'].unique()

user_map = {uid: i for i, uid in enumerate(user_ids)}
item_map = {iid: i for i, iid in enumerate(item_ids)}

user_map_rev = {v: k for k, v in user_map.items()}
item_map_rev = {v: k for k, v in item_map.items()}

# 3. Create interaction matrix
rows = train_df['userId'].map(user_map)
cols = train_df['movieId'].map(item_map)
data = train_df['rating']

interaction_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
print(f" Interaction matrix shape: {interaction_matrix.shape}")

# 4️. Train ALS model or load saved
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # prevent multithreading issues
MODEL_PATH = "../data/ml-32m/als_model_new.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        als = pickle.load(f)
    print(" ALS model loaded from disk")
else:
    als = AlternatingLeastSquares(
        factors=50,
        regularization=0.1,
        iterations=20,
        random_state=42
    )
    als.fit(interaction_matrix.tocsr())
    print("✅ ALS model trained")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(als, f)
    print("💾 ALS model saved to disk")

# 5. Load movies metadata
movies_path = '../data/ml-32m/movies.csv'
movies_df = pd.read_csv(movies_path)

# 6. Top-N recommendations
def recommend_top_n(model, train_matrix, user_id, item_df, N=10):
    if user_id not in user_map:
        raise ValueError(f"User {user_id} not in training set")

    user_idx = user_map[user_id]
    recs, scores = model.recommend(user_idx, train_matrix.tocsr()[user_idx], N=N)
    recs_df = pd.DataFrame({'item_idx': recs.astype(int), 'score': scores})
    recs_df['movieId'] = recs_df['item_idx'].map(item_map_rev)
    recs_df = recs_df.merge(item_df, on='movieId', how='left')
    return recs_df[['movieId', 'title', 'score']].sort_values('score', ascending=False)

# 7. Example usage + evaluation
if __name__ == "__main__":
    # 1. Print top-10 for a random user
    random_user = np.random.choice(user_ids)
    print(f"\n📌 Top-10 recommendations for user {random_user}:")
    top10 = recommend_top_n(als, interaction_matrix, random_user, movies_df, N=10)
    print(top10)

    # 2. Evaluate metrics for 300 users from test set
    sample_users = test_df['userId'].drop_duplicates().sample(300, random_state=42)

    k = 10
    precision_scores = []
    recall_scores = []
    map_scores = []
    ndcg_scores = []

    for uid in sample_users:
        user_test = test_df[test_df['userId'] == uid]
        relevant = set(user_test[user_test['rating'] >= 4]['movieId'])
        if not relevant:
            continue

        recs_df = recommend_top_n(als, interaction_matrix, uid, movies_df, N=k)
        recommended_items = recs_df['movieId'].tolist()

        # Precision & Recall
        hits = len(set(recommended_items) & relevant)
        precision_scores.append(hits / k)
        recall_scores.append(hits / len(relevant))

        # MAP@K
        ap_sum = 0
        hits_count = 0
        for idx, item in enumerate(recommended_items, start=1):
            if item in relevant:
                hits_count += 1
                ap_sum += hits_count / idx
        map_scores.append(ap_sum / min(len(relevant), k))

        # NDCG@K
        true_relevance = [1 if m in relevant else 0 for m in recommended_items]
        predicted_scores = recs_df['score'].tolist()

        ndcg_scores.append(ndcg_score([true_relevance], [predicted_scores]))

    print("\n✅ Evaluation metrics (Collaborative):")
    print(f"Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Recall@{k}: {np.mean(recall_scores):.4f}")
    print(f"MAP@{k}: {np.mean(map_scores):.4f}")
    print(f"NDCG@{k}: {np.mean(ndcg_scores):.4f}")
