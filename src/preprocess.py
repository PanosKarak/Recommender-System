import pandas as pd
from sklearn.model_selection import train_test_split

# Load Ratings and Movies
ratings_path = '../data/ml-32m/ratings.csv'
movies_path = '../data/ml-32m/movies.csv'

# load ratings
ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_csv(ratings_path, usecols=ratings_cols)

# convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
print("Ratings loaded:", ratings.shape)

# load movies
movies = pd.read_csv(movies_path)

# split genres into list
if 'genres' in movies.columns:
    movies['genres'] = movies['genres'].str.split('|')

print("Movies loaded:", movies.shape)

# Merge Ratings with Movie Info
data = ratings.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
print("Ratings merged with movies:", data.shape)

# Filter Sparse Users / Movies (fixed)
min_ratings_user = 20
min_ratings_movie = 10

# get users with enough ratings
users_to_keep = data['userId'].value_counts()[data['userId'].value_counts() >= min_ratings_user].index
# get movies with enough ratings
movies_to_keep = data['movieId'].value_counts()[data['movieId'].value_counts() >= min_ratings_movie].index

# filter and make a copy to avoid SettingWithCopyWarning
data_filtered = data[data['userId'].isin(users_to_keep) & data['movieId'].isin(movies_to_keep)].copy()
print("\nData after filtering sparse users/movies:", data_filtered.shape)

# Encode User and Movie IDs
user2idx = {user: idx for idx, user in enumerate(data_filtered['userId'].unique())}
movie2idx = {movie: idx for idx, movie in enumerate(data_filtered['movieId'].unique())}

data_filtered['user_idx'] = data_filtered['userId'].map(user2idx)
data_filtered['movie_idx'] = data_filtered['movieId'].map(movie2idx)
print("\nUser and Movie IDs encoded")

# Split into Train / Test
train, test = train_test_split(data_filtered, test_size=0.2, random_state=42)
print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)

# Save Preprocessed Data
train.to_csv('../data/ml-32m/train.csv', index=False)
test.to_csv('../data/ml-32m/test.csv', index=False)
print("\nPreprocessed data saved to CSV")