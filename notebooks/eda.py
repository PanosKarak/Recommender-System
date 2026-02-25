import pandas as pd
import matplotlib.pyplot as plt

# Φόρτωση ratings και movies
ratings = pd.read_csv('../data/ml-32m/ratings.csv')
movies = pd.read_csv('../data/ml-32m/movies.csv')

print("Φόρτωση βαθμολογιών:", ratings.shape)
print("Φόρτωση ταινιών:", movies.shape)

# συνένωση ratings και movies στο movieId ώστε να φαίνεται και ο τίτλος της ταινίας
data = ratings.merge(movies[['movieId', 'title']], on='movieId')
print("\nΣυνένωση δεδομένων:", data.shape)

# Βασικά στατιστικά
print("\nΔείγμα πρώτων γραμμών:")
print(data.head())

print("\nΜέση Βαθμολογία:", round(data['rating'].mean(), 2)) #με δύο δεκαδικά ψηφία
print("Αριθμός Μοναδικών Χρηστών:", data['userId'].nunique())
print("Αριθμός Μοναδικών Ταινιών:", data['movieId'].nunique())

# Ratings ανα χρήστη
ratings_per_user = data.groupby('userId')['rating'].count()
print("\nΑξιολογήσεις ανα χρήστη:")
print(ratings_per_user.describe()) #μέσο όρο, διάμεσο, min, max

# Ratings ανα ταινία
ratings_per_movie = data.groupby('movieId')['rating'].count()
print("\nRatings per movie : ")
print(ratings_per_movie.describe()) #μέσο όρο, διάμεσο, min, max


#  Άθροιση ελλείψεων και διπλότυπων
print("\nΕλλιπείς τιμές ανα στήλη : ")
print(data.isna().sum())

print("\nΔιπλότυπες τιμές : ", data.duplicated().sum())

# Ποσοστό κενών κελιών στον πίνακα χρηστών-ταινιών
num_unique_users = data['userId'].nunique() #μοναδικοί χρήστες
num_unique_items = data['movieId'].nunique() #μοναδικές ταινίες
num_interactions = len(data) #πόσα ratings έχει συνολικά
sparsity = 1 - (num_interactions / (num_unique_users * num_unique_items)) #ποσοστό του πίνακα χωρίς βαθμολογίες
print("\nMatrix sparsity:", round(sparsity, 6))


# Οπτικοποίηση

# Γράφημα για ratings (συχνότητα εμφάνισης κάθε βαθμολογίας 1-5)
plt.figure(figsize=(6,4))
plt.hist(data['rating'], bins=10, edgecolor='black')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Οι 20 πιο πολυαξιολογημένες ταινίες
top_movies = data['title'].value_counts().head(20)
plt.figure(figsize=(12,5))
top_movies.plot(kind='bar')
plt.title("Top 20 Most Rated Movies")
plt.ylabel("Number of Ratings")
plt.show()

# Διανομή αριθμού βαθμολογιών ανά χρήστη σε δείγμα
sample_users = ratings_per_user.sample(min(10000, len(ratings_per_user)), random_state=42)
plt.figure(figsize=(6,4))
plt.hist(sample_users, bins=30, edgecolor='black')
plt.title("Number of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Count")
plt.show()

# Διανομή αριθμού βαθμολογιών ανά ταινία σε δείγμα
sample_movies = ratings_per_movie.sample(min(10000, len(ratings_per_movie)), random_state=42)
plt.figure(figsize=(6,4))
plt.hist(sample_movies, bins=30, edgecolor='black')
plt.title("Number of Ratings per Movie")
plt.xlabel("Number of Ratings")
plt.ylabel("Count")
plt.show()


# Heatmap

# Δείγμα 5000 για αποφυγή υπερφόρτωσης της μνήμης
# Δειγματοληψία δεδομένων
sample_data = data.sample(20000, random_state=42)

# Pivot πίνακας
sample_matrix = sample_data.pivot(index='userId', columns='movieId', values='rating')

# Μετατροπή σε sparse format
sparse_matrix = sample_matrix.notna()

plt.figure(figsize=(12, 6))

# spy plot – δείχνει μόνο πού υπάρχει rating
plt.spy(sparse_matrix, markersize=2, aspect='auto')

plt.title("Sparse User-Movie Matrix (Ratings Positions)")
plt.xlabel("Movie Index")
plt.ylabel("User Index")

plt.show()

