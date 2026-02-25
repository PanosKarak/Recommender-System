import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import ndcg_score



topRecommendations = 10  # Αριθμός ταινιών που θα προταθούν
evaluation_sample_size = 300  # δείγμα το οποίο θα χρησιμοποιηθεί για την αξιολόγηση

# φόρτωση των αρχείων που θα χρησιμοποιήσουμε
moviesPath = '../data/ml-32m/movies.csv'
trainPath = '../data/ml-32m/train.csv'
testPath = '../data/ml-32m/test.csv'

movies = pd.read_csv(moviesPath)
trainRatings = pd.read_csv(trainPath)
testRatings = pd.read_csv(testPath)

# Στο movies, αν υπάρχουν κενά βάλε κενό και όπου υπάρχει |, βάλε κενό
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ', regex=False)

#prints οτι όλα φορτώθηκαν οκ
print(f"Movies loaded: {movies.shape}")
print(f"Train ratings: {trainRatings.shape}")
print(f"Test ratings: {testRatings.shape}")
print(movies.head())


# Μετατροπή των genres σε αριθμητικά διανύσματα
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
#print ότι όλα πήγαν καλά
print(f" TF-IDF matrix created with shape {tfidf_matrix.shape}")

# function για προτάσεις
def recommended_content_for_user(user_id, top_k=10):

    # Βρες απο το train set ποιές ταινίες έχει δεί και αξιολογήσει ο χρήστης μας
    user_seen_movie_ids = trainRatings[trainRatings['userId'] == user_id]['movieId'].tolist()
    if not user_seen_movie_ids: # σφάλμα αν ο χρήστης δεν έχει ιστορικό
        raise ValueError(f"User {user_id} has no ratings in train set.")

    # Άν υπάρχει στο function μας attribute με τίτλο printed_example
    if hasattr(recommended_content_for_user, 'printed_example'):
        print_example = False #αν έχουμε τυπώσει ήδη κάτι είναι false
    else:
        print_example = True #αν δεν έχουμε τυπώσει είναι true
        recommended_content_for_user.printed_example = True
        #την επόμενη μην ξανατυπώσεις, γιατί έχουμε ήδη τυπώσει

    if print_example: # αν = true
        seen_movies = movies[movies['movieId'].isin(user_seen_movie_ids)]
        print(f"\n User {user_id} has seen these movies:")
        print(seen_movies[['movieId', 'title', 'genres']])

    # απο τις ταινίες που έχει δει,
    seen_indexes = movies[movies['movieId'].isin(user_seen_movie_ids)].index
    sim_scores = linear_kernel(tfidf_matrix[seen_indexes], tfidf_matrix)
    #linear kernel = cosine similarity για TF-IDF με κανονικοποιημένο vector
    sim_scores_sum = sim_scores.mean(axis=0) #cosine similarity


    recommended_indices = sim_scores_sum.argsort()[::-1]
    #ταξινόμηση με βάση το cos_sim, απο μεγάλη σε μικρή
    recommended_indices = [i for i in recommended_indices if movies.loc[i, 'movieId']
                           not in user_seen_movie_ids]
    #αποκλείουμε ήδη προβεβλημένες ταινίες


    recommendations = movies.iloc[recommended_indices[:top_k]].copy()
    #επιλέγουμε τις τop ταινίες με βάση τη similarity
    recommendations['cos_sim'] = sim_scores_sum[recommended_indices[:top_k]]
    #προσθέτουμε στήλη cos_sim για το similarity των προτάσεων με το ιστορικό του χρήστη

    if print_example: #= true
        print(f"\nTop-{top_k} Content-Based Recommendations for user {user_id}:")
        print(recommendations[['movieId', 'title', 'genres', 'cos_sim']])

    return recommendations

# Μετρικές αξιολόγησης
def evaluate_metrics_on_users(sample_users, best=topRecommendations):
    precision_scores = []
    recall_scores = []
    map_scores = []
    ndcg_scores = []

    for uid in sample_users:
        # θέτουμε ως relevant, για τον τρέχοντα χρήστη στο test, ταινίες με rating>4
        relevant = set(testRatings[(testRatings['userId'] == uid) &
                                   (testRatings['rating'] >= 4)]['movieId'].unique())
        #αν δεν υπάρχει relevant προχώρησε σε επόμενο χρήστη
        if not relevant:
            continue

        # παίρνουμε τις προτάσεις για τον ίδιο χρήστη με πάνω
        recs_df = recommended_content_for_user(uid, top_k=best)
        recommended_items = recs_df['movieId'].tolist()

        # Precision K = πόσο καλές είναι οι προτάσεις recall = πόσο πλήρεις είναι οι προτάσεις
        hits = len(set(recommended_items) & relevant)
        # προτεινόμενες ταινίες που είναι και relevant(με rating>4)
        precision = hits / best
        # ποσοστό προτεινόμενων ταινιών που είναι και relevant δλδ rating>4
        recall = hits / len(relevant)
        #ποσοστό των relevant(με rating>4) και εμείς τις προβλέψαμε

        # MAP K
        average_precision_sum = 0 # αρχικοποίηση
        hits_count = 0 # αρχικοποίηση
        for idx, item in enumerate(recommended_items, start=1):
            # απο τις προτεινόμενες παίρνουμε τη θέση και το id
            if item in relevant:
                # αν το rating>4, το movieId, δηλαδή η ταινία είναι σωστή για τον χρήστη
                hits_count += 1 # + μία σωστή ταινία
                average_precision_sum += hits_count / idx
                # relevant ταινίες με index που είναι νωρίς στη λίστα
        map_k = average_precision_sum / min(len(relevant), best)
        # πόσο καλά μπορεί η λίστα μας να εμφανίζει πιο πάνω τις relevant

        # NDCG K == αν relevant είναι στην κορυφή της λίστας τότε υψηλό σκορ
        true_relevance = [1 if movie in relevant else 0 for movie in recommended_items]
        #ποιές είναι relevant απο αυτές που προτάθηκαν
        ndcg = ndcg_score([true_relevance], [list(range(best, 0, -1))])
        # οι πρώτες θέσεις της λίστα των προτάσεων έχουν μεγαλύτερο βάρος

        precision_scores.append(precision)
        recall_scores.append(recall)
        map_scores.append(map_k)
        ndcg_scores.append(ndcg)

    return { #μέσος όρος για όλους τους χρήστες
        "Precision@K": np.mean(precision_scores),
        "Recall@K": np.mean(recall_scores),
        "MAP@K": np.mean(map_scores),
        "NDCG K": np.mean(ndcg_scores)
    }

if __name__ == "__main__":
    #τρέξε μόνο εδω σαν αυτούσιο πρόγραμμα, αν γίνεται import αλλού μην τρέχεις
    random_user = np.random.choice(trainRatings['userId'].unique())
    # απο το train, πάρε έναν μοναδικό χρήστη
    recommended_content_for_user(random_user, top_k=topRecommendations)
    # δώσε 10 προτάσεις για αυτόν το χρήστη

    print("\nEvaluating on sample users...") #πάρε απο το test κάποιο δείγμα χρηστών
    sample_users = testRatings['userId'].drop_duplicates().sample(
        min(evaluation_sample_size, len(testRatings)), random_state=42)
    #42 είναι το seed που μου εξασφαλίζει οτι παίρνω σταθερό δείγμα
    metrics = evaluate_metrics_on_users(sample_users)
    #πάνω τους πάρε μετρικές αξιολόγησης(κάλεσε τη συνάρτηση)
    print("\nEvaluation metrics (Content-Based):")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
