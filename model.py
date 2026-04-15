import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD DATA ----------------

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')

# Select required columns
movies = movies[['id','title','overview','genres','keywords','cast','crew']]

# Rename id → movie_id (clean way)
movies = movies.rename(columns={'id': 'movie_id'})

# Remove missing values
movies.dropna(inplace=True)

# ---------------- JSON CLEANING ----------------

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# ---------------- CAST ----------------

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# ---------------- CREW ----------------

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# ---------------- TEXT PROCESSING ----------------

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# remove spaces (e.g., "Sam Worthington" → "SamWorthington")
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# ---------------- CREATE TAGS ----------------

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# convert list → string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# final dataframe
new_df = movies[['movie_id','title','tags']]

# lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# ---------------- VECTORIZATION ----------------

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# ---------------- SIMILARITY ----------------

similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND FUNCTION ----------------

def recommend(movie):
    if movie not in new_df['title'].values:
        print("❌ Movie not found!")
        return

    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    
    print(f"\n🎬 Recommended movies for '{movie}':\n")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# ---------------- TEST ----------------

recommend("Avatar")

# ---------------- SAVE CLEAN DATA ----------------

new_df.to_csv("processed_movies.csv", index=False)