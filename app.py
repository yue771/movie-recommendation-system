import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")
st.write("A movie recommender built with collaborative filtering, content-based filtering, and hybrid recommendation.")

# =========================
# 1. Load data
# =========================
@st.cache_data
def load_data():
    links = pd.read_csv('/kaggle/input/datasets/rounakbanik/the-movies-dataset/links_small.csv')
    movies = pd.read_csv('/kaggle/input/datasets/rounakbanik/the-movies-dataset/movies_metadata.csv')
    ratings = pd.read_csv('/kaggle/input/datasets/rounakbanik/the-movies-dataset/ratings_small.csv')

    movies = movies[['id', 'title', 'genres', 'overview']].copy()
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)
    movies['overview'] = movies['overview'].fillna('')
    movies = movies.drop_duplicates(subset='id')

    links = links[['movieId', 'tmdbId']].copy()
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)

    return links, movies, ratings


links, movies_full, ratings = load_data()

# =========================
# 2. Build collaborative part
# =========================
@st.cache_data
def build_collaborative_model(links, movies_full, ratings):
    ratings_links = ratings.merge(links, on='movieId', how='inner')
    df_cf = ratings_links.merge(movies_full, left_on='tmdbId', right_on='id', how='inner')
    df_cf = df_cf[['userId', 'movieId', 'rating', 'title']]

    user_movie_matrix = df_cf.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    )

    movie_rating_counts = df_cf['title'].value_counts()
    popular_movies = movie_rating_counts[movie_rating_counts >= 20].index

    filtered_matrix = user_movie_matrix[popular_movies]
    movie_user_matrix = filtered_matrix.T.fillna(0)

    movie_similarity = cosine_similarity(movie_user_matrix)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    return df_cf, movie_similarity_df


df_cf, movie_similarity_df = build_collaborative_model(links, movies_full, ratings)

# =========================
# 3. Build content part
# =========================
def parse_genres(x):
    try:
        genres_list = ast.literal_eval(x)
        return ' '.join([i['name'] for i in genres_list])
    except:
        return ''

@st.cache_data
def build_content_model(movies_full):
    movies_cb = movies_full[['id', 'title', 'genres', 'overview']].copy()
    movies_cb['genres_text'] = movies_cb['genres'].apply(parse_genres)
    movies_cb['content'] = movies_cb['genres_text'] + ' ' + movies_cb['overview']

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_cb['content'])

    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    content_similarity_df = pd.DataFrame(
        content_similarity,
        index=movies_cb['title'],
        columns=movies_cb['title']
    )

    return movies_cb, content_similarity_df


movies_cb, content_similarity_df = build_content_model(movies_full)

# =========================
# 4. Recommendation functions
# =========================
def recommend_movies(movie_title, top_n=10):
    if movie_title not in movie_similarity_df.columns:
        return pd.DataFrame(columns=['Movie', 'Collaborative Score'])

    sim_scores = movie_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]
    return pd.DataFrame({
        'Movie': sim_scores.index,
        'Collaborative Score': sim_scores.values
    })

def recommend_by_content(movie_title, top_n=10):
    if movie_title not in content_similarity_df.columns:
        return pd.DataFrame(columns=['Movie', 'Content Score'])

    sim_scores = content_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]
    return pd.DataFrame({
        'Movie': sim_scores.index,
        'Content Score': sim_scores.values
    })

def recommend_hybrid(movie_title, top_n=10, alpha=0.5):
    cf_df = recommend_movies(movie_title, top_n=50)
    cb_df = recommend_by_content(movie_title, top_n=50)

    if cf_df.empty and cb_df.empty:
        return pd.DataFrame(columns=['Movie', 'Collaborative Score', 'Content Score', 'Hybrid Score'])

    merged = pd.merge(cf_df, cb_df, on='Movie', how='outer').fillna(0)
    merged['Hybrid Score'] = alpha * merged['Collaborative Score'] + (1 - alpha) * merged['Content Score']
    merged = merged.sort_values(by='Hybrid Score', ascending=False).head(top_n)

    return merged

# =========================
# 5. UI
# =========================
st.sidebar.header("⚙️ Settings")

movie_list = sorted(list(set(movies_cb['title'].dropna())))
selected_movie = st.sidebar.selectbox("Choose a movie", movie_list)

method = st.sidebar.radio(
    "Recommendation Method",
    ["Collaborative Filtering", "Content-Based", "Hybrid"]
)

top_n = st.sidebar.slider("Top N Recommendations", min_value=5, max_value=20, value=10)

alpha = 0.5
if method == "Hybrid":
    alpha = st.sidebar.slider("Alpha (Collaborative Weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.subheader(f"Selected Movie: {selected_movie}")
st.write(f"Method: **{method}**")

if st.button("Get Recommendations"):
    if method == "Collaborative Filtering":
        result = recommend_movies(selected_movie, top_n=top_n)
        st.dataframe(result, use_container_width=True)

    elif method == "Content-Based":
        result = recommend_by_content(selected_movie, top_n=top_n)
        st.dataframe(result, use_container_width=True)

    else:
        result = recommend_hybrid(selected_movie, top_n=top_n, alpha=alpha)
        st.dataframe(result, use_container_width=True)

st.markdown("---")
st.markdown("### 📌 About This App")
st.write("""
This app demonstrates three movie recommendation strategies:

- Collaborative Filtering
- Content-Based Recommendation
- Hybrid Recommendation
""")