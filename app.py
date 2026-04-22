import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("A simple hybrid-style movie recommendation demo based on collaborative filtering and content-based filtering.")

# =========================
# 1. Load data
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_small.csv")
    ratings = pd.read_csv("ratings_small.csv")
    links = pd.read_csv("links_small.csv")
    return movies, ratings, links

movies, ratings, links = load_data()

# =========================
# 2. Prepare content-based data
# =========================
@st.cache_data
def prepare_content_data(movies_df):
    movies_df = movies_df.copy()

    movies_df["overview"] = movies_df["overview"].fillna("")
    movies_df["genres"] = movies_df["genres"].fillna("")

    def parse_genres(x):
        try:
            genres_list = ast.literal_eval(x)
            return " ".join([i["name"] for i in genres_list])
        except:
            return ""

    movies_df["genres_text"] = movies_df["genres"].apply(parse_genres)
    movies_df["content"] = movies_df["genres_text"] + " " + movies_df["overview"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df["content"])

    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    content_similarity_df = pd.DataFrame(
        content_similarity,
        index=movies_df["title"],
        columns=movies_df["title"]
    )

    return movies_df, content_similarity_df

movies_content, content_similarity_df = prepare_content_data(movies)

# =========================
# 3. Prepare collaborative filtering data
# =========================
@st.cache_data
def prepare_cf_data(ratings_df, movies_df):
    # 这里只保留 title 和一个 movieId 映射
    if "id" in movies_df.columns:
        movie_map = movies_df[["id", "title"]].copy()
        movie_map["id"] = pd.to_numeric(movie_map["id"], errors="coerce")
        movie_map = movie_map.dropna(subset=["id"])
        movie_map["id"] = movie_map["id"].astype(int)
        movie_map = movie_map.drop_duplicates(subset=["id"])
    else:
        movie_map = movies_df[["title"]].copy()
        movie_map["id"] = range(len(movie_map))

    # links_small 里 movieId -> tmdbId
    links_df = links_df = links.copy()
    links_df["tmdbId"] = pd.to_numeric(links_df["tmdbId"], errors="coerce")
    links_df = links_df.dropna(subset=["tmdbId"])
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)

    ratings_links = ratings_df.merge(links_df[["movieId", "tmdbId"]], on="movieId", how="inner")
    df = ratings_links.merge(movie_map, left_on="tmdbId", right_on="id", how="inner")

    df = df[["userId", "movieId", "rating", "title"]].copy()

    user_movie_matrix = df.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    )

    movie_rating_counts = df["title"].value_counts()
    popular_movies = movie_rating_counts[movie_rating_counts >= 20].index

    filtered_matrix = user_movie_matrix[popular_movies]
    movie_user_matrix = filtered_matrix.T.fillna(0)

    movie_similarity = cosine_similarity(movie_user_matrix)

    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    return df, movie_similarity_df, movie_rating_counts

cf_df, movie_similarity_df, movie_rating_counts = prepare_cf_data(ratings, movies)

# =========================
# 4. Recommendation functions
# =========================
def recommend_by_content(movie_title, top_n=10):
    if movie_title not in content_similarity_df.columns:
        return pd.DataFrame(columns=["Movie", "Content Score"])

    sim_scores = content_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]

    recommendations = pd.DataFrame({
        "Movie": sim_scores.index,
        "Content Score": sim_scores.values
    })
    return recommendations


def recommend_by_collaborative(movie_title, top_n=10):
    if movie_title not in movie_similarity_df.columns:
        return pd.DataFrame(columns=["Movie", "Collaborative Score"])

    sim_scores = movie_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]

    recommendations = pd.DataFrame({
        "Movie": sim_scores.index,
        "Collaborative Score": sim_scores.values
    })
    return recommendations


def recommend_hybrid(movie_title, top_n=10, alpha=0.5):
    cf_rec = recommend_by_collaborative(movie_title, top_n=50)
    content_rec = recommend_by_content(movie_title, top_n=50)

    if cf_rec.empty and content_rec.empty:
        return pd.DataFrame(columns=["Movie", "Collaborative Score", "Content Score", "Hybrid Score"])

    merged = pd.merge(cf_rec, content_rec, on="Movie", how="outer").fillna(0)

    merged["Hybrid Score"] = (
        alpha * merged["Collaborative Score"] +
        (1 - alpha) * merged["Content Score"]
    )

    merged = merged.sort_values("Hybrid Score", ascending=False).head(top_n).reset_index(drop=True)
    return merged


# =========================
# 5. Sidebar
# =========================
st.sidebar.header("⚙️ Settings")

available_movies = sorted(list(set(movies_content["title"]).intersection(set(movie_similarity_df.columns))))

selected_movie = st.sidebar.selectbox("Choose a movie", available_movies)
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
alpha = st.sidebar.slider("Hybrid weight (Collaborative)", 0.0, 1.0, 0.5, 0.1)

recommendation_type = st.sidebar.radio(
    "Recommendation mode",
    ["Hybrid", "Collaborative", "Content-Based"]
)

# =========================
# 6. Main display
# =========================
st.subheader(f"Selected Movie: {selected_movie}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Movie Info")
    movie_row = movies_content[movies_content["title"] == selected_movie]
    if not movie_row.empty:
        overview = movie_row.iloc[0]["overview"]
        genres_text = movie_row.iloc[0]["genres_text"]
        st.write(f"**Genres:** {genres_text if genres_text else 'N/A'}")
        st.write(f"**Overview:** {overview if overview else 'No overview available.'}")

with col2:
    st.markdown("### Dataset Info")
    st.write(f"**Movies in content dataset:** {movies_content.shape[0]}")
    st.write(f"**Ratings records:** {ratings.shape[0]}")
    st.write(f"**Movies available for collaborative filtering:** {len(movie_similarity_df.columns)}")

st.markdown("---")
st.markdown("## 🔍 Recommendations")

if recommendation_type == "Hybrid":
    result = recommend_hybrid(selected_movie, top_n=top_n, alpha=alpha)
elif recommendation_type == "Collaborative":
    result = recommend_by_collaborative(selected_movie, top_n=top_n)
else:
    result = recommend_by_content(selected_movie, top_n=top_n)

if result.empty:
    st.warning("No recommendations found for this movie.")
else:
    st.dataframe(result, use_container_width=True)

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download recommendations as CSV",
        data=csv,
        file_name=f"{selected_movie}_recommendations.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("### 📌 About this app")
st.markdown(
    """
- **Collaborative Filtering**: recommends movies liked by users with similar rating behavior.
- **Content-Based Filtering**: recommends movies with similar genres and descriptions.
- **Hybrid Recommendation**: combines both approaches using a weighted score.
"""
)
