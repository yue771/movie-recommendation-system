# movie-recommendation-system
# 🎬 Movie Recommendation System

## 📌 Project Overview

This project builds a movie recommendation system using collaborative filtering based on user rating data.

It recommends similar movies by analyzing user behavior patterns using cosine similarity.

---

## 📊 Dataset

- MovieLens (ratings_small.csv)
- TMDB metadata (movies_metadata.csv)
- Mapping file (links_small.csv)

---

## ⚙️ Methodology

1. Data cleaning and preprocessing
2. ID mapping across datasets (MovieLens → TMDB)
3. Constructing user-movie matrix
4. Filtering popular movies (≥ 20 ratings)
5. Computing cosine similarity
6. Generating recommendations

---

## 🎯 Example

Input:
Toy Story
Output:
Toy Story 2
Star Wars
Forrest Gump
Jurassic Park
...

---

## 📈 Key Results

- Captures user preference patterns effectively
- Produces meaningful recommendations for popular movies
- Simple and interpretable baseline model

---

## ⚠️ Limitations

- Cold start problem (new users/movies)
- No content-based features
- Depends heavily on rating density

---

## 🚀 Future Work

- Hybrid recommendation system
- Matrix factorization (SVD)
- Deploy with Streamlit

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
import kagglehub

## 🛠️ Download latest version-Dataset
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
print("Path to dataset files:", path)
