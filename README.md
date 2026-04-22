# movie-recommendation-system
# 🎬 Hybrid Movie Recommendation System

This project builds a **movie recommendation system** using three approaches:

- Collaborative Filtering
- Content-Based Recommendation
- Hybrid Recommendation (combined model)

The goal is to compare different recommendation strategies and improve recommendation quality by combining multiple methods.

---

## 📌 Project Overview

Recommendation systems are widely used in platforms like Netflix, Amazon, and Spotify.

In this project, we:

- analyze movie rating data
- build similarity-based recommendation models
- combine different recommendation approaches into a hybrid system

---

## 📂 Dataset

- MovieLens dataset (user ratings)
- Movie metadata (genres, overview)

---

## ⚙️ Methods

### 1. Collaborative Filtering

- Construct user-movie rating matrix
- Compute cosine similarity between movies
- Recommend movies based on similar user behavior

---

### 2. Content-Based Recommendation

- Use TF-IDF on movie genres / descriptions
- Compute cosine similarity between movies
- Recommend movies based on content similarity

---

### 3. Hybrid Recommendation

We combine both methods:

**Hybrid Score = α × Collaborative + (1 - α) × Content**

- α controls the weight of each model
- allows flexible recommendation strategies

---

## 📊 Results

We tested different values of α:

- α = 0.2 → more content-based
- α = 0.5 → balanced
- α = 0.8 → more collaborative

Hybrid recommendation provides more robust and flexible results.

---

## 🚀 Project Highlights

- Built collaborative filtering recommender from scratch
- Implemented TF-IDF based content recommender
- Designed hybrid recommendation model
- Compared multiple recommendation strategies
- Structured project as a portfolio-ready notebook

---

## 🧠 Key Takeaways

- Collaborative filtering captures user behavior patterns
- Content-based recommendation captures semantic similarity
- Hybrid model improves recommendation quality and flexibility

---


## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
import kagglehub

---

## 🛠️ Download latest version-Dataset

path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
print("Path to dataset files:", path)

---

## 📬 Future Improvements

- Add deep learning models (e.g., Neural CF)
- Deploy as a web app (Streamlit)
- Introduce ranking optimization
