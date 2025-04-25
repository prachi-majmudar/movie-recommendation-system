# 🎬 Movie Recommendation System | TMDb Live Integration 🚀

Welcome to the Movie Recommendation System — a content-based movie recommender that intelligently suggests similar movies based on description analysis!

🌟 **Live App**: [Click Here to Try the App](https://movie-recommendation-system-iungy5cn9hg4qjgp7dzpuk.streamlit.app/)  

---

## 📈 Project Overview

This project was built to showcase real-world skills in:

- Data Analysis (Pandas, Numpy)
- Machine Learning (TF-IDF Vectorization, Cosine Similarity)
- API Integration (TMDb API)
- Frontend Web Application (Streamlit)
- Cloud Deployment (Streamlit Community Cloud)
- Version Control (Git & GitHub)

---

## 📚 Dataset

- Source: A curated list of movies with title and overview
- File: `movies.csv`
- Fields used: `title`, `overview`

---

## 🚀 How It Works

1. The user inputs a movie title.
2. If the movie exists in the dataset:
   - The app recommends 10 similar movies based on overview similarity.
3. If the movie is **NOT** in the dataset:
   - The app **fetches live overview** from **The Movie Database (TMDb) API**.
   - It dynamically vectorizes the fetched overview and recommends similar movies!
   
✅ Real-time movie recommendation even beyond the original dataset.

---

## 🛠 Tech Stack

- **Python 3.10**
- **Pandas**, **NumPy**
- **Scikit-Learn** (TF-IDF, Cosine Similarity)
- **Streamlit** (for web deployment)
- **TMDb API** (for dynamic movie search)

