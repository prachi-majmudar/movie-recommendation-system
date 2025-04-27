# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process  # ✅ Typo Handling

# ✅ Your TMDb API Key
TMDB_API_KEY = '64d21f7ceb56fe43fca0f4041c9e8ae5'

# ✅ Fetch poster from TMDb
def fetch_poster(movie_name):
    base_url = 'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_name
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data.get('results'):
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
    return "https://via.placeholder.com/300x450?text=No+Image"

# ✅ Fetch overview if movie not found locally
def fetch_movie_overview_from_tmdb(movie_name):
    base_url = 'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_name
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if data.get('results'):
        return data['results'][0]['overview']
    else:
        return None

# ✅ Load Dataset
movies = pd.read_csv('data/movies.csv')

# ✅ Preprocessing
movies['overview'] = movies['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ✅ Recommendation function with Typo Handling + Genre Filter
def recommend(title, selected_genre=None):
    title = title.title().strip()
    
    # Typo Handling
    movie_titles = movies['title'].tolist()
    best_match = process.extractOne(title, movie_titles, score_cutoff=70)
    
    if best_match:
        title = best_match[0]
    else:
        st.error("❌ Could not find any movie matching your input. Please try again.")
        return [], []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:30]  # Take top 30 to apply genre filter
    
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = movies.iloc[movie_indices]
    
    # Genre Filtering
    if selected_genre:
        recommended_movies = recommended_movies[recommended_movies['genres'].str.contains(selected_genre, case=False, na=False)]
    
    recommended_movies = recommended_movies.head(10)

    posters = []
    for movie in recommended_movies['title']:
        posters.append(fetch_poster(movie))
    
    return recommended_movies['title'].tolist(), posters

# ✅ Streamlit Frontend
st.set_page_config(page_title="Movie Recommendation System 🎬", page_icon="🎥")
st.title('🎬 Movie Recommendation System')

# Genre Filter Dropdown
genre_options = ['Action', 'Comedy', 'Drama', 'Thriller', 'Animation', 'Romance', 'Adventure', 'Science Fiction', 'Fantasy', 'Crime', 'Mystery', 'Family']
selected_genre = st.selectbox('Select Genre (Optional)', options=["Any"] + genre_options)

# Movie Title Input
movie_title = st.text_input('Enter a Movie Title')

if st.button('Recommend'):
    if movie_title:
        genre_filter = selected_genre if selected_genre != "Any" else None
        recommended_movies, posters = recommend(movie_title, genre_filter)
        
        if recommended_movies:
            st.subheader('Recommended Movies:')
            for title, poster in zip(recommended_movies, posters):
                st.image(poster, width=200)
                st.write(f"**{title}**")
                st.markdown("---")
    else:
        st.warning('⚠️ Please enter a movie title to get recommendations.')

# Footer
st.markdown("---")
st.caption("Area Under Construction")
