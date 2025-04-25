# app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Your TMDb API Key
TMDB_API_KEY = '64d21f7ceb56fe43fca0f4041c9e8ae5'

# ✅ Function to fetch movie overview from TMDb if not found locally
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

# ✅ Recommendation Function
def recommend(title):
    title = title.title().strip()
    
    if title not in indices:
        st.warning("Movie not found in local database! Fetching from TMDb...")
        overview = fetch_movie_overview_from_tmdb(title)
        
        if overview is None:
            return ["❌ Movie not found. Please check spelling or try another title."]
        
        # Create TF-IDF vector for the fetched movie
        overview_vector = tfidf.transform([overview])
        
        # Calculate similarity with existing movies
        similarity_scores = cosine_similarity(overview_vector, tfidf_matrix)
        similarity_scores = similarity_scores.flatten()
        
        sim_indices = similarity_scores.argsort()[-10:][::-1]
        return movies['title'].iloc[sim_indices].tolist()
    
    else:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # top 10 excluding itself
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()

# ✅ Streamlit Frontend
st.set_page_config(page_title="Movie Recommendation System 🎬", page_icon="🎥")
st.title('🎬 Movie Recommendation System')

movie_title = st.text_input('Enter a Movie Title')

if st.button('Recommend'):
    if movie_title:
        recommendations = recommend(movie_title)
        st.subheader('Recommended Movies:')
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    else:
        st.warning('⚠️ Please enter a movie title to get recommendations.')

# Footer
st.markdown("---")
st.caption("Built with ❤️")

