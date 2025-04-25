import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
movies = pd.read_csv('data/movies.csv')
movies['overview'] = movies['overview'].fillna('')

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create title to index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found."]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit App
st.title('🎬 Movie Recommendation System')

st.subheader('Enter a movie you like:')
movie_input = st.text_input('Movie Title', '')

if st.button('Recommend'):
    if movie_input:
        recommendations = recommend_movies(movie_input)
        if recommendations[0] == "Movie not found.":
            st.error("Sorry, the movie was not found in our database.")
        else:
            st.success(f"Movies similar to **{movie_input}**:")
            for idx, movie in enumerate(recommendations, start=1):
                st.write(f"{idx}. {movie}")
    else:
        st.warning("Please enter a movie name to get recommendations.")
