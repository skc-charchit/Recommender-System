import streamlit as st
import pickle

# Load the data
movies = pickle.load(open("movies.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))
movies_list = movies["title"].values

st.header("TMDB Movies Recommender System")
select_value = st.selectbox("Select a Movie", movies_list)

def recommend(selected_movie):
    # Get the index of the selected movie
    movie_index = movies[movies['title'] == selected_movie].index[0]
    
    # Calculate distances and sort them
    distance = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    
    # Collect recommended movie titles
    recommend_movie = []
    for i in distance[1:6]:  # Get top 5 recommendations excluding the selected movie
        recommend_movie.append(movies.iloc[i[0]].title)
    
    return recommend_movie

if st.button("Show Recommendations"):
    movie_names = recommend(select_value)
    
    # Display recommendations in columns
    cols = st.columns(5)  # Create 5 columns dynamically
    
    for col, movie_name in zip(cols, movie_names):
        with col:
            st.text(movie_name)