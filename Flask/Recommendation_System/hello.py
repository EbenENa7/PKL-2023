from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model, Model, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import  Input, dot, concatenate
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
import re, os, ast

app = Flask(__name__)

# Deep learning 
def load_model_dl_data():
    global n_movies
    global model2
    dataset = pd.read_csv('movie.csv')
    dataset['userId'] = dataset['userId'].astype('category').cat.codes
    dataset['movieId'] = dataset['movieId'].astype('category').cat.codes
    model2 = load_model('model2.h5')
    n_movies = dataset['movieId'].nunique()
    return model2, dataset

# TF-IDF similarity
def load_similarity_data():
    movie = pd.read_csv('movie.csv')
    movie.drop_duplicates(subset=['movieId'], inplace=True)
    # movie.drop('Unnamed: 0', axis=1, inplace=True)
    movie.reset_index(inplace=True)
    movie = movie[['movieId', 'original_title', 'genres', 'tagline']]
    movie['movieId'] = pd.Series(range(1, len(movie) + 1))
    movie['features'] = movie['genres'].str.lower() + ' ' + movie['tagline'].str.lower()
    movie['features'] = movie['features'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    tfidf = TfidfVectorizer()
    feature_matrix = tfidf.fit_transform(movie['features'])
    similarity_scores = cosine_similarity(feature_matrix)

    return movie, similarity_scores

model2, dataset = load_model_dl_data()
movie, similarity_scores = load_similarity_data()

def recommend_dl_movies(user_id, num_recommendations=10):
    user_movies = dataset[dataset['userId'] == user_id]['movieId']
    user_movies = user_movies.tolist()
    all_movies = list(range(1, n_movies + 1))  # Adjust the range
    movies_to_recommend = [movie for movie in all_movies if movie not in user_movies]
    user_input = np.array([user_id] * len(movies_to_recommend))
    movie_input = np.array(movies_to_recommend)
    predicted_ratings = model2.predict([user_input, movie_input])
    movie_ratings = list(zip(movies_to_recommend, predicted_ratings.flatten()))
    movie_ratings.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = movie_ratings[3:num_recommendations+3]
    recommended_movie_ids = [movie_id for movie_id, _ in top_recommendations]
    return recommended_movie_ids

def get_movie_titles(movie_ids):
    movie_titles = [movie[movie['movieId'] == movie_id]['original_title'].values[0] for movie_id in movie_ids]
    return movie_titles

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form.get('user_id'))
    selected_method = request.form.get('method')

    if selected_method == 'dl':
        dl_recommendations = recommend_dl_movies(user_id)
        dl_recommendations_with_titles = get_movie_titles(dl_recommendations)
        recommendations = dl_recommendations_with_titles
    elif selected_method == 'similarity':
        similarity_recommendations = get_movie_recommendations(user_id)
        similarity_recommendations_with_titles = get_movie_titles(similarity_recommendations)
        recommendations = similarity_recommendations_with_titles
    else:
        recommendations = []
    
    return render_template('index.html', recommendations=recommendations)


def get_movie_recommendations(user_id, top_n=10):
    user_movies = dataset[dataset['userId'] == user_id]['movieId']
    user_movies = user_movies.tolist()
    all_movies = list(range(1, n_movies + 1)) 
    movies_to_recommend = [movie for movie in all_movies if movie not in user_movies]
    movie_input = np.array(movies_to_recommend)
    user_movie_index = user_movies[-1] - 1
    movies_to_recommend_array = np.array(movies_to_recommend)
    predicted_scores = similarity_scores[user_movie_index, movies_to_recommend_array - 1]
    predicted_scores_sorted = sorted(predicted_scores, reverse=True)
    movie_scores = list(zip(movies_to_recommend, predicted_scores))
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_movie_ids = movies_to_recommend_array[predicted_scores.argsort()[::-1]] 
    recommended_movie_ids = recommended_movie_ids[:top_n]
    return recommended_movie_ids    

if __name__ == '__main__':
    app.run(debug=True)



