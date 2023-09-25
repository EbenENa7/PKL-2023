from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

movie = pd.read_csv('movie.csv')
movie.drop_duplicates(subset=['movieId'], inplace=True)
movie.drop('Unnamed: 0', axis=1, inplace=True)
movie.reset_index(inplace=True)
movie['movieId'] = pd.Series(range(1, len(movie) + 1))

movie['features'] = movie['genres'].str.lower() + ' ' + movie['tagline'].str.lower()
movie['features'] = movie['features'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(movie['features'])

similarity_scores = cosine_similarity(feature_matrix)


def get_movie_recommendations(movie_id, top_n=5):
    if movie_id not in movie['movieId'].values:
        return None

    movie_index = movie.index[movie['movieId'] == movie_id].tolist()[0]
    similar_movies_indices = similarity_scores[movie_index].argsort()[::-1][1:(top_n + 1)]
    recommended_movies = movie.iloc[similar_movies_indices][['movieId', 'original_title']]
    return recommended_movies


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_id = int(request.form['movie_id'])
    recommended_movies = get_movie_recommendations(movie_id)
    if recommended_movies is None:
        return render_template('recommender.html', error_message='Invalid movieId. Please enter a valid movieId.')
    else:
        return render_template('recommender.html', recommended_movies=recommended_movies.to_dict('records'))


if __name__ == '__main__':
    app.run(debug=True)













# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import json

# app = Flask(__name__)

# movie = pd.read_csv('movie.csv')
# # movie['movie_id'] = pd.Series(range(1, 4810))
# movie['features'] = movie['genres'].str.lower() + ' ' + movie['keywords'].str.lower() + ' ' + movie['cast'].str.lower()

# tfidf = TfidfVectorizer()
# feature_matrix = tfidf.fit_transform(movie['features'])
# similarity_scores = cosine_similarity(feature_matrix)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/recommendations', methods=['POST'])
# def recommendations():
#     movie_id = int(request.form['movieId'])
#     top_n = 10  # Number of recommendations to show

#     movie_index = movie[movie['movieId'] == movie_id].index[0]
#     similar_movies = list(enumerate(similarity_scores[movie_index]))
#     sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
#     top_similar_movies = sorted_similar_movies[1:top_n + 1]
#     recommended_movies = [
#         {
#             'movie_id': movie.iloc[movie_idx]['movieId'],
#             'title': movie.iloc[movie_idx]['original_title'],
#             'similarity_score': similarity
#         }
#         for movie_idx, similarity in top_similar_movies
#     ]

#     return render_template('recommendations.html', movie_id=movie_id, movie_title=movie[movie['movieId'] == movie_id]['original_title'].values[0], recommendations=recommended_movies)
# if __name__ == '__main__':
#     app.run(debug=True)

# def recommendations():
#     movie_id = int(request.form['movie_id'])
#     top_n = 10  # Number of recommendations to show

#     # Get movie recommendations
#     movie_index = movie[movie['movie_id'] == movie_id].index[0]
#     similar_movies = list(enumerate(similarity_scores[movie_index]))
#     sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
#     top_similar_movies = sorted_similar_movies[1:top_n + 1]
#     recommended_movies = [(movie.iloc[movie_idx]['movie_id'], movie.iloc[movie_idx]['title'], similarity) for
#                           movie_idx, similarity in top_similar_movies]

#     movie_recommendations = pd.DataFrame(recommended_movies, columns=['Movie ID', 'Title', 'Similarity Score'])
#     # movie_recommendations = pd.DataFrame(recommended_movies, columns=['Movie ID', 'Title'])

#     # Render the recommendations template with the movie recommendations
#     return render_template('recommendations.html', movie_id=movie_id,
#                            movie_title=movie[movie['movie_id'] == movie_id]['title'].values[0],
#                            recommendations=movie_recommendations)


