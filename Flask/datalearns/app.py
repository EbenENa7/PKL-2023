from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import json

app = Flask(__name__)

data = pd.read_csv('datalearns_tags.csv')
data['topic_name'] = data['topic_name'].astype(str)
grouped_data = data.groupby('nid')['topic_name'].agg(', '.join).reset_index()
data = data.merge(grouped_data, on='nid', suffixes=('', '_grouped'))
data = data.drop_duplicates(subset='title').reset_index(drop=True)
data['features'] = data['title'].str.lower() +' '+ data['topic_name_grouped'].str.lower() +' '+ data['tag_name'].str.lower()

def remove_punctuation(text):
    if isinstance(text, str):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    else:
        return text

data['features'] = data['features'].apply(remove_punctuation)

tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(data['features'])
similarity_scores = cosine_similarity(feature_matrix)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    item_id = int(request.form['item_id'])
    num_recommendations = 7 

    item_index = data[data['nid'] == item_id].index[0]
    item_similarity = similarity_scores[item_index]
    similar_item_indices = item_similarity.argsort()[::-1][1:num_recommendations+1]

    recommended_items = []
    for index in similar_item_indices:
        recommended_item = {
            "Item ID": int(data.iloc[index]['nid']),
            "Title": data.iloc[index]['title']
        }
        recommended_items.append(recommended_item)
    # selected_item = {
    #     "Item ID": int(item_id),
    #     "Title": data[data['nid'] == item_id]['title'].values[0]
    # }
    # return render_template('recommendations.html', selected_item=selected_item, recommended_items=recommended_items)
    return jsonify(recommended_items=recommended_items)

if __name__ == '__main__':
    app.run(debug=True, port = 5002)
