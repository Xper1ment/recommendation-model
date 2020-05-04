import pickle ,json,math
from flask import Flask,jsonify,request
from book import recomm_books
from flask_cors import CORS
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

with open('./data/model.pkl','rb') as f:
    count_matrix = pickle.load(f)

with open('./data/df2.pkl','rb') as f:
    df2 = pickle.load(f) 
    
df2 = df2.reset_index()
#count_matrix = count1.fit_transform(df2['soup'])    

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df2.index, index=df2['title'])

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = {'name': value , 'website': dict1[key]}   
    return dict3


def get_recommendations(title, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    dict1 =  df2['title'].iloc[movie_indices].to_dict()
    
    dict2 = df2['homepage'].iloc[movie_indices]
    dict2 = dict2.fillna('')
    dict2 = dict2.to_dict()
    dict3 = mergeDict(dict2,dict1)
    keys_values = dict3.items()
    result = {str(key): value for key, value in keys_values}
    #result = json.dumps(new_d) 
    return result

@app.route('/recommend/',methods=['GET'])

def recommend():
    title = request.args.get("title", None)
    result = get_recommendations(title,cosine_sim2)
    return jsonify(result)

@app.route('/recommend_books/',methods=['GET'])

def recommend_books():
    title = request.args.get("title", None)
    result = recomm_books(title)
    return jsonify(result)


@app.route('/')

def default():
    return "<h1>Welcome</h1>"

if __name__ == "__main__":
    app.run(threaded = True,port = 5000)    
