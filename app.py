import pickle ,json,math
from flask import Flask,jsonify,request
from flask_cors import CORS
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

app = Flask(__name__)
CORS(app)

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify(error=e), 404
'''
from sklearn.feature_extraction.text import TfidfVectorizer


df1=pd.read_csv('./data/tmdb_5000_credits.csv')
df2=pd.read_csv('./data/tmdb_5000_movies.csv')
#df2 = pd.read_csv('data2.csv')
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')
df2['title'] =  df2['title'].str.lower()
#print(df2['title'].head(10))

# Parse the stringified features into their corresponding python objects
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

# Print the new features of the first 3 films
#df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['corpus'] = df2.apply(create_soup, axis=1)

#df2.to_csv('data2.csv')
# Import CountVectorizer and create the count matrix
count = TfidfVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['corpus'])     

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

y = False
def get_recommendations(title, cosine_sim):    
    # Get the index of the movie that matches the title
    try:
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
        return df2['title'].iloc[movie_indices]
    except KeyError:
        y = True
        pass
'''
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
    try:
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
    except KeyError:
        pass

@app.route('/recommend/',methods=['GET'])

def recommend():
    title = request.args.get("title", None)
    result = get_recommendations(title,cosine_sim2)
    if result is None:
        return resource_not_found("Resource not found")
    return jsonify(result)

'''@app.route('/recommend_books/',methods=['GET'])

def recommend_books():
    title = request.args.get("title", None)
    result = recomm_books(title)
    return jsonify(result)
'''

@app.route('/')

def default():
    return "<h1>Welcome</h1>"

if __name__ == "__main__":
    app.run(threaded = True)    
