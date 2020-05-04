import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


with open('./data/book_model.pkl', 'rb') as f:
    tfidf_matrix_corpus = pickle.load(f)

with open('./data/books.pkl','rb') as f:
    books = pickle.load(f)

cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = {'name': value , 'website': dict1[key]}   
    return dict3


# Build a 1-dimensional array with book titles
titles = books['title']
url = books['image_url']
indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of books tags
def recomm_books(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    urls = url.iloc[book_indices].to_dict()
    name = titles.iloc[book_indices].to_dict()
    ans = mergeDict(urls,name)
    keys_values = ans.items()
    result = {str(key): value for key, value in keys_values}
    return result

