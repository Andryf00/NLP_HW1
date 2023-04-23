import gzip
import pickle
import numpy as np

numb = ['1','2','3','4', '5', '6','7','8','9','10']
def load_embedding(embedding_path):


    embeddingsIn = gzip.open(embedding_path, "rt", encoding="utf8") 
    embedding = {}
    i=0
    for word in embeddingsIn:
        word = word.split()
        if(len(word[1:])>300): continue
        embedding[word[0]] = [float(x) for x in word[1:]]
    with open("embeddings.pkl", 'wb') as f:
        pickle.dump(embedding, f)
    return embedding
e=load_embedding("komninos_english_embeddings.gz")
print(e['6'])


with open("embeddings.pkl", 'rb') as f:
    embedding = pickle.load( f)
print(embedding['6'])
print(len(embedding["6"]))