import gzip
import pickle
import numpy as np
import torch



from zipfile import ZipFile

embedding_file = "glove.6B.zip"
embedding = {}
i=0
# Open the embedding file and read the words and vectors
with ZipFile(embedding_file, 'r') as zip:
    with zip.open('glove.6B.300d.txt') as file:
        for line in file:
            #print(line)
            # Split the line into word and vector components
            line = line.decode().strip().split()
            word = line[0]
            i+=1
            try: vector = [float(x) for x in line[1:]]
            except: 
                print(i)
                print(line)
            embedding[word]=vector
        
with open("embeddings_glove.pkl", 'wb') as f:
    pickle.dump(embedding, f)
exit()
numb = ['1','2','3','4', '5', '6','7','8','9','10']
def load_embedding(embedding_path):


    embeddingsIn = gzip.open(embedding_path, "rt", encoding="utf8") 
    print(embeddingsIn)
    embedding = {}
    i=0
    for word in embeddingsIn:
        word = word.split()
        if(len(word[1:])>300): continue
        embedding[word[0]] = [float(x) for x in word[1:]]
    with open("embeddings.pkl", 'wb') as f:
        pickle.dump(embedding, f)
    return embedding
e=load_embedding("glove.zip")
print(e['6'])


with open("embeddings.pkl", 'rb') as f:
    embedding = pickle.load( f)
print(embedding['6'])
print(len(embedding["6"]))