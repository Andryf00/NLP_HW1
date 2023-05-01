import gzip
import pickle
import numpy as np
import torch
from zipfile import ZipFile

embedding_file = "glove.6B.zip"
embedding = {}
i=0
def load_glove():
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
    return embedding
        


def load_komn():


    embeddingsIn = gzip.open("komninos_english_embeddings.gz", "rt", encoding="utf8") 
    print(embeddingsIn)
    embedding = {}
    i=0
    for word in embeddingsIn:
        word = word.split()
        if(len(word[1:])>300): continue
        embedding[word[0]] = [float(x) for x in word[1:]]
    return embedding


def load_embeddings(name = "komninos"):
    if name == "komninos":
        return load_komn()
    elif name == "glove":
        return load_glove()
