import torch
import pickle
from Dataset import BIODataset
from vocab import Vocab
from load_embeddings import load_embeddings

device="cuda" if torch.cuda.is_available else "cpu"

#build the vocabulary
def build_vocab(dataset):
    vocab = Vocab()
    vocab.stoi['<unk>']=0
    vocab.stoi['<pad>']=1
    idx = 2
    #add every new to token to the dictionary from string to idx
    for i in range(len(dataset)):
        for token in dataset.get_raw_element(i)["tokens"]:
            token = token.lower()
            if token is not None and token not in vocab.stoi.keys():
                vocab.stoi[token]=idx
                idx+=1
    #map from index to string is just a list
    vocab.itos = list(vocab.stoi.keys())
    return vocab

#do the same for the labels
def build_label_vocab(dataset):
    vocab = Vocab()
    idx = 0
    for i in range(len(dataset)):
        for token in dataset.get_raw_element(i)["labels"]:
            if token is not None and token not in vocab.stoi.keys():
                vocab.stoi[token]=idx
                idx+=1
    vocab.itos = list(vocab.stoi.keys())
    return vocab

def load_data():
    #create Dataset and vocabulary
    train_dataset = BIODataset('./data/train.jsonl', device=device)
    dev_dataset = BIODataset('./data/dev.jsonl', device=device)
    test_dataset = BIODataset('./data/test.jsonl', device=device)
    vocabulary = build_vocab(train_dataset)
    label_vocabulary = build_label_vocab(train_dataset)

    #load pretrained embeddings
    with open("embeddings.pkl", 'rb') as f:
        embedding = pickle.load(f)

    #embedding = load_embeddings("komninos")

    #create the embedding matrix
    pretrained_embeddings = torch.randn(len(vocabulary), len(embedding['and']))
    for i, w in enumerate(vocabulary.itos):
        if w in embedding.keys():
            vec = embedding[w]
            pretrained_embeddings[i] = torch.Tensor(vec)
            #pretrained embeddings are frozen during training
            #not freezing them leads to very quick overfitting
            pretrained_embeddings[i].requires_grad = False
        else:
            #words that are not present in ptretrained embeddings are mapped to a randomly initialized
            #vector, whose weights get updated during training
            pretrained_embeddings[i].requires_grad = True

    #index the dataset, more details in the Dataset class
    train_dataset.index_dataset(vocabulary, label_vocabulary)
    dev_dataset.index_dataset(vocabulary, label_vocabulary)
    test_dataset.index_dataset(vocabulary, label_vocabulary)

    return train_dataset, dev_dataset, test_dataset, vocabulary, label_vocabulary, pretrained_embeddings

