import torch
import json
import pickle
import torch
from torch import nn
from Dataset import BIODataset
from torchtext.vocab import Vocab
from collections import Counter
from Model import Model
from Trainer import Trainer

device="cuda" if torch.cuda.is_available else "cpu"

with open("embeddings.pkl", 'rb') as f:
    embeddings = pickle.load(f)

def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for i in range(len(dataset)):
        for token in dataset.get_raw_element(i)["tokens"]:
            if token is not None:
                counter[token]+=1
    return Vocab(counter)

def build_label_vocab(dataset):
    counter = Counter()
    for i in range(len(dataset)):
        for token in dataset.get_raw_element(i)["labels"]:
            if token is not None:
                counter[token]+=1
    return Vocab(counter)

#input_file = "/content/drive/My Drive/conll_2018_ud/ud-en-treebank-v2.2/UD_English-EWT/en_ewt-ud-train.conllu"

train_dataset = BIODataset('./data/train.jsonl', device=device)
dev_dataset = BIODataset('./data/dev.jsonl', device=device)
vocabulary = build_vocab(train_dataset, min_freq=2)
label_vocabulary = build_label_vocab(train_dataset)

    


with open("embeddings.pkl", 'rb') as f:
    embedding = pickle.load(f)

pretrained_embeddings = torch.randn(len(vocabulary), len(embedding['and']))
initialised = 0
for i, w in enumerate(vocabulary.itos):
    if w in embedding.keys():
        initialised += 1
        vec = embedding[w]
        pretrained_embeddings[i] = torch.Tensor(vec)
print("initialized words:",initialised)

train_dataset.index_dataset(vocabulary, label_vocabulary)
dev_dataset.index_dataset(vocabulary, label_vocabulary)


class HParams():
    vocab_size = len(vocabulary)
    hidden_dim = 100
    embedding_dim = 300
    num_classes = len(label_vocabulary) # number of different universal POS tags
    bidirectional = True
    num_layers = 2
    dropout = 0.5
    embeddings = pretrained_embeddings
net = Model(HParams).to(device)
trainer=Trainer(net, nn.CrossEntropyLoss(ignore_index=label_vocabulary['<pad>']), torch.optim.NAdam(net.parameters()), label_vocabulary)
trainer.train(train_dataset, dev_dataset, 5)
