import torch
import json
import pickle
import torch
from torch import nn
from Dataset import BIODataset
from torchtext.vocab import Vocab
from collections import Counter
#from Model import Model
from lstm_var import AWD_LSTM
from Trainer import Trainer
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2 
from Model_2 import Model_2
from Model import Model
device="cuda" if torch.cuda.is_available else "cpu"


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
    return Vocab(counter, specials=[])

def build_char_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\\|_@#$%^&*˜‘+-=<>()[]{}"
    return {char: idx for idx, char in enumerate(chars)}

train_dataset = BIODataset('./data/train.jsonl', device=device)
dev_dataset = BIODataset('./data/dev.jsonl', device=device)
test_dataset = BIODataset('./data/test.jsonl', device=device)
vocabulary = build_vocab(train_dataset, min_freq=2)
label_vocabulary = build_label_vocab(train_dataset)
char_vocabulary = build_char_vocab()

#FIRST I BUILD CHAR_VOCAB, MAPPING CHARS TO IDX, THEN WHEN INDEXING THE DATASET, I CREATE A LIST(SENTENCE) OF LIST(CHARS IN WORD),
# THEN I PASS IT ALL TO THE MODEL, WHERE I EMBED SIMILARLY TO WORDS, PLUS CONV AND MAX POOLING. 
# HOW DOES EMBEDDING WORK WITH MULTIPLE CHARS? HOW WILL THIS BEHAVE WITH MINI BATCH SIZE != 1? 

    


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

train_dataset.index_dataset(vocabulary, label_vocabulary)#, char_vocabulary)
dev_dataset.index_dataset(vocabulary, label_vocabulary)#, char_vocabulary)
test_dataset.index_dataset(vocabulary, label_vocabulary)#, char_vocabulary)

class HParams():
    vocab_size = len(vocabulary)
    hidden_dim = 100
    embedding_dim = 300
    num_classes = len(label_vocabulary)
    bidirectional = True
    num_layers = 2
    dropout = 0.25
    embeddings = pretrained_embeddings
net = Model(HParams).to(device)
#net = Model_2(HParams).to(device)


def train():
    trainer=Trainer(net, nn.CrossEntropyLoss(ignore_index=label_vocabulary['<pad>']), torch.optim.Adam(net.parameters(), lr=0.001), label_vocabulary)
    trainer.train(train_dataset, dev_dataset, 25)

train()

def decode_output(x):
    return [label_vocabulary.itos[int(idx)] for idx in x]

def validation():
    net.load_state_dict(torch.load("best_relu"))
    outputs=[]
    total_labels=[]
    for sample in test_dataset:
        inputs = sample['inputs']
        labels = sample['outputs']
        labels = labels.view(-1)
        tag_scores, tag_seq = net(inputs)
        #sample_loss = -net.crf(tag_scores.unsqueeze(0), labels.unsqueeze(0), mask=None)
        tag_seq = [x[0] for x in tag_seq]
        """
        predictions = net(inputs)
        labels = labels.view(-1)
        preds = []
        for p in predictions:
            preds.append(torch.argmax(p).item())
        outputs.append(decode_output(preds))
        total_labels.append(decode_output([x.item() for x in labels]))"""
        outputs.append(decode_output(tag_seq))
        total_labels.append(decode_output([x.item() for x in labels]))
    valid_f1 = f1_score(outputs, total_labels, average="macro", scheme=IOB2, mode='strict', zero_division=0)
    print(valid_f1)  
    trues=0
    for i in range(len(test_dataset)):
        if outputs[i]==total_labels[i]: trues+=1
        #print(outputs[i]==total_labels[i], outputs[i], total_labels[i])
    print("TRUES", trues,i)
            
        
validation()