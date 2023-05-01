import torch
from torch import nn
from Dataset import BIODataset
from torchtext.vocab import Vocab
from collections import Counter
import nltk
from nltk.corpus import wordnet
from Model import Model
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2 
import matplotlib.pyplot as plt

train_loss_lin = [0.2666, 0.1753, 0.1531, 0.1420, 0.1343, 0.1286, 0.1235, 0.1179, 0.1124, 0.1067, 0.1016, 0.0959, 0.0912]
valid_loss_lin = [0.1826, 0.1559, 0.1465, 0.1447, 0.1433, 0.1442, 0.1424, 0.1443, 0.1478, 0.1505, 0.1578, 0.1623, 0.1631]
f1_score_lin = [0.4640, 0.6456, 0.6914, 0.6966, 0.7029, 0.6999, 0.7064, 0.7040, 0.7033, 0.7019, 0.6973, 0.7039, 0.6951]
#bi crf glove
train_loss_glove = [5.95, 3.86, 3.45, 3.23, 3.07, 2.93, 2.8, 2.67, 2.52, 2.38, 2.23, 2.109, 1.96]
val_loss_glove = [4.12, 3.60, 3.46, 3.39, 3.39, 3.41, 3.4, 3.47, 3.53, 3.66, 3.74, 3.90, 4.02 ]
f1_score_glove= [0.59, 0.69, 0.7027, 0.7068, 0.7104, 0.7065, 0.7111, 0.707, 0.709, 0.7028, 0.701, 0.692, 0.6919]

# bi crf komn
train_loss_komn = [5.32, 3.57, 3.3, 3.11, 2.98, 2.84, 2.71, 2.58, 2.43, 2.28]
val_loss_komn = [3.96, 3.60, 3.5, 3.41, 3.42, 3.47, 3.50, 3.60, 3.67, 3.87]
f1_komn = [0.6379, 0.6977, 0.7141, 0.7107, 0.7152, 0.7136, 0.7086, 0.7079, 0.7069, 0.7025]

#var
train_loss_var = [5.23, 4.57, 4.38, 4.30, 4.23, 4.21, 4.14, 4.13, 4.14, 4.1, 4.08, 4.08, 4.06]
val_loss_var = [3.68, 3.50, 3.43, 3.36, 3.39, 3.36, 3.35, 3.34, 3.45, 3.41, 3.33, 3.42, 3.46]
f1_var = [0.6532, 0.6697, 0.685, 0.692, 0.6957, 0.702, 0.7076, 0.7, 0.7059, 0.7024, 0.7021, 0.7040, 0.7049]

plt.plot(train_loss_komn, color='red', label = "train-naive")
plt.plot(val_loss_komn, color='orange', label = "val-naive")
plt.plot(val_loss_var, color='blue', label= "train-variational")
plt.plot(train_loss_var, color='cyan', label="val-variational")
plt.title("Naive vs Variational dropout")
plt.legend()
plt.show()

exit()

plt.plot(train_loss_glove, 'r', label = "glove")
plt.plot(train_loss_lin, 'b', label = "softmax")
plt.plot(train_loss_komn, 'g', label= "komn")
plt.plot(train_loss_var, 'm', label="variational")
plt.title("Training loss")
plt.legend()
plt.show()

plt.plot(val_loss_glove, 'r', label = "glove")
plt.plot(valid_loss_lin, 'b', label = "softmax")
plt.plot(val_loss_komn, 'g', label= "komn")
plt.plot(val_loss_var, 'm', label="variational")
plt.title("Validation loss")
plt.legend()
plt.show()

plt.plot(f1_score_glove, 'r', label = "glove")
plt.plot(f1_score_lin, 'b', label = "softmax")
plt.plot(f1_komn, 'g', label= "komn")
plt.plot(f1_var, 'm', label="variational")
plt.title("F1 score")
plt.legend()
plt.show()

exit()


def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for i in range(len(dataset)):
        for token in dataset.get_raw_element(i)["tokens"]:
            if token is not None:
                counter[token.lower()]+=1
    return Vocab(counter)

def build_label_vocab(dataset):
    counter = Counter()
    for i in range(len(dataset)):
        for token in dataset.get_raw_element(i)["labels"]:
            if token is not None:
                counter[token]+=1
    return Vocab(counter, specials=[])

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dataset = BIODataset('./data/train.jsonl', device=device)
dev_dataset = BIODataset('./data/dev.jsonl', device=device)
test_dataset = BIODataset('./data/test.jsonl', device=device)
vocabulary = build_vocab(train_dataset, min_freq=1)
label_vocabulary = build_label_vocab(train_dataset)
train_dataset.index_dataset(vocabulary, label_vocabulary)
dev_dataset.index_dataset(vocabulary, label_vocabulary)
test_dataset.index_dataset(vocabulary, label_vocabulary)

print(label_vocabulary.stoi)

class HParams():
    vocab_size = len(vocabulary)
    hidden_dim = 100
    embedding_dim = 300
    num_classes = len(label_vocabulary)
    bidirectional = True
    num_layers = 2
    dropout = 0.25
    embeddings = None
    crf = False
net = Model(HParams).to(device)
net.load_state_dict(torch.load("best_linear"))

class HParams():
    vocab_size = len(vocabulary)
    hidden_dim = 100
    embedding_dim = 300
    num_classes = len(label_vocabulary)
    bidirectional = True
    num_layers = 2
    dropout = 0.25
    embeddings = None
    crf = True
net2 = Model(HParams).to(device)
net2.load_state_dict(torch.load("best_0.709903979062401"))

def decode_output(x):
    return [label_vocabulary.itos[int(idx)] for idx in x]


outputs=[]
total_labels=[]
for sample in test_dataset:
    inputs = sample['inputs']
    casing = sample['casing']
    pos = sample['pos']
    labels = sample['outputs']
    labels = labels.view(-1)
    predictions = net(inputs, casing, pos)
    labels = labels.view(-1)
    preds = []
    for p in predictions:
        preds.append(torch.argmax(p).item())
    outputs.append(decode_output(preds))
    total_labels.append(decode_output([x.item() for x in labels]))
valid_f1 = f1_score(outputs, total_labels, average="macro", scheme=IOB2, mode='strict', zero_division=0)
print(valid_f1) 
total_labels = []
outputs2 = []
for sample in test_dataset:
    inputs = sample['inputs']
    casing = sample['casing']
    pos = sample['pos']
    labels = sample['outputs']
    labels = labels.view(-1)
    tag_scores, tag_seq = net2(inputs, casing, pos)
    #sample_loss = -net.crf(tag_scores.unsqueeze(0), labels.unsqueeze(0), mask=None)
    tag_seq = [x[0] for x in tag_seq]
    outputs2.append(decode_output(tag_seq))
    total_labels.append(decode_output([x.item() for x in labels]))
valid_f1 = f1_score(outputs2, total_labels, average="macro", scheme=IOB2, mode='strict', zero_division=0)
print(valid_f1) 


def is_legal(sequence):
    B = False
    for lab in sequence:
        if lab in ['B-ACTION', 'B-CHANGE', 'B-SCENARIO', 'B-SENTIMENT', 'B-POSSESSION']: B=True
        if lab in ['I-CHANGE', 'I-SCENARIO', 'I-ACTION', 'I-POSSESSION', 'I-SENTIMENT'] and B==False: return False
    return True 

ill_crf = 0
ill_lin = 0
for i in range(2000):
    show = False
    if not is_legal(outputs[i]): 
        ill_lin+=1
        show = True
    if not is_legal(outputs2[i]): 
        ill_crf+=1
        show = True
    if False:
        print("__________________")
        print(outputs[i])
        print(outputs2[i])
        print(total_labels[i])
print(ill_lin, ill_crf)


exit()

iv = 0
ov = 0
syn = 0
for i in range(len(test_dataset)):
    elem = test_dataset.data[i]
    sentence=elem["tokens"]
    labels = elem["labels"]
    for w in sentence:
        if w in vocabulary.stoi: # vocabulary string to integer
            iv+=1
        else:
            ov += 1
            synonyms = wordnet.synsets(w)
            found = False
            for w in synonyms:
                w = w.lemmas()[0].name()
                if w in vocabulary.stoi: # vocabulary string to integer
                    syn+=1
                    found = True
                    break

print("There where {} words in vocabulary, and {} out of vocabulary. Of those, {} where substituted by a synonym, while {} were mapped to <unk>".format(iv, ov,  syn, ov-syn))
exit()




verb2lab = {}
noun2lab = {}
adv2lab = {}
for label in label_vocabulary.itos:
    verb2lab[label] = 0
    noun2lab[label] = 0
    adv2lab[label] = 0
print(verb2lab)
for i in range(len(train_dataset)):
        elem = train_dataset.data[i]
        sentence=elem["tokens"]
        labels = elem["labels"]
        tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
        tagged_sentence = nltk.pos_tag(sentence, tagset='universal')
        one_hot_encoding = torch.zeros((len(sentence), len(tags)))
        i=0
        for word,tag in tagged_sentence:
            if tag == 'VERB': 
                verb2lab[labels[i]]+=1
            elif tag == 'NOUN':
                noun2lab[labels[i]]+=1
            elif tag == 'ADV':
                adv2lab[labels[i]]+=1
            i+=1

print(verb2lab)

import matplotlib.pyplot as plt

values = list(verb2lab.values())
keys = list(verb2lab.keys())
plt.bar(keys, values)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Frequency of labels for "VERB"')
plt.show()
values = list(noun2lab.values())
keys = list(noun2lab.keys())
plt.bar(keys, values)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Frequency of labels for "NOUN"')
plt.show() 
values = list(adv2lab.values())
keys = list(adv2lab.keys())
plt.bar(keys, values)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Frequency of labels for "ADV"')
plt.show() 