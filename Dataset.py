import json
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import wordnet

#Dataset class
#Note that we do not truncate or pad sentences, because for training I'm always using batch_size = 1


class BIODataset(Dataset):

    def __init__(self, 
                 input_file:str, 
                 device="cuda"):

        self.input_file = input_file
        with open(input_file) as f:
            # read the entire file
            self.data = [json.loads(line) for line in f]
        self.device = device
        self.encoded_data = None
    #index the dataset, also encode the casing and POS tags for each word
    def index_dataset(self, vocabulary, label_vocabulary):
        self.encoded_data = []
        for i in range(len(self.data)):
            elem = self.data[i] #elem is a dict with keys "tokens" and "labels"
            encoded_pos = self.encode_pos(elem["tokens"]).to(self.device)
            encoded_casing = self.encode_casing(elem["tokens"]).to(self.device)
            encoded_words = torch.LongTensor(self.encode_text(elem["tokens"], vocabulary)).to(self.device)
            encoded_labels = torch.LongTensor([label_vocabulary.stoi[label] for label in elem["labels"]]).to(self.device)
            self.encoded_data.append({"inputs":encoded_words,
                                      "casing":encoded_casing, 
                                      "pos": encoded_pos,
                                      "outputs":encoded_labels})
    #using the nltk POS tagger, I encode the tag of each word in a one hot encoding vector
    def encode_pos(self, sentence):
        tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
        tagged_sentence = nltk.pos_tag(sentence, tagset='universal')
        one_hot_encoding = torch.zeros((len(sentence), len(tags)))
        i=0
        for word,tag in tagged_sentence:
            one_hot_encoding[i, tags.index(tag)] = 1
            i+=1
        return one_hot_encoding

    #encode the casing of a word in a one hot encoder vector
    def encode_casing_word(self, word):
        if word.isnumeric():
            return torch.tensor([1, 0, 0, 0, 0, 0, 0])  # numeric
        elif sum(1 for c in word if c.isnumeric()) / len(word) > 0.5:
            return torch.tensor([0, 1, 0, 0, 0, 0, 0])  # mainly numeric
        elif word.islower():
            return torch.tensor([0, 0, 1, 0, 0, 0, 0])  # all lower
        elif word.isupper():
            return torch.tensor([0, 0, 0, 1, 0, 0, 0])  # all upper
        elif word[0].isupper() and word[1:].islower():
            return torch.tensor([0, 0, 0, 0, 1, 0, 0])  # initial upper
        elif any(c.isdigit() for c in word):
            return torch.tensor([0, 0, 0, 0, 0, 1, 0])  # contains digit
        else:
            return torch.tensor([0, 0, 0, 0, 0, 0, 1])  # other
    #encode the casing of each word in a sentence, returns a matrix of one hot encodings
    def encode_casing(self, sentence):
        encoding = torch.zeros((len(sentence), 7))
        for i in range(len(sentence)):
            encoding[i]=self.encode_casing_word(sentence[i])
        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]
    
    def get_raw_element(self, idx):
        return self.data[idx]
    #indexes each word in a sentence
    def encode_text(self, sentence, l_vocabulary):
        indices = list()
        for w in sentence:
            w = w.lower()
            #if word is present in the vocabulary we're goode
            if w in l_vocabulary.stoi.keys(): # vocabulary string to integer
                indices.append(l_vocabulary.stoi[w])
            #if the word is not present in the vocabulary
            else:
                synonyms = wordnet.synsets(w)
                found = False
                #check whether any synonym is present in the vocabulary
                for w in synonyms:
                    w = w.lemmas()[0].name()
                    if w in l_vocabulary.stoi.keys():
                        #use the index of the synonym as the index of the word
                        indices.append(l_vocabulary.stoi[w])
                        found = True
                        break
                #if there are no synonyms in the dataset, index to 0, which is the index of <unk>
                if not found:
                    indices.append(0)
        return indices
    
    