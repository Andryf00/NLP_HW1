import json
from torch.utils.data import Dataset
import torch
import nltk
from nltk.corpus import wordnet

class BIODataset(Dataset):

    def __init__(self, 
                 input_file:str, 
                 device="cuda"):
        """
        We assume that the dataset pointed by input_file is already tokenized 
        and can fit in memory.
        Args:
            input_file (string): The path to the dataset to be loaded.
            window_size (integer): The maximum length of a sentence in terms of 
            number of tokens.
            device (string): device where to put tensors (cpu or cuda).
        """

        self.input_file = input_file
        with open(input_file) as f:
            # read the entire file with reader.read() e parse it
            self.data = [json.loads(line) for line in f]
        self.device = device
        self.encoded_data = None
    
    def index_dataset(self, vocabulary, label_vocabulary):
        self.encoded_data = list()
        for i in range(len(self.data)):
            elem = self.data[i]
            encoded_pos = self.encode_pos(elem["tokens"]).to(self.device)
            encoded_casing = self.encode_casing(elem["tokens"]).to(self.device)
            encoded_words = torch.LongTensor(self.encode_text(elem["tokens"], vocabulary)).to(self.device)
            encoded_labels = torch.LongTensor([label_vocabulary[label] for label in elem["labels"]]).to(self.device)
            self.encoded_data.append({"inputs":encoded_words,
                                      "casing":encoded_casing, 
                                      "pos": encoded_pos,
                                      "outputs":encoded_labels})
    def encode_pos(self, sentence):
        tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
        tagged_sentence = nltk.pos_tag(sentence, tagset='universal')
        one_hot_encoding = torch.zeros((len(sentence), len(tags)))
        i=0
        for word,tag in tagged_sentence:
            one_hot_encoding[i, tags.index(tag)] = 1
            i+=1
        return one_hot_encoding

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

    def encode_text(self, sentence, l_vocabulary):
        indices = list()
        for w in sentence:
            w = w.lower()
            if w in l_vocabulary.stoi: # vocabulary string to integer
                indices.append(l_vocabulary[w])
            else:
                synonyms = wordnet.synsets(w)
                found = False
                for w in synonyms:
                    w = w.lemmas()[0].name()
                    if w in l_vocabulary.stoi: # vocabulary string to integer
                        indices.append(l_vocabulary[w])
                        found = True
                        break
                if not found:
                    indices.append(l_vocabulary.unk_index)
        return indices
    
    