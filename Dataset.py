import json
from torch.utils.data import Dataset
import torch

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
            encoded_elem = torch.LongTensor(self.encode_text(elem["tokens"], vocabulary)).to(self.device)
            encoded_labels = torch.LongTensor([label_vocabulary[label] for label in elem["labels"]]).to(self.device)
            self.encoded_data.append({"inputs":encoded_elem, 
                                      "outputs":encoded_labels})

    def preprocess_words(self, sentences):
        """ 
        Args:
            sentences (list of lists of dictionaries, 
                          where each dictionary represents a word occurrence parsed from a CoNLL line)
        """
        for sentence in sentences:
            for d in sentence:
                d["form"] = d["form"].lower()
        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]
    
    def get_raw_element(self, idx):
        return self.data[idx]

    @staticmethod
    def encode_text(sentence, l_vocabulary):
        """
        Args:
            sentences (list): list of OrderedDict, each carrying the information about
            one token.
            l_vocabulary (Vocab): vocabulary with mappings from words to indices and viceversa.
        Return:
            The method returns a list of indices corresponding to the input tokens.
        """
        indices = list()
        for w in sentence:
            if w in l_vocabulary.stoi: # vocabulary string to integer
                indices.append(l_vocabulary[w])
            else:
                indices.append(l_vocabulary.unk_index)
        return indices
    
    