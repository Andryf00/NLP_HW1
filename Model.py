
import torch
from torch import nn
from torchcrf import CRF
import torch.nn.utils.rnn as rnn_utils

#Implementation of the main architecture used for this homework. 
#Allows both a crf classifier and a softmax classfier, choose using hparams.crf

class Model(nn.Module):
    def __init__(self, hparams):
        super(Model, self).__init__()   
        #load pretrained word embeddings
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim,
                                           _weight=hparams.embeddings)
        
        #input dim is 300 (komninos embds) + 7 (casing) +12 (POS tags), if using softmax classifier we don't use casing adn POS
        self.lstm = nn.LSTM(hparams.embedding_dim + 7 + 12 if hparams.crf else hparams.embedding_dim,
                            hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.crf = hparams.crf
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hparams.dropout)
        if self.crf:
            self.crf = CRF(hparams.num_classes)
            self.crf_linear = nn.Linear(lstm_output_dim, hparams.num_classes)
    
        self.classifier = nn.Linear(64, hparams.num_classes)
        self.dense2 = nn.Linear(in_features=128, out_features = 64)
        self.dense1 = nn.Linear(in_features=lstm_output_dim, out_features=128)
        
        
    
    def forward(self, x, casing, pos):

        embeddings = self.word_embedding(x)
        #only use casing and pos if using crf
        if self.crf: embeddings = torch.cat((embeddings, casing, pos), dim=1)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        if self.crf:
            o = self.crf_linear(o)
            o = self.dropout(o)
            tag_scores = nn.functional.log_softmax(o, dim=1)
            return tag_scores, self.crf.decode(tag_scores.unsqueeze(0), mask=None)
        else:
            o = self.dropout(o)
            o = self.dense1(o)
            o = self.relu(o)
            o = self.dropout(o)
            o = self.dense2(o)
            o = self.relu(o)
            o = self.dropout(o)
            output = self.classifier(o)
        
        return output