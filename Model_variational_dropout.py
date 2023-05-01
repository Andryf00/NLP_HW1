
import torch
from torch import nn
from torchcrf import CRF
from lstm_var import AWD_LSTM

#This module implements an architecture based on a Bi-LSTM with Variational dropout and a CRF classifier
#This architecture performed worse than Model.py, but I kept the implementation to show how it works



class Model_2(nn.Module):
    def __init__(self, hparams):
        super(Model_2, self).__init__()   

        self.lstm = AWD_LSTM(hparams)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.crf_linear = nn.Linear(lstm_output_dim, hparams.num_classes)
        self.crf = CRF(hparams.num_classes)
        
    
    def forward(self, x, casing, pos, hiddens):
        o, (h, c) = self.lstm(x, casing, pos, hiddens)
        tag_scores = nn.functional.log_softmax(o, dim=1)
        return tag_scores, self.crf.decode(tag_scores.unsqueeze(0), mask=None)
    
    