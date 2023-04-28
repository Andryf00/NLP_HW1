
import torch
from torch import nn
from torchcrf import CRF


class Model(nn.Module):
    def __init__(self, hparams):
        super(Model, self).__init__()   
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim,
                                           _weight=hparams.embeddings)
        self.word_embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(64, hparams.num_classes)
        self.dense2 = nn.Linear(in_features=128, out_features = 64)
        self.dense1 = nn.Linear(in_features=lstm_output_dim, out_features=128)
        
        self.crf = CRF(hparams.num_classes)
        self.crf_linear = nn.Linear(lstm_output_dim, hparams.num_classes)

    
    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)

        o = self.crf_linear(o)
        tag_scores = nn.functional.log_softmax(o, dim=1)
        return tag_scores, self.crf.decode(tag_scores.unsqueeze(0), mask=None)
        """
        o = self.dropout(o)
        o = self.dense1(o)
        o = self.relu(o)
        o = self.dropout(o)
        o = self.dense2(o)
        o = self.relu(o)
        o = self.dropout(o)
        output = self.classifier(o)
        """
        return output