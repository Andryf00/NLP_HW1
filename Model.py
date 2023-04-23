
from torch import nn
from torchcrf import CRF

class Model(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(Model, self).__init__()   
        # Embedding layer: a mat∂rix vocab_size x embedding_dim where each index 
        # correspond to a word in the vocabulary and the i-th row corresponds to 
        # a latent representation of the i-th word in the vocabulary.
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim,
                                           _weight=hparams.embeddings)

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs 
        # a new **contextual** representation of each word that depend
        # on the preciding words.
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(64, hparams.num_classes)
        self.dense2 = nn.Linear(in_features=128, out_features = 64)
        self.dense1 = nn.Linear(in_features=lstm_output_dim, out_features=128)

    
    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        o = self.dense1(o)
        o = self.relu(o)
        o = self.dropout(o)
        o = self.dense2(o)
        o = self.relu(o)
        o = self.dropout(o)
        output = self.classifier(o)
        return output