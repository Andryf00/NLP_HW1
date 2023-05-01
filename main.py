from Trainer import Trainer
from load_data import load_data
from Model import Model
import torch
from torch import nn
import pickle



def main():
    device = "cuda" if torch.cuda.is_available else "cpu"
    train_dataset, dev_dataset, test_dataset, vocabulary, label_vocabulary, pretrained_embeddings = load_data()
    
    with open("vocabulary.pkl", 'wb') as f:
        pickle.dump(vocabulary, f)

    with open("label_vocabulary.pkl", 'wb') as f:
        pickle.dump(label_vocabulary, f)

    class HParams():
        vocab_size = len(vocabulary)
        hidden_dim = 100
        embedding_dim = 300
        num_classes = len(label_vocabulary)
        bidirectional = True
        num_layers = 2
        dropout = 0.25
        embeddings = pretrained_embeddings
        crf = True
    
    net = Model(HParams).to(device)
    #net.load_state_dict(torch.load("2best_relu"))
    trainer=Trainer(net, nn.CrossEntropyLoss(), torch.optim.Adam(net.parameters(), lr=0.001), label_vocabulary, crf=HParams.crf)
    trainer.train(train_dataset, dev_dataset, 25)
    _, f1 = trainer.evaluate(test_dataset)
    print(f1)

if __name__ == "__main__":
    main()