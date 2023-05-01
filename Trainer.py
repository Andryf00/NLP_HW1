#The trainer class was adapted from the one presented on the colab notebook
import torch
from torch import nn
from Dataset import BIODataset
from torchtext.vocab import Vocab
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

import warnings
warnings.filterwarnings('always')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.plot()
    plt.show()
    return ax

class Trainer():

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        label_vocab: Vocab,
        patience=5,
        crf = True):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab
        self.patience = patience
        self.crf = crf
        
    
    #decodes from label index to label string
    def decode_output(self, x):
        return [self.label_vocab.itos[int(idx)] for idx in x]

    def train(self, train_dataset:BIODataset, 
              valid_dataset:BIODataset, 
              epochs:int=1):
        assert epochs > 1 and isinstance(epochs, int)
        print('Training ...')
        train_loss = 0.0
        #needed for early stopping
        epochs_no_improvement = 0
        best_f1 = 0.0
        #hiddens = self.model.lstm.init_hiddens(batch_size=1) this is only needed when using the model with variational dropout
        for epoch in range(epochs):

            if(False and epoch == 5): # I experimented with unfreezing embeddings after a few epochs but decided against it
                self.model.word_embedding.weight.requires_grad = True
            self.model.lstm.training = True
            
            print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            # for each batch 
            for step, sample in enumerate(train_dataset):
                inputs = sample['inputs']
                casing = sample['casing']
                labels = sample['outputs']
                pos = sample['pos']
                self.optimizer.zero_grad()
                labels = labels.view(-1)
                #if using coindtional random fields classifier:
                if self.crf:
                    tag_scores, tag_seq = self.model(inputs, casing, pos)#, hiddens)
                    sample_loss = -self.model.crf(tag_scores.unsqueeze(0), labels.unsqueeze(0), mask=None)
                #if using softmax classifier
                else:
                    predictions = self.model(inputs, casing, pos)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    sample_loss = self.loss_function(predictions, labels)
                    
                sample_loss.backward()
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss, valid_f1 = self.evaluate(valid_dataset)


            print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))
            print('  [E: {:2d}] f1 score = {:0.4f}'.format(epoch, valid_f1))

            #EARLY STOPPING, monitor the f1_score
            if valid_f1>best_f1:
                best_f1=valid_f1
                torch.save(self.model.state_dict(), "best_"+str(valid_f1))
                epochs_no_improvement=0
            else: epochs_no_improvement += 1
            #self.patience is the threshold of training epochs without improvement on the f1_score
            if epochs_no_improvement>self.patience:
                break

        print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    
    #self explanatory
    def evaluate(self, valid_dataset):
        self.model.lstm.training = False
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            outputs=[]
            total_labels=[]
            #hiddens = self.model.lstm.init_hiddens(1) this is only needed when using the model with variational dropout
            for sample in valid_dataset:
                inputs = sample['inputs']
                casing = sample['casing']
                pos = sample['pos']
                labels = sample['outputs']
                labels = labels.view(-1)
                if self.crf:
                    tag_scores, tag_seq = self.model(inputs, casing, pos)#, hiddens)
                    sample_loss = -self.model.crf(tag_scores.unsqueeze(0), labels.unsqueeze(0), mask=None)
                    tag_seq = [x[0] for x in tag_seq] #looks silly, but needed
                    outputs.append(self.decode_output(tag_seq))
                else:
                    predictions = self.model(inputs, casing, pos)
                    sample_loss = self.loss_function(predictions, labels) 
                    preds = []
                    for p in predictions:
                        preds.append(torch.argmax( torch.nn.functional.softmax(p)).item())
                    outputs.append(self.decode_output(preds))
                
                total_labels.append(self.decode_output([x.item() for x in labels]))
                valid_loss += sample_loss.tolist()
            #compute the f1 score using the seqeval f1_score function
            valid_f1 = f1_score(outputs, total_labels, average="macro", scheme=IOB2, mode='strict', zero_division=0)  
        #plot_confusion_matrix(y_pred=outputs, y_true=total_labels, normalize=True)  
        return valid_loss / len(valid_dataset), valid_f1
    