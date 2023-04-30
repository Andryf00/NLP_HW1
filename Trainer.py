#The trainer class was adapted from the one presented on the colab notebook
import torch
from torch import nn
from Dataset import BIODataset
from torchtext.vocab import Vocab
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2 

import warnings
warnings.filterwarnings('always')

class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        label_vocab: Vocab,
        log_steps:int=10_000,
        log_level:int=2,
        patience=5):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.best_f1 = 0.0
        self.patience = patience
    
    
    def decode_output(self, x):
        return [self.label_vocab.itos[int(idx)] for idx in x]

    def train(self, train_dataset:BIODataset, 
              valid_dataset:BIODataset, 
              epochs:int=1):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            valid_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.
            epochs: the number of times to iterate over train_dataset.

        Returns:
            avg_train_loss: the average training loss on train_dataset over
                epochs.
        """
        assert epochs > 1 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss = 0.0
        epochs_no_improvement = 0
        hiddens = self.model.lstm.init_hiddens(batch_size=1)
        for epoch in range(epochs):
            self.model.lstm.training = True
            if self.log_level > 0:
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
                tag_scores, tag_seq = self.model(inputs, casing, pos, hiddens)
                sample_loss = -self.model.crf(tag_scores.unsqueeze(0), labels.unsqueeze(0), mask=None)
                """
                predictions = self.model(inputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                sample_loss = self.loss_function(predictions, labels)
                """
                sample_loss.backward()
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                if((step+1)%1000==0):
                    print(step, epoch_loss/step)

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))

                #self.evaluate(valid_dataset)
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss, valid_f1 = self.evaluate(valid_dataset)


            if self.log_level > 0:
                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))
                print('  [E: {:2d}] f1 score = {:0.4f}'.format(epoch, valid_f1))

            #EARLY STOPPING
            if valid_f1>self.best_f1:
                self.best_f1=valid_f1
                torch.save(self.model.state_dict(), "best_"+str(valid_f1))
                epochs_no_improvement=0
            else: epochs_no_improvement += 1
            if epochs_no_improvement>self.patience:
                break

        if self.log_level > 0:
            print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    

    def evaluate(self, valid_dataset):
        self.model.lstm.training = False
        """
        Args:
            valid_dataset: the dataset to use to evaluate the model.

        Returns:
            avg_valid_loss: the average validation loss over valid_dataset.
        """
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            outputs=[]
            total_labels=[]
            for sample in valid_dataset:
                inputs = sample['inputs']
                casing = sample['casing']
                pos = sample['pos']
                labels = sample['outputs']
                labels = labels.view(-1)
                hiddens = self.model.lstm.init_hiddens(1)
                tag_scores, tag_seq = self.model(inputs, casing, pos, hiddens)
                sample_loss = -self.model.crf(tag_scores.unsqueeze(0), labels.unsqueeze(0), mask=None)
                tag_seq = [x[0] for x in tag_seq]
                """
                predictions = self.model(inputs)
                sample_loss = self.loss_function(predictions, labels) 
                preds = []
                for p in predictions:
                    preds.append(torch.argmax(p).item())
                outputs.append(self.decode_output(preds))
                """
                outputs.append(self.decode_output(tag_seq))
                total_labels.append(self.decode_output([x.item() for x in labels]))
                valid_loss += sample_loss.tolist()
            valid_f1 = f1_score(outputs, total_labels, average="macro", scheme=IOB2, mode='strict', zero_division=0)  
            
        return valid_loss / len(valid_dataset), valid_f1
    