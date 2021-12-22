import torch
from torch import nn
import torch.utils.data as data_utils
import torch.optim as optim
from data import SEQ_LEN,SEQ_SKIP, BATCH_SIZE
from functions import encode
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    dev = "cuda:0"
else:
	dev = "cpu"
print(dev)

train_loader = torch.load('train_loader.pt')
test_loader = torch.load('test_loader.pt')

device = torch.device(dev)

# Paramètres d'apprentissage
NB_ITER = 1

print(train_loader.shape)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.lstm1 = nn.LSTM(NB_CHARS, 256, batch_first = True)
        self.drop = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(256,256, batch_first = True)
        self.dense = nn.Linear(256,NB_CHARS)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        output_lstm1,_= self.lstm1(x)
        output_dropout1 = self.drop(output_lstm1)
        output_lstm2,_ = self.lstm2(output_dropout1)
        
        output_last = output_lstm2[:,-1,:]
        
        output_dropout2 = self.drop(output_last)
        output_dense = self.dense(output_dropout2)
        logits = self.softmax(output_dense)
        
        return logits
        

model = NeuralNetwork().to(device)
print(model)

nb_of_batch = (len(train_loader))


loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.05)

Loss = np.zeros((NB_ITER,nb_of_batch))

def train_loop(dataloader, model, loss_fn, optimizer, t):
    model.train()
    
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X).float()

        loss = loss_fn(pred, y.float())


        # Lr[t,batch] = scheduler.get_last_lr()[0]
        Loss[t,batch] = loss.item()


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    model.eval()
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X).float()
            test_loss += loss_fn(pred, y).item()

    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train():
    for t in range(NB_ITER):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader,model, loss_function, optimizer, t)
        test_loop(test_loader, model, loss_function)
        # torch.save(model, f'./model2_epoch_{t}.pt')
    print("Done!")

