import numpy as np
import pandas as pd
from tqdm import tqdm
from random import randint

import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

mnist_train = MNIST(root="../datasets",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None)

mnist_test = MNIST(root="../datasets",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None)

train_X, train_y = mnist_train.data.float(), mnist_train.targets
test_X, test_y = mnist_test.data.float(), mnist_test.targets

train_X = train_X.to(device)
train_y = train_y.to(device)
test_X = test_X.to(device)
test_y = test_y.to(device)

BS  = 32
train_dataloader = DataLoader(dataset=mnist_train, 
    batch_size=BS,
    shuffle=True)

test_dataloader = DataLoader(dataset=mnist_test,
    batch_size=BS,
    shuffle=False)

class model1(nn.Module):
  def __init__(self):
    super(model1, self).__init__()

    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784,out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64,out_features=32),
        nn.ReLU()
        )

    self.classifier = nn.Sequential(
        nn.Linear(in_features=32,out_features=10),
        nn.LogSoftmax(dim=1)
        )
  def forward(self, x):
    x = self.layer_stack(x)
    x = self.classifier(x)
    return x

mymodel = model1()
mymodel = mymodel.to(device)

def get_accuracy(y_pred, y_true):
  return (y_pred==y_true).sum()

n_epochs = 5

accuracies = []
losses = []

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=mymodel.parameters(), lr=0.1, momentum=0.0)

for epoch in range(n_epochs):
  print(f"Epoch: {epoch+1}")
  if len(losses) > 0 and len(accuracies)>0: 
    print(f"Loss: {losses[len(losses)-1]}")
    print(f"Accuracy: {accuracies[len(accuracies)-1]}")

  epoch_loss = 0
  epoch_accuracy = 0

  for n_batch, (X, y) in enumerate(train_dataloader):
    optim.zero_grad()

    preds = mymodel(X)
    loss = criterion(preds, y)

    epoch_loss+=loss.item()

    loss.backward()

    optim.step()

    # get accuracy for this batch
    guesses = preds.argmax(dim=1)
    batch_accuracy = get_accuracy(guesses, y)
    epoch_accuracy += batch_accuracy

    if n_batch % 200 == 0:
      print(f"--- Batch {n_batch} / {60000/32}")

losses.append(epoch_loss)
print(f"    Accumulated Loss: {epoch_loss}")
print(f"    Epoch Accuracy: {(epoch_accuracy/60000) * 100} %")      

