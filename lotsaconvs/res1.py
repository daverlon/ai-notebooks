import time
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloader import batch_dataloader

class ResBlock(nn.Module):
    def __init__(self, inc, midc):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=midc, out_channels=midc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=midc, out_channels=inc, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
    def forward(self, x):
        y = self.layer_stack(x)
        return y + x

class DaNet(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.layer_stack = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            ResBlock(32, 64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            ResBlock(64, 96),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(48*7*7, 128),
            nn.LogSoftmax(dim=1)
        )
        self.path = path
        if os.path.exists(path):
            self.load_state_dict(torch.load(self.path))
            print("Loaded existing model:", self.path)
        else:
            torch.save(self.state_dict(), self.path)
            print("Created new model:", self.path)
    def forward(self, x):
        return self.layer_stack(x)

if __name__ == "__main__":

    BS = 256
    LR = 0.001
    MOM = 0.01

    N_EPOCHS = 5

    device = torch.device("mps")

    model = DaNet("nets/res5.pt").to(device)
    print(model)
    model.train()

    optim = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOM)
    criterion = nn.NLLLoss()

    losses = []
    accs = []

    start_time = time.time()
    for n_epoch in range(N_EPOCHS):
        for n_batch, (x, y) in enumerate(batch_dataloader(BS)):

            optim.zero_grad()

            x = x.to(device)
            pred = model(x)
            loss = criterion(pred, y.to(device))
            acc = (pred.detach().cpu().argmax(dim=1)==y).sum().item()/float(BS)
            losses.append(loss.detach().cpu().item()/BS)
            accs.append(acc)

            if n_batch % 10 == 0:
                print(f"Epoch: {n_epoch+1}, Batch: {n_batch} --- Loss: {loss.detach().cpu().item()/float(BS):20f}, Acc: {acc}")

            loss.backward()
            optim.step()

        print("Model saved:", model.path)
        plt.plot(losses)
        plt.plot(accs)
        plt.savefig("recent_plot.png")
        torch.save(model.state_dict(), model.path)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds.")