import time
import os
import sys
import torch
import torch.nn as nn

from dataloader import batch_dataloader

class ResBlock(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
    def forward(self, x):
        y = self.layer_stack(x)
        # print(y.shape)
        return y + x

class DaNet(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.layer_stack = nn.Sequential(
            ResBlock(1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(7*7*64, 128),
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

    BS = 128
    LR = 0.0001
    MOM = 0.0

    N_EPOCHS = 5

    device = torch.device("mps")

    model = DaNet("nets/res1.pt").to(device)
    print(model)
    model.train()

    optim = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOM)
    criterion = nn.NLLLoss()

    start_time = time.time()
    for n_epoch in range(N_EPOCHS):
        for n_batch, (x, y) in enumerate(batch_dataloader(BS)):

            optim.zero_grad()

            x = x.to(device)
            pred = model(x)
            loss = criterion(pred, y.to(device))
            acc = (pred.detach().cpu().argmax(dim=1)==y).sum().item()/float(BS)

            if n_batch % 10 == 0:
                print(f"Epoch: {n_epoch+1}, Batch: {n_batch} --- Loss: {loss.detach().cpu().item()/float(BS):20f}, Acc: {acc}")

            loss.backward()
            optim.step()

        print("Model saved:", model.path)
        torch.save(model.state_dict(), model.path)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds.")