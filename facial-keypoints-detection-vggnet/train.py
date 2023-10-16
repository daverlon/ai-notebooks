import math
import os
import sys
import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import load_batch

class Block(nn.Module):
    def __init__(self, inc, outc, k, s, p):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.layer_stack(x)
         
class DaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            Block(1, 24, 7, 1, 3), # -> 48x48
            Block(24, 12, 5, 1, 2), # -> 24x24
            Block(12, 12, 3, 1, 1), # -> 12x12
            nn.Flatten(),
            nn.Linear(in_features=12*12*12, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=4),
        )

    def forward(self, x):
        return(self.layer_stack(x))

class DaNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), # 48


        )

    def forward(self, x):
        x = self.layer_stack(x)
        #print(x.shape)
        return(x)



if __name__ == "__main__":

    device = torch.device("cuda")

    #model = DaNet().to(device)
    model = DaNet2().to(device)
    print(model)

    N_EPOCHS = 5
    N_SAMPLES = 7048
    BS = 64
    N_BATCHES = int(N_SAMPLES/BS)
    LR = 0.000001

    criterion = nn.MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):

        print(f"Epoch: {epoch}/{N_EPOCHS}")

        for n_batch in range(N_BATCHES):

            batch, labels = load_batch(BS)

            batch = batch.to(device)

            pred = model(batch)

            loss = criterion(pred, labels.to(device))
            loss_a = math.sqrt(loss.detach().cpu().item())
            
            #if n_batch % 5 == 0: print(f"     Batch: {n_batch}/{N_BATCHES} --- Loss: {loss_a}") 
            print(f"     Batch: {n_batch}/{N_BATCHES} --- Loss: {loss_a}") 


