import os
import sys
import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import load_batch
from model import VGGnet

if __name__ == "__main__":

    device = torch.device("mps")

    conv_blocks = [
        [(1, 12), (12, 12)], # 48x48 x12
        [(12, 24), (24, 32)], # 24x24 x32
    ]

    lin_blocks = [
        [18432, 2048],
        [2048, 1024],
        [1024, 512],
        [512, 512],
        [512, 256],
        [256, 4]
    ]

    model = VGGnet(conv_blocks, lin_blocks).to(device)

    N_EPOCHS = 5
    N_SAMPLES = 7048
    BS = 128
    N_BATCHES = int(N_SAMPLES/BS)
    LR = 0.0001

    criterion = nn.MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):

        print(f"Epoch: {epoch}/{N_EPOCHS}")

        for n_batch in range(N_BATCHES):

            batch, labels = load_batch(BS)

            batch = batch.to(device)

            pred = model(batch)

            loss = criterion(pred, labels.to(device))
            loss_a = loss.detach().cpu().item()/BS
            
            if n_batch % 5 == 0: print(f"     Batch: {n_batch}/{N_BATCHES} --- Loss: {loss_a}") 