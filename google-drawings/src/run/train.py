import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dataloader import load_batch

def train_model(model_path, model, criterion, hyper_params):

    model.train()

    device = torch.device("mps")
    # model = model.to(device)

    if not os.path.exists(model_path):
        print("Created new model:", model_path)
        torch.save(model.state_dict(), model_path)
    else:
        print("Loaded existing model:", model_path)
        model.load_state_dict(torch.load(model_path))

    model = model.to(device)

    N_EPOCHS, BS, LR, MOM = [*hyper_params]

    optim = torch.optim.SGD(params=model.parameters(), lr=LR)

    # pbar = tqdm(range(N_EPOCHS))

    N_SAMPLES = 1000
    N_BATCHES = int(N_SAMPLES*128/BS)
    print(N_BATCHES)

    for epoch in range(N_EPOCHS):

        print(f"Epoch: {epoch}/{N_EPOCHS}")

        for n_batch in range(N_BATCHES):

            x, y = load_batch(BS)
            # print(x.shape)
            x = x.to(device)

            optim.zero_grad()

            forward = model(x)
            loss = criterion(forward, y.to(device))

            loss_amount = loss.detach().cpu().sum().item()/float(BS) 

            acc = (forward.argmax(dim=1).detach().cpu()==y).sum().item()/BS

            loss.backward()
            optim.step()

            if n_batch % 25 == 0: 
                # pbar.set_description(f"{loss_amount}, {acc}/{BS}")
                print(f"   Batch: {n_batch}/{N_BATCHES} --- {loss_amount}, {acc}")
            


        print("Saving model:", model_path)
        torch.save(model.state_dict(), model_path)
        

if __name__ == "__main__":

    x, y = load_batch(3)
    print(x.shape, y.shape)