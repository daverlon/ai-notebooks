import os
import sys
from random import randint
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from model import MiniGoogLeNet
from data_preprocessing import load_batch

if __name__ == "__main__":

    modeltype = torch.float32
    device = torch.device("cpu")

    if torch.cuda.is_available():
       device = torch.device("cuda")
       modeltype = torch.double


    print(device)

    # just the eyes
    labels_npy = pd.read_csv("../dataset/train/training.csv")[["left_eye_center_x","left_eye_center_y", "right_eye_center_x", "right_eye_center_y"]].to_numpy()

    MODEL_PATH = "../nets/"

    N_EPOCHS, BS, LR, MOMENTUM, MODEL_NAME = sys.argv[1:]
    N_EPOCHS = int(N_EPOCHS)
    BS = int(BS)
    LR = float(LR)
    MOMENTUM = float(MOMENTUM)
    MODEL_PATH += MODEL_NAME

    N_SAMPLES = 7048
    N_BATCHES = int(N_SAMPLES/BS)

    model = MiniGoogLeNet()

    if os.path.exists(MODEL_PATH + '.pt'):
        print("Loaded model:", MODEL_PATH + '.pt')
        model.load_state_dict(torch.load(MODEL_PATH + '.pt', map_location="cpu"))
    else:
        print("Could not find model:", MODEL_PATH + '.pt')
        print("Creating new model.")

    model = model.to(device).type(modeltype)

    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOMENTUM)

    n_checkpoint = 0

    for epoch in tqdm(range(N_EPOCHS)):

        for batch in range(N_BATCHES):

            n_checkpoint += 1

            optim.zero_grad()

            # get batch
            batch_i = randint(0, N_SAMPLES-BS)
            x, y = load_batch(batch_i, BS, labels=labels_npy)
            x = x.to(device).type(modeltype)
            y = y.to(device).type(modeltype)

            forward = model(x)
            loss = loss_fn(forward, y)
            if torch.isnan(loss): 
                print("Loss = NaN, exiting.")
                exit(0)

            loss.backward()
            optim.step()

            # prog.set_description(f"Loss: {losses[len(losses)-1]}")
            print(f"Loss: {[loss.detach().cpu().item()/torch.Tensor([64]).type(modeltype), loss.detach().item()]}")
            torch.save(model.state_dict(), MODEL_PATH + '.pt')



        
