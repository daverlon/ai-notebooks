import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

from model import NiN
from data_preprocessing import prep_dataloader
from eval import do_eval

import argparse

MODEL_PATH = "./nets/"

if __name__ == "__main__":

  device = torch.device("cuda")

  MODEL_FILE, N_EPOCHS, BS, LR, MOMENTUM = [*sys.argv[1:]]
  MODEL_PATH += MODEL_FILE
  N_EPOCHS = int(N_EPOCHS)
  BS = int(BS)
  LR = float(LR)
  MOMENTUM = float(MOMENTUM)

  train_dataloader = prep_dataloader(BS, True)

  model = NiN()
  if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))

  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=MOMENTUM)

  for e in tqdm(range((N_EPOCHS))):
    for b, (x, y) in enumerate(train_dataloader):

      x = x.type(torch.float).to(device)
      y = y.to(device)

      optim.zero_grad()

      forward = model(x)
      loss = loss_fn(forward, y)

      acc = ((forward.detach().cpu().argmax(dim=1)==y.detach().cpu()).sum()/BS).item()

      loss.backward() 
      optim.step()

      if b % 100 == 0: 
        print(f"Epoch: {e+1}/{N_EPOCHS}, Batch: {b:4}/{int(60_000/BS)}, Loss: {loss.detach().cpu().item()/float(BS):6f}, Acc: {acc*100.0:4f}")

    print("Saving model to:", MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH) 

  do_eval(model)