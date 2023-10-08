import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

from model import NiN
from data_preprocessing import prep_dataloader
from eval import do_eval

MODEL_PATH = "nets/"
BS = 32
N_EPOCHS = 5

if __name__ == "__main__":

  device = torch.device("cuda")

  dataloader_train = prep_dataloader(BS, True)

  MODEL_PATH += sys.argv[1]

  model = NiN()
  if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))

  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.SGD(params=model.parameters(), lr=0.1)

  for e in tqdm(range(N_EPOCHS)):
    for b, (x, y) in enumerate(dataloader_train):

      x = x.to(device)
      y = y.to(device)

      optim.zero_grad()

      forward = model(x)
      loss = loss_fn(forward, y)

      acc = ((forward.detach().cpu().argmax(dim=1)==y.detach().cpu()).sum()/BS).item()

      loss.backward() 
      optim.step()

      if b % 50 == 0: print("Epoch:", e, "Batch:", b, "Loss:", loss.detach().cpu().item()/float(BS), "Acc:", acc*100.0)

    print("Saving model to:", MODEL_PATH)
    torch.save(model.state_dict(), MODEL_PATH) 

  do_eval(model)