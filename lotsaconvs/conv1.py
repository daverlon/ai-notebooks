import time
import os
import sys
import torch
import torch.nn as nn

from dataloader import batch_dataloader

class DaNet(nn.Module):
  def __init__(self, path):
    super().__init__()
    self.layer_stack = nn.Sequential(

      nn.Conv2d(in_channels=1, out_channels=48, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=48, out_channels=48, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=48, out_channels=72, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(in_channels=72, out_channels=72, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=72, out_channels=72, kernel_size=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=72, out_channels=96, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Flatten(),
      nn.Linear(in_features=64*3*3, out_features=128),

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

  # python3 net1.py 256 0.01 0.0 "nets/big.pt"
  BS, LR, MOM, MODEL_PATH = sys.argv[1:]
  BS = int(BS)
  LR = float(LR)
  MOM = float(MOM)

  N_EPOCHS = 5

  device = torch.device("mps")

  model = DaNet(MODEL_PATH).to(device)
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

