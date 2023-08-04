import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

from tqdm import tqdm

from random import randint

def unpickle(file):
  import pickle
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    return dict

def load_batches():

  Y = np.array([])
  X = np.array([])

  batches = []
  for i in range(5): batches.append(unpickle(f"datasets/cifar-10-batches-py/data_batch_{i+1}"))

  for batch in batches:
    y = batch[b'labels']
    x = batch[b'data']

    Y = np.concatenate([Y, y])
    X = np.append(X, x)

  X = X.reshape(50_000, 3072)

  # test
  i = randint(0, 50)
  assert ((X[i]-batches[0][b'data'][i]).sum()) == 0
  assert Y[i] == batches[0][b'labels'][i]

  # subtract per pixel mean
  X = X.reshape(50000, 32, 32, 3)
  mean_per_channel = np.mean(X, axis=(0, 1, 2))
  X = X - mean_per_channel

  X = X.reshape(50000, 3, 32, 32)

  X = torch.from_numpy(X.astype(np.float32))
  Y = torch.from_numpy(Y.astype(np.float32))

  return X, Y

class PlainResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1)
    self.relu = ReLU(inplace=True)
    self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x)
    res = x
    out = self.conv1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = bn(out)
    out += res
    out = self.relu(out)
    return x

class PlainResNet(nn.Module):
  def __init__(self, n):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), stride=1, padding=1)
    self.relu = nn.ReLU(inplace=True)
    self.layers

  def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
    layers = []
    layers.append(PlainResBlock(in_channels, out_channels, stride=stride))
    for _ in range(1, num_blocks):
      layers.append(PlainResBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    return x

  
if __name__ == "__main__":
  X, y = load_batches()
  print("x:", X.shape)
  print("y:", y.shape)
