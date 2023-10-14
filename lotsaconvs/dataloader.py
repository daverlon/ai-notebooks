import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_PATH = "../datasets/google-drawings/"
N_SAMPLES_PER_FILE = 1000
FILE_NAMES = [f for f in os.listdir(DATASET_PATH) if ".npy" in f]
N_FILES = len(FILE_NAMES)
N_SAMPLES = N_SAMPLES_PER_FILE * N_FILES

def file_name(file_index: int):
  return DATASET_PATH + FILE_NAMES[file_index]

def batch_dataloader(BS: int):
  N_BATCHES = int((N_FILES*N_SAMPLES_PER_FILE)/BS)
  for n_batch in range(N_BATCHES):
    file_ixs = np.random.randint(0, N_FILES, (BS,2))
    file_ixs[:,1] = np.random.randint(0, N_SAMPLES_PER_FILE, BS)
    x = torch.zeros(BS, 1, 28, 28)
    y = torch.zeros(BS).type(torch.int64)
    for a, i in enumerate(file_ixs):
      x[a] = torch.from_numpy(np.load(file_name(i[0]))[i[1]]).reshape((28,28)).type(torch.float)
      y[a] = i[0]
    yield x, y

if __name__ == "__main__":

  for i, (x, y) in enumerate(batch_dataloader(32)):
    #print(i, x.shape, y.shape)
    print(F.one_hot(torch.tensor(y.argmax()), 128))
    


