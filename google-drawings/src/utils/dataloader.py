import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image

import matplotlib.pyplot as plt


DATASET_PATH = "../../datasets/google-drawings/"
FILES = sorted([name for name in os.listdir(DATASET_PATH) if ".npy" in name]) # filter out any other files?
N_FILES = len(FILES)

def get_class(i):
    return FILES[i]


def load_batch(BS):
    
    batch = torch.zeros((BS, 1, 28, 28))
    labels = torch.zeros(BS).type(torch.uint8)

    file_indices = np.random.randint(0, N_FILES, BS)
    batch_indices = np.random.randint(0, 1000, BS)
    
    for i, (fi, bi) in enumerate(zip(file_indices, batch_indices)):
        item = np.load(DATASET_PATH+FILES[fi])[bi]
        batch[i] = torch.from_numpy(item).type(torch.float).reshape(28,28)
        labels[i] = torch.tensor(fi)
        
    return batch, labels

def load_sample(sample_path):
    img = Image.open(sample_path)
    sample = np.array(img)
    # plt.imshow(sample)
    # plt.show()
    # print(sample)

    t = torch.from_numpy(sample)

    return t
