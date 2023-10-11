import os
import sys
from random import randint
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

COLUMNS = ["left_eye_center_x", "left_eye_center_y", "right_eye_center_x", "right_eye_center_y"]

LABELS_PATH = "../datasets/facial-keypoint-detection/training.csv"
DATASET_PATH = "../datasets/facial-keypoint-detection/images/train_images/"
FILES = os.listdir(DATASET_PATH)
N_FILES = len(FILES)

def load_batch(BS):

    file_indices = np.random.randint(0, N_FILES, BS)

    batch = torch.zeros((BS, 1, 96, 96)).type(torch.float)
    labels = torch.zeros((BS, 4)).type(torch.float)

    for i in range(BS):
        fi = file_indices[i]

        x = np.array(Image.open(DATASET_PATH + str(fi) + ".jpg").convert("L"))
        y = pd.read_csv(LABELS_PATH)[COLUMNS].to_numpy()[fi]

        batch[i] = torch.from_numpy(x)
        labels[i] = torch.from_numpy(y)
    
    return batch, labels

if __name__ == "__main__":
    print(load_batch(32))