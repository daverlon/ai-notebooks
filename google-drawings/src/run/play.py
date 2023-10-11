import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from PIL import Image
import numpy as np 

from utils.dataloader import load_sample, get_class

import matplotlib.pyplot as plt

def eval_model(model):

    model.eval()

    device = torch.device("mps")
    # model = model.to(device)

    sample = load_sample("sample.jpg").unsqueeze(dim=0).unsqueeze(dim=0).type(torch.float)
    plt.imshow(sample.squeeze()) 
    plt.show()

    pred = torch.exp(model(sample))

    print(pred)
    pred_i = pred.argmax()

    print(get_class(pred_i), pred[0,pred_i].item())


