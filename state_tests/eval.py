import sys
import os

import torch
import torch.nn

from model import NiN
from data_preprocessing import prep_dataloader

MODEL_PATH = "./nets/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def do_eval(model):
    # device = next(model.parameters()).device
    BS = 32
    test_dataloader = prep_dataloader(BS, False)
    score = 0
    for x, y in test_dataloader:
        x = x.type(torch.float).to(DEVICE)
        out = model(x)
        score += (out.detach().cpu().argmax(dim=1)==y).sum().item()
    print("Score:", score)
    return score
  
if __name__ == "__main__":

    MODEL_PATH += sys.argv[1]

    model = NiN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    
    do_eval(model.to(DEVICE))
  