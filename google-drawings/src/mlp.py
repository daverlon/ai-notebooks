import os
import sys

import torch
import torch.nn as nn

from run.train import train_model
from run.play import eval_model

class MLP(nn.Module):
    def __init__(self, layers=[]):
        super().__init__()
        layers = [(nn.Linear(int(i1), int(i2)), nn.ReLU()) for (i1, i2) in zip(layers[0:], layers[1:])]
        self.layer_stack = nn.Sequential(nn.Flatten())
        for l in layers:
            self.layer_stack.append(l[0])
            self.layer_stack.append(l[1])
        self.layer_stack.append(nn.LogSoftmax(dim=1))
        print(self)

    def forward(self, x):
        return self.layer_stack(x)



if __name__ == "__main__":
    

    model_path = sys.argv[2]
    layer_arg = sys.argv[3].split(",") # example: "128,64,64,10"
    layers = [int(l) for l in layer_arg]

    if sys.argv[1] == "train":
        # python3 mlp.py "train" "nets/ok.pt" "784,512,512,256,256,256,256,256,256,128" "200,128,0.001,0.0"

        print("Training MLP Model")

        # epoch, bs, lr, mom
        hyper_params = [int(item) if item.isdigit() else float(item) for item in sys.argv[4].split(",")]

        model = MLP(layers)

        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()

        train_model(model_path, model, criterion, hyper_params)

    elif sys.argv[1] == "play":
        # python3 mlp.py "play" "nets/ok.pt" "784,512,512,256,256,256,256,256,256,128"

        print("Testing MLP Model")

        model = MLP(layers)
        if not os.path.exists(model_path):
            print("Cannot find model:", model_path)
            exit(1)
        else:
            print("Loaded model:", model_path)
            model.load_state_dict(torch.load(model_path))

        eval_model(model)


  

