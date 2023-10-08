#!/usr/bin/env python3

import torch
import torch.nn as nn

class NiN_Block(nn.Module):
    def __init__(self, inc, outc, ks, stride, padding):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=ks, stride=stride, padding=padding),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),            
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.layer_stack(x)

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            NiN_Block(1, 24, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14
            NiN_Block(24, 10, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2), #7x7         
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),   
            #nn.Softmax(dim=1)
        )   
    def forward(self, x):
        return self.layer_stack(x)
