import torch
import torch.nn as nn


class VGG_Conv_Block(nn.Module):
    def __init__(self, conv_channels=[]):
        super().__init__()
        
        layers = []
        for inc, outc in conv_channels:
            layers.append(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            
        self.layer_stack = nn.Sequential(*layers, nn.MaxPool2d(kernel_size=2, stride=2))
        
    def forward(self, x):
        return self.layer_stack(x)
    
    
class VGG_Linear_Block(nn.Module):
    def __init__(self, in_features=4096, out_features=4096):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.layer_stack(x)
    

class VGGnet(nn.Module):
    def __init__(self, conv_blocks=[], lin_blocks=[]):
        super().__init__()
        
        convs = []
        for block in conv_blocks:
            convs.append(VGG_Conv_Block((block[0], block[1])))   
        lins = []
        for block in lin_blocks:
            lins.append(VGG_Linear_Block(block[0], block[1]))    
        self.layer_stack = nn.Sequential(
            *convs,
            nn.Flatten(),
            *lins,
            nn.LogSoftmax(dim=1)
        )     
        
    def forward(self, x):
        return self.layer_stack(x)