import torch
import torch.nn as nn

class Inception(nn.Module):
    # output channels 
    def __init__(self, in_channels: int, bc1: int, bc2: [], bc3: [], bc4: int):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=bc1, kernel_size=1),
            # nn.BatchNorm2d(bc1),
            nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=bc2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=bc2[0], out_channels=bc2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=bc3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=bc3[0], out_channels=bc3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=bc4, kernel_size=1)
        )

    def forward(self, x):
        return torch.concat(
            (self.b1(x), self.b2(x), self.b3(x), self.b4(x)),
            dim=1
        )
     
class MiniGoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1),
            # nn.ReLU(),
            nn.ReLU(),
            
            Inception(12, 32, [16, 32], [16, 32], 32),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x4 x48x48

            Inception(32*4, 64, [48, 64], [48, 64], 64),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x4 x24x24

            Inception(64*4, 32, [48, 32], [48, 32], 32),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x4 x12x12
            
            # Inception(32*4, 16, [24, 16], [24, 16], 24),
            
            # nn.MaxPool2d(kernel_size=2, stride=2), # 16x4 x6x6


            nn.Conv2d(in_channels=32*4, out_channels=36, kernel_size=1, stride=1),
            # nn.BatchNorm2d(36),
            
            nn.Conv2d(in_channels=36, out_channels=8, kernel_size=1, stride=1),
            # nn.BatchNorm2d(8),

            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1, stride=1),
            # nn.BatchNorm2d(3),
            
            nn.Flatten(),
            nn.Linear(in_features=3*12*12, out_features=4)
        )
    def forward(self, x):
        return self.layer_stack(x)