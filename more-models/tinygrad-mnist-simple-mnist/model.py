import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

class TestNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=True)
        self.l2 = Linear(128, 10, bias=True)

    def forward(self, x: Tensor):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        x = x.relu()
        x = x.log_softmax(axis=1)
        return x


