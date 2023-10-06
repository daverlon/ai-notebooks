import numpy as np 
from torchvision.datasets import mnist
from model import TestNet
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD
from tinygrad.state import get_parameters
from tqdm import tqdm
import torch

def sparse_categorical_crossentropy(out, Y):
  num_classes = out.shape[-1]
  YY = Y.flatten().astype(np.int32)
  y = np.zeros((YY.shape[0], num_classes), np.float32)
  # correct loss for NLL, torch NLL loss returns one per row
  y[range(y.shape[0]),YY] = -1.0*num_classes
  y = y.reshape(list(Y.shape)+[num_classes])
  y = Tensor(y)
  return out.mul(y).mean()

if __name__ == "__main__":

    train_data = mnist.MNIST("../../datasets", train=True)
    test_data = mnist.MNIST("../../datasets", train=False)

    print(train_data.data[0].numpy())
    X_train = train_data.data.reshape(60000, 1, 784).numpy()
    y_train = train_data.targets.numpy().transpose()



    model = TestNet()
    optim = SGD(get_parameters(model), lr=0.001)

    samp_x = Tensor(X_train[:5])
    samp_y = y_train[:5]
    out = model.forward(samp_x)
    print(out.numpy())
    loss = sparse_categorical_crossentropy(out, samp_y)
    # print("test loss:", loss.numpy())
    exit()

    BS = 32
    losses = []

    for step in tqdm(range(1000)):
        ids = np.random.randint(0, 60000, size=(BS))

        # cannot index with a tensor
        x_batch = Tensor(X_train[ids], requires_grad=False)
        y_batch = y_train[ids]

        out = model.forward(x_batch)

        loss = sparse_categorical_crossentropy(out, y_batch)
        losses.append(loss.numpy())

        print("Loss:", loss.numpy())

        optim.zero_grad()

        loss.backward()

        optim.step()

        # print(f"{pred=}, {acc=}")

        
