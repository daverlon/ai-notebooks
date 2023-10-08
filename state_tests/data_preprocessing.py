from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

def prep_dataloader(BS: int, train: bool):
  dataloader_train = DataLoader(
    FashionMNIST(root="../datasets", train=train, download=False, transform=ToTensor()),
    batch_size=BS, shuffle=train
  )
  return dataloader_train