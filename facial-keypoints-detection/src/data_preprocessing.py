from PIL import Image
import torch
from torchvision.transforms import ToTensor

import numpy as np


t = ToTensor()


def load_batch(batch_i, BS, labels: np.ndarray =None):

    dataset_path = "../dataset/train/train_images/"

    image_names = [dataset_path+str(i)+".jpg" for i in range(batch_i, batch_i+BS)]

    image_tensor = torch.Tensor(size=[BS, 3, 96, 96])

    for i, name in enumerate(image_names):
        image_tensor[i] = t(Image.open(name))

    image_labels = t((labels[batch_i:batch_i+BS]).squeeze())

    return image_tensor, image_labels.squeeze()
