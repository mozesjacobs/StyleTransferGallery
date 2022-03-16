import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def imshow(tensor, ax, title=None, fontsize=None):
    unloader = transforms.ToPILImage()e
    image = tensor.cpu().clone()
    image = image.squeeze()      
    image = unloader(image)
    ax.imshow(image)
    if title is not None:
        if fontsize is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=fontsize)