import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import os
import matplotlib.pyplot as plt


def imshow(tensor, ax, title=None, fontsize=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze()      # remove the fake batch dimension
    image = unloader(image)
    ax.imshow(image)
    if title is not None:
        if fontsize is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=fontsize)
    #ax.pause(0.001) # pause a bit so that plots are updated