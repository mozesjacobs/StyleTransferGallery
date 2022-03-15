import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os


# Arguments
#
# fpath : path to the numpy file of data
# imsize : size to resize the image to
#
class ImgDataset(Dataset):

    def __init__(self, fpath, imsize, device):
        # load data
        self.imsize = imsize
        self.transform = transforms.Compose([
            transforms.Resize((imsize, imsize)),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor
        
        self.data = []
        for f in os.listdir(fpath):
            if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
                self.data.append(self.load_image(fpath + f, device)) 
        
    def load_image(self, image_name, device):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.transform(image).unsqueeze(0)
        return image.to(device, torch.float)
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    

    
    
    