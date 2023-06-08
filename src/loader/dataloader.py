import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import yaml

class MyDataloader():
    """
    GetDataloader class takes the configuration file parameters load the dataset then 
    return the dataset wrapped in the dataloader

    Arguments:
        config: configuration file
    Return:
    dataloader: with function get_loader, return the dataloader
    """
    def __init__(self, config):
        self.config = config
        # root directory of the dataset
        self.dataroot = config["DATA"]["DATA_ROOT_PATH"]
        # batch size during training
        self.batch_size = config["DATA"]["BATCH_SIZE"]
        # training image size
        self.image_size = config["DATA"]["IMAGE_SIZE"]
        #number of channels
        self.num_channels = config["DATA"]["NUM_CHANNELS"]
        
        # transformation to the images
        self.transform_input = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
    def get_loader(self):
        # create dataset
        self.dataset = torchvision.datasets.ImageFolder(root=self.dataroot,
                                                transform=self.transform_input)
        #create data loader
        self.dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                )

        return self.dataloader
    
                                            