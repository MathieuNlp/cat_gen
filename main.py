import torch

print(torch.cuda.is_available())


import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


import dataloader as my_dataloader
# Set random seed for reproducibility
manual_seed = 100
print('Random Seed: ', manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# root directory of the dataset
dataroot = "data"
#number of workers for dataloader
num_workers = 1
# batch size during training
batch_size = 128
# training image size
image_size = 64
# number of channels 
num_channels = 3
# latent vector size
z_size = 10
# generator feature map size
gen_f_size = 64
# discriminator feature map size
dis_f_size = 64
# number of epochs
num_epochs = 5
# learning rate
lr = 0.0001
beta = 0.5
n_gpu = 1

dataloader = my_dataloader.dataloader

# Implementation Part

# Weight initialization
def weights_init(m) : 
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 :
        nn.init.normal_(m.weight.data, 0.0, 0.02) # in the paper mean=0, std=0.02
    elif classname.find("BatchNorm") != -1 :
        nn.init.normal_(m.weight.data, 1, 0.02) # batch norm params
        nn.init.constant_(m.bias.data, 0) # bias to 0

