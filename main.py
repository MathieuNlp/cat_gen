import torch
import argparse
import os
import random
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
import gan_model as gan

# Set random seed for reproducibility
manual_seed = 100
print('Random Seed: ', manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

print("is gpu available ?:", torch.cuda.is_available())
if torch.cuda.is_available() : 
    nb_gpu = 1
else:
    nb_gpu = 0

    
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


dataloader = my_dataloader.dataloader

# Implementation Part

# Weight initialization
# in the paper, all weights are initialized randomly with normal distrib. mean=0, std=0.02
def weights_init(m) : 
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 :
        nn.init.normal_(m.weight.data, 0.0, 0.02) 
    elif classname.find("BatchNorm") != -1 :
        nn.init.normal_(m.weight.data, 1, 0.02) 
        nn.init.constant_(m.bias.data, 0) # bias to 0

# generator
