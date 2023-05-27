import torch
import argparse
import os
import random
import torchvision
import torch.nn as nn
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

# global variables 
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
beta1 = 0.5

# Create dataloader
dataloader = my_dataloader.dataloader
device = my_dataloader.device

# Create generator 
generator = gan.Generator(nb_gpu, z_size, gen_f_size)
# Weight initialization
generator.apply(gan.weights_init)

# Create discriminator 
discriminator = gan.Discriminator(nb_gpu, dis_f_size)
# Weight initialization
discriminator.apply(gan.weights_init)

loss = nn.BCELoss()
# Create batch of latent vectors (for visualisation)
fixed_noise = torch.randn(64, z_size, 1, 1, device=device)

# convention for components (in paper)
real_label = 1
fake_label = 0

optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# training loop