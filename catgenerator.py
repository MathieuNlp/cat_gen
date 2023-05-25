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

# create dataset
transform_input = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                           transform=transform_input)

#create data loader
dataloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)
# device 
device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 1) else "cpu") 

# plot examples
def plot_sample(dataloader, n_img=64):
    real_batch = next(iter(dataloader))
    plt.figsize=(8,8)
    plt.axis("off")
    plt.title("Sample of training images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:2], padding=2, normalize=True).cpu(), (1,2,0)))
