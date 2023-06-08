import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# root directory of the dataset
dataroot = "../dataset"
#number of workers for dataloader
num_workers = 0
# batch size during training
batch_size = 64
# training image size
image_size = 64
#number of channels
num_channels = 3
    
# # transformation to the images
transform_input = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])

# create dataset
dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                        transform=transform_input)
#create data loader

dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        )
# device 
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") 

# plot examples
def plot_sample(dataloader, n_img=32):
    real_batch = next(iter(dataloader)) # take first batch of 64 images
    plt.figsize=(8,6)
    plt.axis("off")
    plt.title("Sample of training images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:n_img], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()
