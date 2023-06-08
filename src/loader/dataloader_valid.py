import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# root directory of the dataset
dataroot = "../data/cats"
# batch size during training
batch_size = 64
# training image size
image_size = 64
#number of channels
num_channels = 3
    
# # transformation to the images
transform_input = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# create dataset
dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                        transform=transform_input)
#create data loader
train_set, valid_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(train_set, 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        )

test_loader = torch.utils.data.DataLoader(valid_set, 
                                        batch_size=batch_size,
                                        shuffle=True,
                                        )
# device 
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") 

# plot examples
def plot_sample(train_loader, test_loader, n_img=32):
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(train_batch[0].to(device)[:n_img], padding=2, normalize=True).cpu(), (1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Test Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(test_batch[0].to(device)[:n_img], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()
    
#plot_sample(train_loader, test_loader)
