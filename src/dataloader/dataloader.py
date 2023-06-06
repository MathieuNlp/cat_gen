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
#number of workers for dataloader
num_workers = 1
# batch size during training
batch_size = 128
# training image size
image_size = 64
#number of channels
num_channels = 3
    
# # transformation to the images
# transform_input = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

# # create dataset
# dataset = torchvision.datasets.ImageFolder(root=dataroot,
#                                         transform=transform_input)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size):
        super().__init__()
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]
        self.transform = transforms.Compose([
     transforms.Resize(image_size),
     transforms.CenterCrop(image_size),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
    
dataset  = Dataset(dataroot, image_size)
#create data loader

dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=num_workers)
# device 
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") 

# plot examples
def plot_sample(dataloader, n_img=64):
    real_batch = next(iter(dataloader))
    plt.figsize=(8,8)
    plt.axis("off")
    plt.title("Sample of training images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()