import torch
import models.dcgan_model as dcgan
import loader.dataloader as my_dataloader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

z_size = 100
gen_f_size = 64
batch_size = 64

model_PATH = "./saved_model/generator_checkpoint_120.pt"
device = 'cpu'
fixed_noise = torch.randn(batch_size, z_size, 1, 1, device=device)

trained_generator = dcgan.Generator(z_size, gen_f_size)
trained_generator.load_state_dict(torch.load(model_PATH))
img_list=[]
fake = trained_generator(fixed_noise).detach().cpu()
img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

def plot_sample(img_list, n_img=32):
    # take first batch of 64 images
    plt.figsize=(8,6)
    plt.axis("off")
    plt.title("Sample from the generator")
    plt.imshow(np.transpose(torchvision.utils.make_grid(img_list[:n_img], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()

plot_sample(img_list, n_img=32)