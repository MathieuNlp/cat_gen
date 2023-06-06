import torch
import src.models.dcgan_model as gan
import src.dataloader.dataloader as my_dataloader
import torchvision
import matplotlib.pyplot as plt
import numpy as np

z_size = 100
gen_f_size = 64

model_PATH = "./saved_model/new_generator_checkpoint_4.pt"

dataloader = my_dataloader.dataloader
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") 
fixed_noise = torch.randn(64, z_size, 1, 1, device=device)

trained_generator = gan.Generator(z_size, gen_f_size)
trained_generator.load_state_dict(torch.load(model_PATH))

img_list=[]
real_batch = next(iter(dataloader))
fake = trained_generator(fixed_noise).detach().cpu()

img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

def get_real_vs_fake(real_batch, img_list):
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1][:64],(1,2,0)))
    plt.savefig("./plots/fake_vs_real.png")

get_real_vs_fake(real_batch, img_list)
