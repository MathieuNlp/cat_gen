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
# manual_seed = 100
# print('Random Seed: ', manual_seed)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)

# global variables 

# latent vector size
z_size = 100
# generator feature map size
gen_f_size = 64
# discriminator feature map size
dis_f_size = 64
# number of epochs
num_epochs = 50
# learning rate
lr = 0.0001
beta1 = 0.5
show_data_sample = False

if __name__ == '__main__':

    print("is gpu available ?", torch.cuda.is_available())
    if torch.cuda.is_available() : 
        nb_gpu = 1
    else:
        nb_gpu = 0

    # Create dataloader
    dataloader = my_dataloader.dataloader
    device = my_dataloader.device
    if show_data_sample : my_dataloader.plot_sample(dataloader)

    # Create generator and discriminator
    generator = gan.Generator(z_size, gen_f_size)
    discriminator = gan.Discriminator(dis_f_size)
    # Weight initialization
    generator.apply(gan.weights_init)
    discriminator.apply(gan.weights_init)

    generator.to(device)
    discriminator.to(device)

    loss = nn.BCELoss()
    # Create batch of latent vectors (for visualisation)
    fixed_noise = torch.randn(64, z_size, 1, 1, device=device)

    # convention for components (in paper)
    real_label = 1
    fake_label = 0

    # Adam optimizer
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # training loop
    
    # rules to take care : https://github.com/soumith/ganhacks
    save_PATH = "./saved_model/"
    def train(dataloader, generator, discriminator, real_label, fake_label, fixed_noise, device, num_epochs):
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print('-'*5, 'Start Trainnig', '-'*5)
        for epoch in range(num_epochs) : 
            for i, data in enumerate(dataloader):
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ## Train with all-real batch
                discriminator.zero_grad()
                real_batch_device = data[0].to(device)
                b_size = real_batch_device.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = discriminator(real_batch_device).view(-1)
                # Calculate loss on all-real batch
                errD_real = loss(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, z_size, 1, 1, device=device)
                # Generate fake image batch with G
                fake = generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = loss(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients <=> no zero_grad()
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                # (2) Update G Network: maximize log(D(G(z)))
                generator.zero_grad()
                label.fill_(real_label) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = loss(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                
                # Training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plots
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check perfs of generator by saving generator's output on fixed_noise (def line 71)
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = generator(fixed_noise).detach().cpu()
                    img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

                iters+=1
            if (epoch % 10 == 0):
                torch.save(generator.state_dict(), save_PATH+f"new_generator_checkpoint_{epoch // 10}.pt")
                torch.save(discriminator.state_dict(), save_PATH+f"new_discriminator_checkpoint_{epoch // 10}.pt")
        return img_list, G_losses, D_losses
    

    img_list, G_losses, D_losses = train(dataloader, generator, discriminator, real_label, fake_label, fixed_noise, device, num_epochs)
    # Save the trained model
    
    #torch.save(generator.state_dict(), save_PATH)

    # Load model
    def load_and_generate(save_PATH, fixed_noise): 
        img_list = []
        trained_model = gan.Generator(z_size, gen_f_size)
        trained_model.load_state_dict(torch.load(save_PATH))
        fake = generator(fixed_noise).detach().cpu()
        img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))
    #load_and_generate(save_PATH, fixed_noise)

    # Print loss generator vs discriminator
    def plot_loss(G_loss, D_loss):
        
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_loss,label="Generator")
        plt.plot(D_loss,label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./plots/loss_function.png")
    plot_loss(G_losses, D_losses)

    # Visualisation of generator's progress
    def visu_progress(img_list):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())
    # Grab a batch of real images from the dataloader


    real_batch = next(iter(dataloader))

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
