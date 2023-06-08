import torch
import argparse
import os
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import loader.dataloader as mydl
import models.dcgan_model as dcgan
import utils


with open("./config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

# Create dataloader
loader = mydl.MyDataloader(config)
dataloader = loader.get_loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config["DATA"]["SHOW_TRAIN_SAMPLE"] : utils.plot_sample(config, dataloader, device)

# Create generator and discriminator
generator = dcgan.Generator(config)
generator.apply(dcgan.weights_init)

discriminator = dcgan.Discriminator(config)
discriminator.apply(dcgan.weights_init)

generator.to(device)
discriminator.to(device)

loss = nn.BCELoss()
# Create batch of latent vectors (for visualisation)
fixed_noise = torch.randn(config["DATA"]["BATCH_SIZE"], config["GENERATOR"]["Z_SIZE"], 1, 1, device=device)

# convention for components (in paper)
real_label = 1
fake_label = 0

# Adam optimizer
optimizerG = optim.Adam(generator.parameters(), lr=float(config["GENERATOR"]["GENERATOR_LR"]), betas=(config["TRAIN"]["BETA1"], 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=float(config["DISCRIMINATOR"]["DISCRIMINATOR_LR"]), betas=(config["TRAIN"]["BETA1"], 0.999))

# training loop
# rules to take care : https://github.com/soumith/ganhacks

def train(config, dataloader, generator, discriminator, real_label, fake_label, fixed_noise, device):
    """
    Train the generator and discriminator on the dataset wrapped by the dataloader

    Arguments:
        config: config file
        dataloader: dataloader wrapping the dataset
        generator: model of the generator
        discriminator: model of the discriminator
        real_label: equals to 1, label for the classification of images from the dataset
        fake_label: equals to 0, label for the classification of images from the generator
        fixed_noise: tensor z, which is gaussian, representing the latent space variable
        device: device where we loaded models and data

    Return:
        img_list: list of images batches after every 100 iterations (iteration is training on a batch)
        G_losses: list of the generator loss during training
        D_losses:list of the discriminator loss during training
    """
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    num_epochs = config["TRAIN"]["NUM_EPOCHS"]
    print('-'*15, 'Start Trainnig', '-'*15)
    for epoch in range(num_epochs) : 
        for batch, data in enumerate(dataloader):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            discriminator.zero_grad()
            real_batch_device = data[0].to(device) # take the image tensor of size: batch_size,3,128,128
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
            noise = torch.randn(b_size, config["GENERATOR"]["Z_SIZE"], 1, 1, device=device)
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
            
            # Training stats every 100 batches passed
            if batch % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, batch, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plots
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check perfs of generator by saving generator's output on fixed_noise every 100 batches
            if (iters % 100 == 0) or ((epoch == num_epochs-1) and (batch == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

            iters+=1

        if (epoch % 25 == 0):
            torch.save(generator.state_dict(), config["SAVE"]["SAVE_MODEL_PATH"] + f"generator_checkpoint_{epoch}.pt")

    return img_list, G_losses, D_losses

img_list, G_losses, D_losses = train(config, dataloader, generator, discriminator, real_label, fake_label, fixed_noise, device)
torch.save(img_list)
utils.plot_loss(config, G_losses, D_losses)
#utils.plot_visu_progress(img_list)
utils.plot_real_vs_fake(config, dataloader, img_list, device)







