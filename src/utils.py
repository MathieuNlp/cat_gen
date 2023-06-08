import matplotlib.pyplot as plt
import torchvision 
import numpy as np
import matplotlib.animation as animation
import torch

def plot_sample(config, dataloader, device, n_img=32):
    """
    Plot a grid of 32 images from the first batch of the dataloader

    Arguments:
        config: config file 
        dataloader: dataloader loaded with the dataset
        device: device where the data was sent

    Return:
        Save a plot in ./plots folder of 32 samples of the dataset
    """
    path_plots = config['SAVE']['SAVE_PLOTS_PATH']
    real_batch = next(iter(dataloader)) # take first batch of 64 images
    plt.figsize=(8,6)
    plt.axis("off")
    plt.title("Sample of training images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:n_img], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.savefig(config['SAVE']['SAVE_PLOTS_PATH'] + "sample_from_dataset")


def plot_loss(config, G_loss, D_loss):
    """
    Plot a grid of the loss of generator and discriminator during training

    Arguments:
        config: config file
        G_loss: list of loss from generator 
        D_loss: list of loss from discriminator 

    Return:
        Save a plot in ./plots folder of the loss
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_loss,label="Generator")
    plt.plot(D_loss,label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config['SAVE']['SAVE_PLOTS_PATH'] + "loss_D_&_G.png")

def plot_real_vs_fake(config, dataloader, img_list, device):
    """
    Given the list of image generated from the generator every 100 batches and generates a comparison of data from real dataset and generated images
    from the generator

    Arguments:
        config: config file
        dataloader: dataloader wrapping the dataset
        img_list: list of images generated from generator every 100 batches where z vector is the same
        device: device where we loaded the model
    
    Return:
        Save the comparison plot: fake images from generator vs real images from dataset in the folder ./plots
    """
    real_batch = next(iter(dataloader))
    # Plot the real images
    plt.figure(figsize=(16,12))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:32], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(img_list[-1].to(device)[:32], padding=5, normalize=True).cpu(), (1,2,0)))
    plt.savefig("./plots/real_vs_fake.png")


def plot_visu_progress(config, img_list):
    """
    Function that takes the list of image generated from the generator every 100 batches and generates an animation

    Arguments:
        img_list: list of images generated from generator every 100 batches
    
    Return:
        Save the animation in the folder ./plots
    """
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=100, blit=True)
    writervideo = animation.FFMpegWriter(fps=60)
    ani.save(config["SAVE"]["SAVE_PLOTS_PATH"]+'generation.mp4', writer=writervideo)


def plot_generate_fake(config, generator, device, n_img=64):
    # take first batch of 64 images
    img_list = []
    fixed_noise = torch.randn(config["DATA"]["BATCH_SIZE"], config["GENERATOR"]["Z_SIZE"], 1, 1, device=device)
    fake = generator(fixed_noise).detach().cpu()
    plt.figsize=(8,6)
    plt.axis("off")
    plt.title("Sample from the generator")
    plt.imshow(np.transpose(torchvision.utils.make_grid(fake[:n_img], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.savefig(config["SAVE"]["SAVE_PLOTS_PATH"] + "sample_from_generator.png")
    plt.show()