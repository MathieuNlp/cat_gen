import numpy as np
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.utils.data
import torchvision
from PIL import Image
from tqdm import tqdm

from src.models.stylegan_model import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty, DiscriminatorLoss, GeneratorLoss


def cycle_dataloader(data_loader):
    """
    ## Cycle Data Loader

    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            
            yield batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size):
        super().__init__()
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size,image_size)),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)
    

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
    
class Configs():
    def __init__(self):
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        
        self.discriminator = Discriminator
        self.generator = Generator
        self.mapping_network = MappingNetwork

        self.discriminator_loss = DiscriminatorLoss
        self.generator_loss = GeneratorLoss

        self.discriminator_optimizer = torch.optim.Adam
        self.generator_optimizer = torch.optim.Adam
        self.gradient_penalty = GradientPenalty()
        self.gradient_penalty_coefficient = 10

        self.loader = Iterator
        self.batch_size = 32
        self.d_latent = 512
        self.image_size = 64
        self.mapping_network_layers = 8
        self.log_resolution = int(np.log2(self.image_size))

        self.learning_rate = 1e-3
        self.mapping_network_learning_rate = 1e-5
        self.gradient_accumulate_steps = 1
        self.adam_betas = (0.0, 0.99)
        self.style_mixing_prob = 0.9
        self.training_steps = 150_000

        self.lazy_gradient_penalty_interval = 4
        self.lazy_path_penalty_interval = 32
        self.lazy_penalty_after = 5_000
        #self.log_generated_interval = 200
        self.save_checkpoint_interval = 2_000
        self.log_layer_outputs = False
        
        self.dataset_path = '../data/cats'
        self.epochs = 100

    def initialize(self):
        
        dataset  = Dataset(self.dataset_path, self.image_size)
        
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 pin_memory=True)
        self.loader = cycle_dataloader(dataloader)

        
        self.discriminator = Discriminator(self.log_resolution).to(self.device)
        self.generator = Generator(self.log_resolution, self.d_latent).to(self.device)

        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                                        lr=self.learning_rate,
                                                        betas=self.adam_betas)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), 
                                                        lr=self.learning_rate,
                                                        betas=self.adam_betas)
        self.mapping_network_optimizer = torch.optim.Adam(self.mapping_network.parameters(), 
                                                        lr=self.mapping_network_learning_rate,
                                                        betas=self.adam_betas)
        
    def get_w(self, batch_size):
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device) 

            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)

            return torch.cat((w1, w2), dim=0)
        
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)
        
    def get_noise(self, batch_size):
        noise = []
        resolution = 4
        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            noise.append((n1,n2))
            resolution *= 2
        return noise
    

    def generate_images(self, batch_size):
        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)
        images = self.generator(w, noise)
        return images, w
    
    def training_single_step(self, idx):
        # Part Discriminator
        self.discriminator_optimizer.zero_grad()
        for i in range(self.gradient_accumulate_steps):
            generated_images, _ = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images.detach())
            real_images = next(self.loader).to(self.device)

            if (idx +1)%self.lazy_gradient_penalty_interval == 0:
                real_images.requires_grad_()
            real_output = self.discriminator(real_images)

            real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
            discri_loss = real_loss + fake_loss

            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                gp = self.gradient_penalty(real_images, real_output)
                discri_loss = discri_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval
            discri_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.discriminator_optimizer.step()

        # Part Generator
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()
        for i in range(self.gradient_accumulate_steps):
            generated_images, w = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images)
            gen_loss = self.generator_loss(fake_output)

            if idx > self.lazy_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                plp = self.path_length_penalty(w, generated_images)
                if not torch.isnan(plp):
                    gen_loss = gen_loss + plp

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()
    

    def train(self):
        for i in tqdm(range(self.training_steps)):
            self.training_single_step(i)


def main():
    configs = Configs()
    configs.initialize()
    configs.train()

if __name__ == '__main__':
    main()






    












        





