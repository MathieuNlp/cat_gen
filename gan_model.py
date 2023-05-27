import torch
import torch.nn as nn

# # number of channels 
# num_channels = 3
# # latent vector size
# z_size = 10
# # generator feature map size
# gen_f_size = 64
# # discriminator feature map size
# dis_f_size = 64

class Generator(nn.Module) :
    def __init__(self, nb_gpu, z_size, gen_f_size):
        super(Generator, self).__init__()
        self.nb_gpu = nb_gpu
        self.z_size = z_size
        self.gen_f_size = gen_f_size

        self.main = nn.Sequential(
            # input z latent variable 
            nn.ConvTranspose2d(self.z_size, self.gen_f_size*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.gen_f_size*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.gen_f_size*8, self.gen_f_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gen_f_size*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.gen_f_size*4, self.gen_f_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gen_f_size*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.gen_f_size*2, self.gen_f_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.gen_f_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.gen_f_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # output of size 3x64x64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nb_gpu, dis_f_size):
        super(Discriminator, self).__init__()
        self.nb_gpu = nb_gpu
        self.dis_f_size = dis_f_size
        self.main = nn.Sequential(
            nn.Conv2d(3, self.dis_f_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dis_f_size, self.dis_f_size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_f_size*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dis_f_size*2, self.dis_f_size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_f_size*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.dis_f_size*4, self.dis_f_size*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_f_size*8),
            nn.LeakyReLU(0.2, inplace=True),

            #output size disciminator_feature_map_size*8 x 4 x 4
            nn.Conv2d(self.dis_f_size, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)