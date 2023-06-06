import torch
import torch.nn as nn


# in the paper, all weights are initialized randomly with normal distrib. mean=0, std=0.02
def weights_init(m) : 
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 :
        nn.init.normal_(m.weight.data, 0.0, 0.02) 
    elif classname.find("BatchNorm") != -1 :
        nn.init.normal_(m.weight.data, 1, 0.02) 
        nn.init.constant_(m.bias.data, 0) # bias to 0

def gen_block(in_channels, out_channels):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            )

def discri_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            )
    

class Generator(nn.Module) :
    def __init__(self, z_size, gen_f_size):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.gen_f_size = gen_f_size

        self.main = nn.Sequential(
            # input z latent variable 
            nn.ConvTranspose2d(self.z_size, self.gen_f_size*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.gen_f_size*8),
            nn.ReLU(True),
            gen_block(self.gen_f_size*8, self.gen_f_size*4),
            gen_block(self.gen_f_size*4, self.gen_f_size*2),
            gen_block(self.gen_f_size*2, self.gen_f_size),
            nn.ConvTranspose2d(self.gen_f_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # output of size 3x64x64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, dis_f_size):
        super(Discriminator, self).__init__()
        self.dis_f_size = dis_f_size
        self.main = nn.Sequential(
            nn.Conv2d(3, self.dis_f_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            discri_block(self.dis_f_size, self.dis_f_size*2),
            discri_block(self.dis_f_size*2, self.dis_f_size*4),
            discri_block(self.dis_f_size*4, self.dis_f_size*8),
            #output size disciminator_feature_map_size*8 x 4 x 4
            nn.Conv2d(self.dis_f_size*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)