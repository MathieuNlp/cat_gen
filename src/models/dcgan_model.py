import torch
import torch.nn as nn

def weights_init(m) : 
    """
    Weight initialisation, in the paper, all weights are initialized randomly with normal distrib. mean=0, std=0.02
    Depending of which component (Conv or BatchNorm) it is, we initialize it in a certain way
    
    Arguments:
        m: model architecture (composition of all components)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 :
        nn.init.normal_(m.weight.data, 0.0, 0.02) 
    elif classname.find("BatchNorm") != -1 :
        nn.init.normal_(m.weight.data, 1, 0.02) 
        nn.init.constant_(m.bias.data, 0) # bias to 0


def gen_block(in_channels, out_channels):
    """
    Block that is repeated in the generator to get the features mapping

    Arguments:
        in_channels: input channels of the data
        out_channels: output channels wanted

    Return:
        Sequencial module: Sequence of transformation for 1 block
    """
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            )


def discri_block(in_channels, out_channels):
    """
    Block that is repeated in the discriminator to get the features mapping

    Arguments:
        in_channels: input channels of the data
        out_channels: output channels wanted

    Return:
        Sequencial module: Sequence of transformation for 1 block
    """
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            )
    

class Generator(nn.Module) :
    """
    Generator class is taking a vector from the latent space z from a gaussian distribution and generating a 3x64x64 image

    Arguments:
        config: config file

    Return:
        image of size 3x64x64
    """
    def __init__(self, config):
        super(Generator, self).__init__()
        self.z_size = config["GENERATOR"]["Z_SIZE"]
        self.gen_f_size = config["GENERATOR"]["GENERATOR_FEATURE_MAP_SIZE"]

        self.generation = nn.Sequential(
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
        return self.generation(input)


class Discriminator(nn.Module):
    """
    Discriminator class is taking an image 3x64x64 and is classifying 1: from training, 0: from generator

    Arguments:
        config: config file

    Return:
        0 or 1 -> gives the classification of the image
    """
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.dis_f_size = config["DISCRIMINATOR"]["DISCRIMINATOR_FEATURE_MAP_SIZE"]
        self.dicriminate = nn.Sequential(
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
        return self.dicriminate(input)