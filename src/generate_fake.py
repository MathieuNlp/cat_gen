import torch
import models.dcgan_model as dcgan
import yaml
import utils

# Load the config file and generator trained and then generates images 
with open("./config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)
    
device = 'cpu'
model_path = config["SAVE"]["MODEL_CHECKPOINT"]
generator = dcgan.Generator(config)
generator.load_state_dict(torch.load(model_path))
utils.plot_generate_fake(config, generator, device)