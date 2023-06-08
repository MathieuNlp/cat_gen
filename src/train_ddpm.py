import models.diffusion_model as dm
import loader.dataloader as my_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

timesteps = 300

# define beta schedule
betas = dm.linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = dm.torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

dataloader = my_dataloader.dataloader
results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
model = dm.Unet(
    dim=my_dataloader.image_size,
    channels=my_dataloader.num_channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10

for epoch in tqdm(range(epochs), desc="Epochs"):
    for step, batch in enumerate(tqdm(dataloader, desc="Batches")):
      optimizer.zero_grad()

      batch_size = batch[0].shape[0]
      batch = batch[0].to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = dm.p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()

      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = dm.num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: dm.sample(model, batch_size=n, channels=my_dataloader.num_channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
# sample 64 images
samples = dm.sample(model, image_size=my_dataloader.image_size, batch_size=64, channels=my_dataloader.num_channels)
save_PATH = "./saved_model/"
torch.save(model.state_dict(), save_PATH+"ddpm_checkpoint.pt")
# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(my_dataloader.image_size, my_dataloader.image_size, my_dataloader.num_channels), cmap="gray")



random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(my_dataloader.image_size, my_dataloader.image_size, my_dataloader.num_channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')