import torch
from PIL import Image
from diffusers import DDPMScheduler
from datasets import load_dataset
from utils import show_images, device, get_tranform


image_size = 32
batch_size = 64

transform = get_tranform(image_size)

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
dataset.set_transform(transform)
# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)
# xb = next(iter(train_dataloader))["images"].to(device)[:8]
xb = next(iter(train_dataloader))["images"].to(device)[0]
xb = torch.unsqueeze(xb, axis=0)
xb = torch.broadcast_to(xb, (8, xb.shape[1], xb.shape[2], xb.shape[3]))
# print("X shape:", xb.shape)
input_images = show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)
input_images.save('./inputs.jpg', quality=95)



noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
# plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
# plt.legend(fontsize="x-large")

timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb)
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
noisy_images = show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)
noisy_images.save('./diffusions.jpg', quality=95)