import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse
import os
#HF_ENDPOINT=https://hf-mirror.com
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/media/yueyulin/KINGSTON/huggingface'
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '1024'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'


device = 'cuda'
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
vae.eval()
image_file = '/media/yueyulin/KINGSTON/data/images/cc12m/0.jpeg'
from PIL import Image
from torchvision import transforms

image = Image.open(image_file)
image = transforms.Resize((512, 512))(image)
image = transforms.ToTensor()(image).unsqueeze(0).to(device)
z = vae.encode(image).latent_dist.sample()
print(z.shape)
image_recon = vae.decode(z).sample
print(image_recon.shape)
save_image(image_recon, 'recon.jpg', nrow=4, normalize=False, value_range=(-1, 1))