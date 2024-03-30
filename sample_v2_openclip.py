# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models_v2 import DiRwkv_models
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

import open_clip

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device:',device)


    # Load model:
    latent_size = args.image_size // 8
    model = DiRwkv_models[args.model](
        input_size=latent_size,use_pos_emb=args.is_pos_emb
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path,map_location='cpu')
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict,strict=False)
    print(state_dict.keys())
    print(info)
    text_encoder, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    text_encoder = text_encoder.to(device)
    model.text_encoder = text_encoder# Since EMA has the text encoder parameters, overload the text encoder in case the text encoder's parameters are overwritten by the ema
    model = model.bfloat16()
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    # state_dict = {k.replace('vae.', ''): v for k, v in state_dict.items()}
    # info = vae.load_state_dict(state_dict,strict=False)
    # print(info)
    
    vae = vae.to(device)
    vae.eval()
    prompts = ['A black motorcycle','A cat','a motorcycle','a bike in a parking lot']
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    y = tokenizer(prompts).to(device)
    print(f'{y.shape = }')
    n = y.shape[0]
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    z = torch.cat([z, z], 0)
    fixed_y_len = 77
    y_null = torch.tensor([[tokenizer.eot_token_id] + [0] * (fixed_y_len-1)] * n, device=device)
    print(f'{y_null.shape = }')
    # y_null = torch.tensor([[1] * max_length] * n, device=device)
    y = torch.cat([y, y_null], 0)
    print(f'{y.shape = }')
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    # print(model_kwargs)
    from torch.amp import autocast
    print(f'{y.shape = }, {z.shape = }')
    with autocast(device_type=device,dtype=torch.bfloat16):
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    print(samples.shape)
    samples = vae.decode(samples / 0.18215).sample
    import time
    sample_file_name = f'sample_{time.time_ns()}.png'
    save_image(samples, sample_file_name, nrow=4, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiRwkv_models.keys()), default="DiRwkv_XL_2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=4)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='/media/yueyulin/KINGSTON/models/DiRwkv6/DiR-XL-2/XL2_OpenClip_epoch1_model.pt')#Use the model or ema to evaluate
    parser.add_argument("--is-pos-emb", action="store_true",default=False)
    args = parser.parse_args()
    main(args)
