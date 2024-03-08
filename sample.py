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
from models import DiRwkv_models
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

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device:',device)


    # Load model:
    latent_size = args.image_size // 8
    model = DiRwkv_models[args.model](
        input_size=latent_size
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path,map_location='cpu')
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict,strict=False)
    print(state_dict.keys())
    print(info)
    print(state_dict['text_encoder.null_embeddings'])
    model = model.bfloat16()
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    # state_dict = {k.replace('vae.', ''): v for k, v in state_dict.items()}
    # info = vae.load_state_dict(state_dict,strict=False)
    # print(info)
    
    vae = vae.to(device)
    vae.eval()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tokenizer_file = os.path.join(current_dir, 'tokenizer','rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    print(tokenizer.encode('a bike in a parking lot'))
    prompts = ['A black Honda motorcycle parked in front of a garage.','A cat in between two cars in a parking lot.','a motorcycle in a parking lot with a sky background','A blue glass bottle.']
    input_ids = [tokenizer.encode(prompt)+[1] for prompt in prompts]
    print(input_ids)
    n = len(input_ids)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    max_length = max([len(input_id) for input_id in input_ids])
    input_ids = [input_id + [0] * (max_length - len(input_id)) for input_id in input_ids]
    print(input_ids)
    y = torch.tensor(input_ids, device=device)
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([[1] * max_length] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    print(model_kwargs)
    from torch.amp import autocast
    with autocast(device_type=device,dtype=torch.bfloat16):
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    print(samples.shape)
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    # # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # # Create sampling noise:
    # n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # y = torch.tensor(class_labels, device=device)

    # # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiRwkv_models.keys()), default="DiRwkv_XL_2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--cfg-scale", type=float, default=4)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default='/media/yueyulin/KINGSTON/models/DiRwkv6/DiR-XL-2/DiT-XL-2_nlayer_28_model.pth',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
