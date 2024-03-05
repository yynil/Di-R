# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from argparse import Namespace
import argparse
from collections import OrderedDict
import random
from typing import List
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '1024'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '1024'
os.environ['RWKV_CTXLEN'] = '1024'
from src.model import RWKV, Block, RWKV_Tmix_x060, RWKV_CMix_x060,RWKV_TimeMix_RWKV5,RWKV_ChannelMix
from src.model_ext import RwkvForSequenceEmbedding
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import pytorch_lightning as pl
import copy
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.bfloat16())
        return t_emb




#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiRWKVBlock(nn.Module):
    """
    A DiRwkv block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, args,layer_id,  **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        if 'x060' in os.environ["RWKV_MY_TESTING"]:
            self.att = RWKV_Tmix_x060(args, layer_id)
        else:
            self.att = RWKV_TimeMix_RWKV5(args, layer_id)
        if 'x060' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = RWKV_CMix_x060(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)
        self.norm2 = nn.LayerNorm(hidden_size)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.att(modulate(self.norm1(x), shift_msa, scale_msa).bfloat16())
        x = x + gate_mlp.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp).bfloat16())
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiRWKV(pl.LightningModule):
    """
    Diffusion model with a Rwkv backbone.
    """
    def __init__(
        self,
        args = None,
        text_encoder_args = None,
        deepspeed_offload = False,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        mlp_ratio=4.0,
        learn_sigma=True,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        self.args = args
        self.text_encoder_args = text_encoder_args
        assert args.n_embd == text_encoder_args.n_embd
        self.deepspeed_offload = deepspeed_offload
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = nn.Linear(y_nemb, hidden_size, bias=True)
        text_encoder_rwkv = RWKV(text_encoder_args)
        self.text_encoder = RwkvForSequenceEmbedding(text_encoder_rwkv)
        self.class_dropout_prob = class_dropout_prob
        num_patches = self.x_embedder.num_patches

        self.blocks = nn.ModuleList([
            DiRWKVBlock(hidden_size,args,layer_id, mlp_ratio=mlp_ratio) for layer_id in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
    def convert_bfloat16(self):
        #only blocks are bfloat16
        for i in range(len(self.blocks)):
            self.blocks[i].att = self.blocks[i].att.bfloat16()
            self.blocks[i].ffn = self.blocks[i].ffn.bfloat16()
        self.text_encoder = self.text_encoder.bfloat16()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,T) tensor of input_ids with pad(0) and eos(1) tokens
        """
        x = self.x_embedder(x.bfloat16())   # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y_len = y.shape[1]
        batch = y.shape[0]
        if self.training and self.class_dropout_prob > 0:
            #randomly set y to [1]*D
            for i in range(batch):
                if random.random() < self.class_dropout_prob:
                    y[i] = torch.ones(y_len)
        y = self.text_encoder(y)    # (N, D)
        c = t + y                            # (N, D)
        layer_id = 0
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def set_diffusion(self,diffusion):
        self.diffusion = diffusion

    def set_vae(self, vae):
        self.vae = vae
        self.vae.eval()

    def training_step(self, batch, batch_idx) :
        x, y = batch
        self.vae.eval()
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],)).to(x.device)
        model_kwargs = dict(y=y)
        loss_dict = self.diffusion.training_losses(self, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        all_keys = set()
        #frozen vae 
        for n, p in self.vae.named_parameters():
            p.requires_grad_ = False
        for n, p in self.named_parameters():
            if ('vae' in n) or ('diffusion' in n) or not p.requires_grad_:
                #disable the require_grad for the parameter
                p.requires_grad_ = False
                continue
            all_keys.add(n)
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        print('all', all_keys)
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)




#################################################################################
#                                   DiT Configs                                  #
#################################################################################
rwkvArgs = argparse.Namespace()
rwkvArgs.my_pos_emb = 0
rwkvArgs.pre_ffn = 0
rwkvArgs.head_size_divisor = 8
rwkvArgs.ctx_len = 1024
rwkvArgs.dropout = 0.05
rwkvArgs.head_qk = 0
rwkvArgs.grad_cp = 0
rwkvArgs.save_per_batches = 10000
rwkvArgs.my_exit = 3
n_embd = 1024
dim_att = 1024
n_head = 16
dim_ffn = 4096
n_layer = 24
version = '6'
head_size_a = 64 
rwkvArgs.n_embd = n_embd
rwkvArgs.dim_att = dim_att
rwkvArgs.dim_ffn = dim_ffn
rwkvArgs.n_layer = n_layer
rwkvArgs.version = version
rwkvArgs.head_size_a = head_size_a
rwkvArgs.weight_decay = 0.001
rwkvArgs.lr_init = 3e-4
rwkvArgs.lr_final = 1e-5
rwkvArgs.beta1 = 0.9
rwkvArgs.beta2 = 0.99
rwkvArgs.betas = (0.9, 0.99)
rwkvArgs.layerwise_lr = 1
rwkvArgs.my_pile_stage = 1
rwkvArgs.adam_eps = 1e-8
rwkvArgs.vocab_size = 65536
rwkvArgs.warmup_steps = 50

def DiRwkv_XL_2(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    text_encoder_args = copy.copy(rwkvArgs)
    text_encoder_args.n_layer = 12
    args.n_layer = 28
    return DiRWKV(args,text_encoder_args,depth=args.n_layer, hidden_size=1024, patch_size=2, **kwargs)

def DiRwkv_XL_4(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    text_encoder_args = copy.copy(rwkvArgs)
    text_encoder_args.n_layer = 12
    args.n_layer = 28
    return DiRWKV(args,text_encoder_args,depth=args.n_layer, hidden_size=1024, patch_size=4,  **kwargs)

def DiRwkv_XL_8(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    text_encoder_args = copy.copy(rwkvArgs)
    text_encoder_args.n_layer = 12
    args.n_layer = 24
    return DiRWKV(args,text_encoder_args,depth=args.n_layer, hidden_size=1024, patch_size=8, **kwargs)

def DiRwkv_L_2(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    args.n_layer = 16
    text_encoder_args = copy.copy(rwkvArgs)
    text_encoder_args.n_layer = 8
    args.n_layer = 16
    return DiRWKV(args,text_encoder_args,depth=args.n_layer, hidden_size=1024, patch_size=2, **kwargs)

def DiRwkv_L_4(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    args.n_layer = 16
    text_encoder_args = copy.copy(rwkvArgs)
    text_encoder_args.n_layer = 8
    args.n_layer = 24
    return DiRWKV(args,text_encoder_args,depth=args.n_layer, hidden_size=1024, patch_size=4, **kwargs)

def DiRwkv_L_8(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    args.n_layer = 16
    text_encoder_args = copy.copy(rwkvArgs)
    text_encoder_args.n_layer = 8
    args.n_layer = 24
    return DiRWKV(args,text_encoder_args, depth=args.n_layer,hidden_size=1024, patch_size=8, **kwargs)




DiRwkv_models = {
    'DiRwkv_XL_2': DiRwkv_XL_2,  'DiRwkv_XL_4': DiRwkv_XL_4,  'DiRwkv_XL_8': DiRwkv_XL_8,
    'DiRwkv_L_2':  DiRwkv_L_2,   'DiRwkv_L_4':  DiRwkv_L_4,   'DiRwkv_L_8':  DiRwkv_L_8,
}

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

def tokenize_and_pad(input_strs :List[str],tokenizer :TRIE_TOKENIZER, max_len:int = 255,pad_id = 0,eos_id = 1):
    current_max = 0
    input_ids = []
    for input_str in input_strs:
        tokens = tokenizer.encode(input_str)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        tokens.append(eos_id)
        if len(tokens) > current_max:
            current_max = len(tokens)
        input_ids.append(tokens)
    input_ids = [x + [pad_id] * (current_max - len(x)) for x in input_ids]
    return input_ids


if __name__ == '__main__':
    current_path = os.path.dirname(__file__)
    dict_path = os.path.join(current_path, 'tokenizer', 'rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(dict_path)
    print(tokenizer.encode('hello world'))
    input_strs = ['a bike with a red seat', 'a bike with an old black dog']
    input_ids = tokenize_and_pad(input_strs,tokenizer)
    print(input_ids)
    print('test DiRWKV model')
    input_size = 512 // 8
    model = DiRwkv_models['DiT-XL/2'](input_size=input_size)
    model.convert_bfloat16()
    print(model)
    t = 100
    N = 2
    C = 4
    H = input_size
    W = input_size
    model = model.to('cuda')
    x = torch.randn(N,C,H,W).to('cuda')
    t = torch.tensor([t]*N).to('cuda')
    y = torch.tensor(input_ids).to('cuda')
    import torch.amp as amp
    with amp.autocast(device_type='cuda',dtype=torch.bfloat16):
        out = model(x,t,y)
    print(out.shape)
    print(out)