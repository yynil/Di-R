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
import einops
import open_clip

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

#################################################################################
#                                 Core DiRwkv Model                                #
#################################################################################

class DiRWKVBlock(nn.Module):
    """
    A DiRwkv block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, args,layer_id,skip=False,  **block_kwargs):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(hidden_size)
        if 'x060' in os.environ["RWKV_MY_TESTING"]:
            self.att = RWKV_Tmix_x060(args, layer_id)
        else:
            self.att = RWKV_TimeMix_RWKV5(args, layer_id)
        if 'x060' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = RWKV_CMix_x060(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        #with original RWKVBlock to enable the residual connection and drop path
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)

        self.skip_linear = nn.Linear(args.n_embd*2, args.n_embd) if skip else None
    """
    Remove the conditional input because we concatenate the timestep and conditional embeddings into the x input.
    """
    def forward(self, x, x_emb = None,skip=None):
        args = self.args
        B, T, C = x.size()
        if self.skip_linear:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        if self.args.dropout == 0:
            x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, condition=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        if condition == True: 
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        

    def forward(self, x, c=None): 
        if c is not None: 
            c = self.adaLN_modulation(c).squeeze(1)
            shift, scale = c.chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.norm_final(x)
            x = self.linear(x)
        return x



class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class DiRWKV(pl.LightningModule):
    """
    Diffusion model with a Rwkv backbone.
    """
    def __init__(
        self,
        args = None,
        text_encoder = None,
        deepspeed_offload = False,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        learn_sigma=True,
        class_dropout_prob=0.1,
        use_pos_emb=False,
        mlp_time_embed=True,
    ):
        super().__init__()
        self.args = args
        self.deepspeed_offload = deepspeed_offload
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.text_encoder = text_encoder
        self.text_embedder = nn.Linear(text_encoder.token_embedding.weight.shape[1], args.n_embd) if text_encoder is not None else None
        self.null_embeddings = torch.nn.Parameter(torch.randn(args.n_embd))
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        #replace the DiR's time embed to DiS's time embed
        self.time_embed = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd),
            nn.SiLU(),
            nn.Linear(4 * args.n_embd, args.n_embd),
        ) if mlp_time_embed else nn.Identity()
        if self.text_encoder is not None:
            self.extras = 2
        else:
            self.extras = 1
        self.class_dropout_prob = class_dropout_prob
        num_patches = self.x_embedder.num_patches
        if use_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches , hidden_size))
        else:
            self.pos_embed = None
        #in,mid,out blocks to suppor the skip connection
        self.in_blocks = nn.ModuleList([
            DiRWKVBlock(hidden_size,args,layer_id, skip=False) for layer_id in range(depth//2)
        ])
        self.mid_block = DiRWKVBlock(hidden_size,args,depth//2, skip=False)
        self.out_blocks = nn.ModuleList([
            DiRWKVBlock(hidden_size,args,i + depth // 2 + 1, skip=False) for i in range(depth//2)
        ])
        # self.blocks = nn.ModuleList([
        #     DiRWKVBlock(hidden_size,args,layer_id, mlp_ratio=mlp_ratio) for layer_id in range(depth)
        # ])
        if self.text_encoder is not None :
            self.final_layer = FinalLayer(args.n_embd, patch_size, self.out_channels, condition=True)
        else:
            self.final_layer = FinalLayer(args.n_embd, patch_size, self.out_channels)    
        self.initialize_weights()
    def convert_bfloat16(self):
        #only blocks are bfloat16
        for i in range(len(self.in_blocks)):
            self.in_blocks[i].att = self.in_blocks[i].att.bfloat16()
            self.in_blocks[i].ffn = self.in_blocks[i].ffn.bfloat16()
        for i in range(len(self.out_blocks)):
            self.out_blocks[i].att = self.out_blocks[i].att.bfloat16()
            self.out_blocks[i].ffn = self.out_blocks[i].ffn.bfloat16()
        self.mid_block.att = self.mid_block.att.bfloat16()
        self.mid_block.ffn = self.mid_block.ffn.bfloat16()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def unpatchify(self, x, channels=3):
        patch_size = int((x.shape[2] // channels) ** 0.5)
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
        x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
        return x

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of labels
        """
        args = self.args
        x = self.x_embedder(x.bfloat16())
        t_emb = self.time_embed(timestep_embedding(t, args.n_embd).bfloat16())                   # (N, D)
        t_emb = t_emb.unsqueeze(dim=1)
        x = torch.cat([t_emb,x],dim=1)
        if y is not None:
            if self.training and self.class_dropout_prob > 0:
                #randomly set y to [1]*D
                batch = y.shape[0]
                for i in range(batch):
                    if random.random() < self.class_dropout_prob:
                        y[i] = torch.tensor([args.eot]+[0]*(args.len-1))
            with torch.no_grad():
                y_emb = self.text_encoder.encode_text(y)
            
            #y is [N] integer labels tensor
            y_emb = self.text_embedder(y_emb) # (N, D)
            y_emb = y_emb.unsqueeze(dim=1)
            x = torch.cat([y_emb,x],dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        skips = []
        x_emb = x
        for block in self.in_blocks:
            if args.tiny_att_dim > 0:
                x = block(x,x_emb)
            else:
                x = block(x)
            skips.append(x)
        if args.tiny_att_dim > 0:
            x = self.mid_block(x,x_emb)
        else:
            x = self.mid_block(x)
        for block in self.out_blocks:
            if args.tiny_att_dim > 0:
                x = block(x,x_emb,skips.pop())
            else:
                x = block(x,skips.pop())
        if y is not None:
            x = self.final_layer(x, c=t_emb+y_emb)                # (N, T, patch_size ** 2 * out_channels)
        else:
            x = self.final_layer(x)                # (N, T, patch_size ** 2 * out_channels)
        x = x[:, self.extras:, :]
        x = self.unpatchify(x,self.out_channels)                   # (N, out_channels, H, W)
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
rwkvArgs.warmup_steps = 50
rwkvArgs.tiny_att_dim = 1024
rwkvArgs.eot = 49407
rwkvArgs.len = 77
# text encoder with open_clip
text_encoder, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
print(text_encoder)

def DiRwkv_XL_2(**kwargs):
    #setup rwkv args
    args = copy.copy(rwkvArgs)
    args.n_layer = 28
    args.tiny_att_layer = args.n_layer
    args.n_embd = 1536
    args.dim_ffn = 6144
    args.dim_att = 1536
    return DiRWKV(args,text_encoder=text_encoder,depth=args.n_layer, hidden_size=1536, patch_size=2, **kwargs)





DiRwkv_models = {
    'DiRwkv_XL_2': DiRwkv_XL_2
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

    print('test DiRWKV model')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    # 定义要获取嵌入的文本
    texts = ["这是一个测试文本","a bike on the road with a person riding it"]

    # 使用tokenizer处理文本
    texts = tokenizer(texts)
    print(texts)
    print(texts.shape)

    null_token = torch.tensor([tokenizer.eot_token_id]+[0]*(texts.shape[1]-1))
    print(null_token)
    texts = torch.cat([texts, null_token.unsqueeze(0)], dim=0)

    image_size = 256
    input_size = image_size // 8
    model = DiRwkv_models['DiRwkv_XL_2'](input_size=input_size)
    model.convert_bfloat16()
    print(model)
    N = 3
    C = 4
    H = input_size
    W = input_size
    model = model.to('cuda')

    x = torch.randn(N, C, H, W).to('cuda')
    y = texts.to('cuda')
    t = torch.randint(0, 1000, (x.shape[0],)).to(x.device)
    print(t)
    from torch.amp import autocast
    with autocast(device_type='cuda',dtype=torch.bfloat16):
        out = model(x, t, y)
    print(out.shape)
    print(out)
