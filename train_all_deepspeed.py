# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

import copy
from pytorch_lightning import Trainer
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
#HF_ENDPOINT=https://hf-mirror.com
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/home/gpu/data/sdb1/huggingface'
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '1024'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
from models import DiRwkv_models
from custom_dataset import CustomTextDataset
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import pytorch_lightning as pl
import math
from data.ZipFileDataset import ZipFastDataset


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    pass
    # ema_params = OrderedDict(ema_model.named_parameters())
    # model_params = OrderedDict(model.named_parameters())
    # for name, param in model_params.items():
    #     if name in ema_params:
    #         ema_params[name].mul_(decay).add_(param.data.to('cpu'), alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def save_trainable_parameters(model, trainable_dir_output):
    print(f"save trainable parameters to {trainable_dir_output} ")
  

    # 保存可训练的参数
    
    save_path = os.path.join(trainable_dir_output, 'model.pth')
    state_dict = {name: param.data for name, param in model.named_parameters() if param.requires_grad}
    torch.save(state_dict, save_path)
    print(f"save model parameters to {save_path}")

#################################################################################
#                                  Training Loop                                #
#################################################################################

class YueyuTrainCallback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now



        # rank_zero_info(f"{real_step} {lr}")

        if trainer.is_global_zero:
            if  trainer.global_step == 0: # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                import wandb
                import datetime
                wandb.init(project='DiRWKV',
                        name='3090ti_run_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                        save_code=False,
                            )
                trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # update_ema(self.ema, pl_module, decay=0.9999)
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero:  # logging   
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)
            lll = {"loss": trainer.my_loss,   "Gtokens": real_step * token_per_step / 1e9}
            trainer.my_wandb.log(lll, step=int(real_step))
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f}  {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            print(f'saving trainable to {args.trainable_dir_output} epoch {trainer.current_epoch}')
            output_dir = f"{args.trainable_dir_output}/epoch_{trainer.current_epoch}"
            os.makedirs(output_dir, exist_ok=True)
            save_trainable_parameters(pl_module,output_dir)
            #torch.save(trainer.model.state_dict(), os.path.join(output_dir, 'model.pt'))
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    
    #setup rwkv args

    model = DiRwkv_models[args.model](input_size=latent_size,deepspeed_offload=True)
    model.convert_bfloat16()
    # Note that parameter initialization is done within the DiT constructor
    # ema = deepcopy(model)  # Create an EMA of the model for use after training
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    #call wrapped model's configure_optimizers


    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    if not args.is_zip:
        image_folder = os.path.join(args.data_path, "train2017")
        captions_file = os.path.join(args.data_path, "coco_captions_train2017_texts.pkl")
        # dataset = ImageFolder(args.data_path, transform=transform)
        dataset = CustomTextDataset(image_folder, captions_file, transform)
    else:
        rwkv_file = os.path.join(os.path.dirname(__file__), 'tokenizer','rwkv_vocab_v20230424.txt')
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        tokenizer = TRIE_TOKENIZER(rwkv_file)
        dataset = ZipFastDataset(args.data_path, tokenizer=tokenizer,transforms=transform)
    max_len = 128 
    def collate_fn(batch):
        images, input_ids = zip(*batch)
        current_max_len = 0
        new_ids = []
        for input_id in input_ids:
            if len(input_id) > max_len:
                input_id = input_id[:max_len]
            input_id.append(dataset.eos_id)
            current_max_len = max(current_max_len,len(input_id))
            new_ids.append(input_id)
        new_ids = [x + [dataset.pad_id] * (current_max_len - len(x)) for x in new_ids]
        return torch.stack(images),torch.tensor(new_ids)
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    # ema.eval()  # EMA model should always be in eval mode
    # requires_grad(ema, flag=False)  # EMA should not be updated by gradients
    requires_grad(vae, flag=False)  # VAE should not be updated by gradients
    model.set_vae(vae)
    model.set_diffusion(diffusion)


    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    from tqdm import tqdm

    logger.info(f"Training for {args.epochs} epochs...")
    call_back_args = copy.copy(model.args)
    call_back_args.check_val_every_n_epoch = int(1e20)
    call_back_args.log_every_n_steps = int(1e20)
    call_back_args.num_sanity_val_steps = 0
    call_back_args.enable_checkpointing = False
    call_back_args.accumulate_grad_batches = 1
    call_back_args.gradient_clip_val = 1.0
    call_back_args.precision = 'bf16'
    call_back_args.logger = False
    call_back_args.my_pile_stage = 0
    call_back_args.my_pile_edecay = 0
    call_back_args.epoch_begin = 0
    call_back_args.epoch_count = 150
    call_back_args.epoch_save = 1
    call_back_args.epoch_steps = 1000
    call_back_args.max_epochs = call_back_args.epoch_count
    call_back_args.my_exit_tokens = 0
    call_back_args.proj_dir = experiment_dir
    call_back_args.weight_decay_final = -1
    call_back_args.micro_bsz = args.global_batch_size
    call_back_args.num_nodes = 1
    call_back_args.devices = 1
    call_back_args.trainable_dir_output = experiment_dir
    call_back_args.real_bsz = int(call_back_args.num_nodes) * int(call_back_args.devices) * call_back_args.micro_bsz
    from datetime import datetime
    call_back_args.my_timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    call_back_args.my_exit = 150
    call_backs = [YueyuTrainCallback(call_back_args)]
    # call_backs[0].set_ema(ema)
    device = 'cuda'
    trainer = Trainer(accelerator=device,strategy="deepspeed_stage_2_offload",devices='auto',num_nodes=1,precision='bf16-mixed',
            logger=call_back_args.logger,callbacks=call_backs,max_epochs=call_back_args.max_epochs,check_val_every_n_epoch=call_back_args.check_val_every_n_epoch,num_sanity_val_steps=call_back_args.num_sanity_val_steps,
            log_every_n_steps=call_back_args.log_every_n_steps,enable_checkpointing=call_back_args.enable_checkpointing,accumulate_grad_batches=call_back_args.accumulate_grad_batches,gradient_clip_val=call_back_args.gradient_clip_val)
    trainer.fit(model,loader)
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/media/yueyulin/TOUROS/images/laion400m_zip/batch0')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiRwkv_models.keys()), default="DiRwkv_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--is-zip", action="store_true",default=True)
    args = parser.parse_args()
    main(args)
# Copyright (c) Meta Platforms, Inc. and affiliates.
