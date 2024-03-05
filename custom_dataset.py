import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class CustomDataset(Dataset):
    def __init__(self, image_folder, captions_pkl, transforms=None):
        self.image_folder = image_folder
        # Print "开始加载caption" in red and datetime
        print(pd.to_datetime('now'))
        print("\033[91m开始加载caption\033[0m")
        self.captions = pd.read_pickle(captions_pkl)
        # Print "caption加载完成" in green
        print(pd.to_datetime('now'))
        print("\033[92mcaption加载完成\033[0m")

        self.transforms = transforms
        # Print "开始加载image_filenames" in red
        print(pd.to_datetime('now'))
        print("\033[91m开始加载image_filenames\033[0m")
        self.image_filenames = os.listdir(image_folder)
        # Print "image_filenames加载完成" in green
        print(pd.to_datetime('now'))
        print("\033[92mimage_filenames加载完成\033[0m")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_id = int(os.path.splitext(image_filename)[0])
        image = Image.open(os.path.join(self.image_folder, image_filename))

        if self.transforms:
            image = self.transforms(image)

        features = self.captions.loc[image_id,'features'].values
        feature = random.choice(features)

        return image, torch.tensor(feature)
    
class CustomTextDataset(Dataset):
    def __init__(self, image_folder, captions_pkl, transforms=None,tokenizer=None,max_len=255,pad_id=0,eos_id=1):
        self.image_folder = image_folder
        # Print "开始加载caption" in red and datetime
        print(pd.to_datetime('now'))
        print("\033[91m开始加载caption\033[0m")
        self.captions = pd.read_pickle(captions_pkl)
        # Print "caption加载完成" in green
        print(pd.to_datetime('now'))
        print("\033[92mcaption加载完成\033[0m")

        self.transforms = transforms
        # Print "开始加载image_filenames" in red
        print(pd.to_datetime('now'))
        print("\033[91m开始加载image_filenames\033[0m")
        self.image_filenames = os.listdir(image_folder)
        # Print "image_filenames加载完成" in green
        print(pd.to_datetime('now'))
        print("\033[92mimage_filenames加载完成\033[0m")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id
        self.eos_id = eos_id
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_id = int(os.path.splitext(image_filename)[0])
        image = Image.open(os.path.join(self.image_folder, image_filename))

        if self.transforms:
            image = self.transforms(image)

        features = self.captions.loc[image_id,'input_ids'].values
        feature = random.choice(features)

        return image, feature
    
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

if __name__ == '__main__':
    # Setup data:
    image_size = 512
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    image_folder = '/media/yueyulin/KINGSTON/data/images/coco/train2017'
    captions_file = '/media/yueyulin/KINGSTON/data/images/coco/coco_captions_train2017.pkl'
    train_ds = CustomDataset(image_folder, captions_file, transform)
    print(len(train_ds))
    print(train_ds[0][1].shape)

    image_folder = '/media/yueyulin/KINGSTON/data/images/coco/val2017'
    captions_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_val2017.pkl'
    val_ds = CustomDataset(image_folder, captions_file, transform)
    print(len(val_ds))
    print(val_ds[0][1].shape)

    sampler = DistributedSampler(
        train_ds,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42
    )
    loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    for i, (x, y) in enumerate(tqdm(loader)):
        print(x.shape)
        print(y.shape)
        break

    print('----------------------------------------------------')

    image_folder = '/media/yueyulin/KINGSTON/data/images/coco/train2017'
    captions_file = '/media/yueyulin/KINGSTON/data/images/coco/coco_captions_train2017_texts.pkl'
    train_ds = CustomTextDataset(image_folder, captions_file, transform)
    print(len(train_ds))
    print(train_ds[0][1])

    image_folder = '/media/yueyulin/KINGSTON/data/images/coco/val2017'
    captions_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_val2017_texts.pkl'
    val_ds = CustomTextDataset(image_folder, captions_file, transform)
    print(len(val_ds))
    print(val_ds[0][1])

    sampler = DistributedSampler(
        train_ds,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42
    )
    #add a collate_fn to pad the input_ids and add eos_id
    max_len = 255
    def collate_fn(batch):
        images, input_ids = zip(*batch)
        current_max_len = 0
        new_ids = []
        for input_id in input_ids:
            if len(input_id) > max_len:
                input_id = input_id[:max_len]
            input_id.append(train_ds.eos_id)
            current_max_len = max(current_max_len,len(input_id))
            new_ids.append(input_id)
        new_ids = [x + [train_ds.pad_id] * (current_max_len - len(x)) for x in new_ids]
        return torch.stack(images),torch.tensor(new_ids)
    loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=False,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    for i, (x, y) in enumerate(tqdm(loader)):
        print(x.shape)
        print(y.shape)
        print(y)
        break
