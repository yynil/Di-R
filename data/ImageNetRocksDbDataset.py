import torch
from torch.utils.data import Dataset
import glob
import zipfile
import zip_fast_reader
from PIL import Image
import io
from PIL import Image
import io
import datetime
import os
import imagenet_rocksdb
class ImageNetRocksDBDS(Dataset):
    def __init__(self, input_dir,transforms=None,log=False):
        self.input_dir = input_dir
        self.transforms = transforms
        self.access_count = 0
        self.log_access_time = 10000
        self.log = log
        self.db = imagenet_rocksdb.RocksDBWrapper(input_dir)
        self.total_seconds = 0.0
    def __len__(self):
        return self.db.len()
    
    def log_status(self):
        #print total_seconds, access_count, average_seconds in red in one line
        print(f"{self.total_seconds:.3f} {self.access_count} {self.total_seconds/self.access_count:.3f}")

    def __getitem__(self, idx):
        #find the info of the file of the idx
        self.access_count += 1
        time = datetime.datetime.now()
        class_id,image_content = self.db.get(idx)
        elapsed = datetime.datetime.now() - time
        self.total_seconds += elapsed.total_seconds()
        image = Image.open(io.BytesIO(bytearray(image_content)))
        if self.transforms:
            image = self.transforms(image)
        if self.access_count % self.log_access_time == 0:
            self.log_status()
        return image, class_id

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ImageNetRocksDBDS')
    parser.add_argument('--input_dir', type=str, default='/home/yueyulin/data/images/ILSVRC2012_rocksdb_auth', help='input_dir')
    args = parser.parse_args()
    input_dir = args.input_dir
    import os
    from torchvision import transforms
    import numpy as np
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

    def convert_to_3_channels(pil_image):
        # Check the number of channels in the image
        if pil_image.mode == 'L':
            # If the image has one channel, convert it to three channels
            pil_image = pil_image.convert("RGB")
        return pil_image

    image_size = 256
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    dataset = ImageNetRocksDBDS(input_dir,transforms=transform)
    length = len(dataset)
    print(length)
    import time 
    elapsed = time.time()
    image,features = dataset[0]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//2]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//3]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//4]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//5]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//6]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//7]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length//8]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

    elapsed = time.time()
    image,features = dataset[length-1]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)






    dataset.log_status()

