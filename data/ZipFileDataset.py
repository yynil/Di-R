import torch
from torch.utils.data import Dataset
import glob
import zipfile
import zip_fast_reader
from PIL import Image
import io
from PIL import Image
import io
import threading

class ZipFastDataset(Dataset):
    def __init__(self, input_dir,transforms=None,tokenizer=None,max_len=255,pad_id=0,eos_id=1):
        self.input_dir = input_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.inited = False
        

    def init_files(self):
        import time
        elapsed = time.time()
        self.zip_files = glob.glob(self.input_dir + '/*.zip')
        print("init_files time:", time.time() - elapsed)
        elapsed = time.time()
        self.zip_readers = [ zip_fast_reader.ZipReader(file) for file in self.zip_files]
        print("init_zip_readers time:", time.time() - elapsed)
        self.length = 0
        self.all_file_names = []
        elapsed = time.time()
        for zip_reader in self.zip_readers:
            filenames = zip_reader.read_filenames()
            current_file_names = []
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    self.length += 1
                    current_file_names.append(filename)
            self.all_file_names.append(current_file_names)
        print("init_all_file_names time:", time.time() - elapsed, ' file length = ', self.length)
        self.inited = True
    def __len__(self):
        if not self.inited:
            self.init_files()
        return self.length

    def __getitem__(self, idx):
        #find the index of the file
        if not self.inited:
            self.init_files()
        current_file_idx = 0
        while idx >= len(self.all_file_names[current_file_idx]):
            idx -= len(self.all_file_names[current_file_idx])
            current_file_idx += 1
        #get the file
        file_name = None
        try:
            file_name = self.all_file_names[current_file_idx][idx]
            content = self.zip_readers[current_file_idx].read_file_in_zip(file_name)
            base_name = file_name.split('.')[0]
            text_file = self.zip_readers[current_file_idx].read_file_in_zip(base_name + '.txt')
            text_file = str(bytearray(text_file),'utf-8')
            features = self.tokenizer.encode(text_file)
            image = Image.open(io.BytesIO(bytearray(content)))
            if self.transforms:
                image = self.transforms(image)
            return image, features
        except:
            print("Error reading idx:",idx," current_file_idx:",current_file_idx," zipfile:",self.zip_files[current_file_idx])
            print("Error reading file: " + file_name)
            raise Exception("Error reading file: " + file_name+" in zipfile: "+self.zip_files[current_file_idx])
    

if __name__ == '__main__':
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(parent_dir)
    sys.path.append(parent_dir)
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
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
    
    image_size = 512
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])


    rwkv_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tokenizer','rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(rwkv_file)
    dataset = ZipFastDataset('/media/yueyulin/TOUROS/images/laion400m_zip/batch0',tokenizer=tokenizer,transforms=transform)
    print(len(dataset))
    import time 
    elapsed = time.time()
    image,features = dataset[0]
    print("time:", time.time() - elapsed)
    print(features)
    print(image.shape)

