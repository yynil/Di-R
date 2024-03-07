import torch
from torch.utils.data import Dataset
import glob
import zipfile
import zip_fast_reader
from PIL import Image
import io
from PIL import Image
import io

class ZipFastDataset(Dataset):
    def __init__(self, input_dir,transforms=None,tokenizer=None,max_len=255,pad_id=0,eos_id=1):
        self.input_dir = input_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.inited = False
        self.length = 0
        self.init_files()

    def init_files(self):
        #print step and time in red
        import datetime
        print('\033[91m' + 'init_files' + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.zip_files = glob.glob(self.input_dir + '**/*.zip')
        non_txt_files = []
        self.files_list = []
        self.zip_readers_dict = {}
        for zip_file in self.zip_files:
            #print zip_file and time in green
            print('\033[92m' + zip_file + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            zip_reader = zip_fast_reader.ZipReader(zip_file)
            #print read_filenames and time in blue
            print('\033[94m' + 'read_filenames' + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            filenames = zip_reader.read_filenames()
            for filename in filenames:
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    last_dot_idx = filename.rfind('.')
                    if last_dot_idx == -1:
                        continue
                    base_name = filename[:last_dot_idx]
                    text_file_name = base_name + '.txt'
                    if text_file_name not in filenames:
                        non_txt_files.append((zip_file,filename))
                        continue
                    self.length += 1
                    self.files_list.append((zip_file,filename,text_file_name))
            #print zip_reader.close and time in blue
            print('\033[94m' + 'close ' +zip_file + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            del zip_reader

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #find the info of the file of the idx
        zip_file, image_file, text_file = self.files_list[idx]
        if zip_file not in self.zip_readers_dict:
            self.zip_readers_dict[zip_file] = zip_fast_reader.ZipReader(zip_file)
        zip_reader = self.zip_readers_dict[zip_file]
        image_content =  zip_reader.read_file_in_zip(image_file)
        text_content =  str(bytearray(zip_reader.read_file_in_zip(text_file)),'utf-8')
        features = self.tokenizer.encode(text_content)
        image = Image.open(io.BytesIO(bytearray(image_content)))
        if self.transforms:
            image = self.transforms(image)
        return image, features

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

