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
class ZipFastDataset(Dataset):
    def __init__(self, input_dir,transforms=None,tokenizer=None,log=False):
        self.input_dir = input_dir
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.inited = False
        self.length = 0
        self.init_files()
        self.last_access_time = dict()
        self.most_open_files = 240
        self.access_count = 0
        self.log_access_time = 10000
        self.log = log
    def init_files(self):
        import os
        #print step and time in red
        meta_file = self.input_dir + '/meta.txt'
        if os.path.exists(meta_file):
            try:
                #print meta exists , load from meta_file and time in red
                print('\033[91m' + 'meta exists , load from ' + meta_file + '\033[0m')
                self.zip_readers_dict = {}
                with open(meta_file,'r') as f:
                    self.files_list = []
                    for line in f:
                        zip_file, image_file, text_file = line.strip().split('\t')
                        self.files_list.append((zip_file, image_file, text_file))
                    self.length = len(self.files_list)
                return
            except:
                #print in red warning input_dir+'/meta.txt' is corrupted, reinit files
                print('\033[91m' + 'warning ' + self.input_dir + '/meta.txt' + ' is corrupted, reinit files' + '\033[0m')
        import datetime
        print('\033[91m' + 'init_files' + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.zip_files = glob.glob('**/*.zip',root_dir=self.input_dir,recursive=True)
        non_txt_files = []
        self.files_list = []
        self.zip_readers_dict = {}
        for zip_file in self.zip_files:
            #print zip_file and time in green
            print('\033[92m' + zip_file + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            zip_reader = zip_fast_reader.ZipReader(os.path.join(self.input_dir,zip_file))
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
                    #remove the input_dir from zip_file to enable the meta file to be portable
                    self.files_list.append((zip_file,filename,text_file_name))
            #print zip_reader.close and time in blue
            print('\033[94m' + 'close ' +zip_file + '\033[0m', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            del zip_reader
        #write self.files_list to meta_file
        with open(meta_file,'w') as f:
            for zip_file, image_file, text_file in self.files_list:
                f.write(zip_file + '\t' + image_file + '\t' + text_file + '\n')

    def __len__(self):
        return self.length
    
    def log_status(self):
        #print access_count, len of zip_readers_dict, len of last_access_time, value of zip_readers_dict, value of last_access_time in yellow
        print('\033[93m' + 'access_count: ' + str(self.access_count) + '\t' + 'len of zip_readers_dict: ' + str(len(self.zip_readers_dict)) + '\t' + 'len of last_access_time: ' + str(len(self.last_access_time)) + '\t' + 'value of zip_readers_dict: ' + str(self.zip_readers_dict) + '\t' + 'value of last_access_time: ' + str(self.last_access_time) + '\033[0m')

    def close_least_accessed_zip_file(self):
        if len(self.zip_readers_dict) <= self.most_open_files:
            return
        #print close_least_accessed_zip_file in red
        if self.log:
            print('\033[91m' + 'close_least_accessed_zip_file' + '\033[0m')
        sorted_last_access_time = sorted(self.last_access_time.items(), key=lambda x: x[1])
        zip_file = sorted_last_access_time[0][0]
        #print close zip_file and time in red
        if self.log:    
            print('\033[91m' + 'close zip_file: ' + zip_file + '\t' + 'time: ' + str(self.last_access_time[zip_file]) + '\033[0m')
        #remove zip_file from zip_readers_dict and last_access_time
        del self.zip_readers_dict[zip_file]
        del self.last_access_time[zip_file]

    def __getitem__(self, idx):
        #find the info of the file of the idx
        self.access_count += 1
        zip_file, image_file, text_file = self.files_list[idx]
        if zip_file not in self.zip_readers_dict:
            self.zip_readers_dict[zip_file] = zip_fast_reader.ZipReader(os.path.join(self.input_dir, zip_file))
            self.last_access_time[zip_file] = datetime.datetime.now()
        zip_reader = self.zip_readers_dict[zip_file]
        image_content =  zip_reader.read_file_in_zip(image_file)
        text_content =  str(bytearray(zip_reader.read_file_in_zip(text_file)),'utf-8')
        features = self.tokenizer.encode(text_content)
        image = Image.open(io.BytesIO(bytearray(image_content)))
        if self.transforms:
            image = self.transforms(image)
        if self.access_count % self.log_access_time == 0:
            self.log_status()
        self.close_least_accessed_zip_file()
        return image, features

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ZipFastDataset')
    parser.add_argument('--input_dir', type=str, default='/media/yueyulin/TOUROS/images/laion400m_zip', help='input_dir')
    args = parser.parse_args()
    input_dir = args.input_dir
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
    dataset = ZipFastDataset(input_dir,tokenizer=tokenizer,transforms=transform)
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

