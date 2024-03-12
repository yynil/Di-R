# Fast Zip File Dataset based on RUST

## Intention and Background

Image files are often massive and organized as a list of zip files. We need a high efficiency tool to extract images from these zip files and read as a Dataset directly. Rust is a very high performance language and can be interoped with Python. So we use Rust to implement a high performance image dataset reader.

## Why Zip not tar
Tar is an archive format which store files sequentially. So Tar is not suitable for random readings which is highly needed by a dataset.
Meanwhile Zip file has an internal index which can be used to find the offset of a file. So Zip is a good choice for a dataset.

## How to use
### Install

- preliminaries

  We need to install rust environment for your OS first.

- Convert tar files to zip files

  The directory rs/tar_file_utilities contains the code to convert tar files to zip files.

  ``` bash
  cd rs/tar_file_utilities
  cargo run -- <tar_input_dir> <zip_output_dir>
  ```


  The input directory should contain a list of tar files. The output directory will contain a list of zip files.

- Compile and install the Fast zip reader for your python environment

  ``` bash
  cd rs/fast_zip_reader
  maturin develop
  ```
  This step will generate a python package named fast_zip_reader and install it.

- Use the Fast zip reader

  ``` python
  import zip_fast_reader

    import time
    elapsed_time0 = time.time()
    zip_reader = zip_fast_reader.ZipReader('/media/yueyulin/TOUROS/images/laion400m_zip/batch0/00000.zip')
    elapsed_time0 = time.time() - elapsed_time0
    file_names = zip_reader.read_filenames()
    print(file_names)
    import time
    for file_name in file_names:
        elapsed_time = time.time()
        content = zip_reader.read_file_in_zip(file_name)
        elapsed_time = time.time() - elapsed_time
        print(file_name,':', len(content),':', elapsed_time)

    print('elapsed_time0:', elapsed_time0)
   ```

### Install pre-built wheels
Download the pre-built wheels from Baidu Cloud as follows:

链接: https://pan.baidu.com/s/1uzrWtwf5I9rIpMqgzjo3Kg?pwd=k2kc 提取码: k2kc 
--来自百度网盘超级会员v6的分享

Install the wheel:
``` bash
pip install zip_fast_reader-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
```

This wheel file is built for Python 3.10 on 64bits Linux.

To ensure the installation is successful, run the above python code to verify the it.