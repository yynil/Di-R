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