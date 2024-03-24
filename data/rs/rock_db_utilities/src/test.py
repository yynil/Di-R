import imagenet_rocksdb

path="/home/yueyulin/data/images/ILSVRC2012_rocksdb_auth"

reader = imagenet_rocksdb.RocksDBWrapper(path)
print(reader.len())
class_id,data = reader.get(0)
print(class_id)
print(len(data))
image_name = "image_0.JPEG"
with open(image_name, 'wb') as f:
    f.write(bytes(data))