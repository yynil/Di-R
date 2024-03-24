use rocksdb::{DB, Options};
use std::env;
use std::convert::TryInto;
use std::fs;
pub struct RocksDBWrapper {
    db: DB,
    len: usize,
}

impl RocksDBWrapper {
    pub fn new(path: &str) -> RocksDBWrapper {
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = DB::open(&options, path).unwrap();

        let max_id_key :&str = "MAX_ID";
        let len = db.get(max_id_key).unwrap().map(|v| {
            let bytes: [u8; 4] = v[..4].try_into().unwrap(); // Convert the Vec<u8> into a [u8; 4]
            u32::from_le_bytes(bytes) as usize // Convert the [u8; 4] into a u32, then cast to usize
        }).unwrap_or(0);

        RocksDBWrapper { db, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, index: usize) -> Option<(usize, Vec<u8>)> {
        let key = index.to_le_bytes();
        self.db.get(&key).ok().and_then(|value| {
            value.map(|v| {
                let (class_id_bytes, image_binaries) = v.split_at(std::mem::size_of::<usize>());
                let class_id = usize::from_le_bytes(class_id_bytes.try_into().unwrap());
                (class_id, image_binaries.to_vec())
            })
        })
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} <RocksDB directory>", args[0]);
        return;
    }
    let path = &args[1];
    let wrapper = RocksDBWrapper::new(path);
    println!("Length of RocksDB: {}", wrapper.len()); 
    let (class_id, image_binaries) = wrapper.get(0).unwrap();
    println!("Class ID: {}", class_id);
    fs::write("image_0.jpg", &image_binaries).expect("Unable to write file");
    let (class_id, image_binaries) = wrapper.get(1281166).unwrap();
    println!("Class ID: {}", class_id);    
    fs::write("image_1281166.jpg", &image_binaries).expect("Unable to write file");
    let (class_id, image_binaries) = wrapper.get(1000).unwrap();
    println!("Class ID: {}", class_id);    
    fs::write("image_1000.jpg", &image_binaries).expect("Unable to write file");
}