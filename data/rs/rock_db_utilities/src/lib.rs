use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::IntoPyDict;
use rocksdb::{DB, Options};
use std::env;
use std::convert::TryInto;
use std::fs;

#[pyclass]
pub struct RocksDBWrapper {
    db: DB,
    len: usize,
}

#[pymethods]
impl RocksDBWrapper {
    #[new]
    pub fn new(path: &str) -> PyResult<RocksDBWrapper> {
        let mut options = Options::default();
        options.create_if_missing(true);
        
        let db = DB::open_cf_for_read_only(&options, path, &["default"], false).unwrap();

        let max_id_key :&str = "MAX_ID";
        let len = db.get(max_id_key).unwrap().map(|v| {
            let bytes: [u8; 4] = v[..4].try_into().unwrap(); // Convert the Vec<u8> into a [u8; 4]
            u32::from_le_bytes(bytes) as usize // Convert the [u8; 4] into a u32, then cast to usize
        }).unwrap_or(0);

        Ok(RocksDBWrapper { db, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, index: usize) -> PyResult<Option<(usize, Vec<u8>)>>{
        let key = index.to_le_bytes();
        Ok(self.db.get(&key).ok().and_then(|value| {
            value.map(|v| {
                let (class_id_bytes, image_binaries) = v.split_at(std::mem::size_of::<usize>());
                let class_id = usize::from_le_bytes(class_id_bytes.try_into().unwrap());
                (class_id, image_binaries.to_vec())
            })
        }))
    }
}

#[pymodule]
fn imagenet_rocksdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RocksDBWrapper>()?;
    Ok(())
}