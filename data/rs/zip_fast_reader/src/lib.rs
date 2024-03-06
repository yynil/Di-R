use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::IntoPyDict;
use std::fs::File;
use zip::ZipArchive;
use std::io::Read;

#[pyclass]
pub struct ZipReader {
    archive: ZipArchive<File>,
}

#[pymethods]
impl ZipReader {
    #[new]
    #[args(zip_file = "String::new()")]
    fn new(zip_file: String) -> PyResult<Self> {
        let file = File::open(zip_file)?;
        let archive = match ZipArchive::new(file) {
            Ok(archive) => archive,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e))),
        };
        Ok(ZipReader { archive })
    }

    fn read_file_in_zip(&mut self, file_name: &str) -> PyResult<Vec<u8>> {
        let mut file = match self.archive.by_name(file_name) {
            Ok(file) => file,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e))),
        };
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        Ok(contents)
    }

    fn read_filenames(&mut self) -> PyResult<Vec<String>> {
        let mut file_names = Vec::new();
        for i in 0..self.archive.len() {
            let file = match self.archive.by_index(i) {
                Ok(file) => file,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e))),
            };
            file_names.push(file.name().to_string());
        }
        Ok(file_names)
    }
}


#[pymodule]
fn zip_fast_reader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ZipReader>()?;
    Ok(())
}