use std::fs::{self, File};
use std::io::{self,};
use std::path::{Path};
use tar::Archive;
use zip::write::FileOptions;
use zip::CompressionMethod::Stored;
use zip::ZipWriter;
use indicatif::{ProgressBar, ProgressStyle};

pub fn tar_to_zip(input_dir: &str, output_dir: &str) -> io::Result<()> {
    let input_path = Path::new(input_dir);
    let output_path = Path::new(output_dir);

    let files_count = fs::read_dir(&input_path)?.count();
    let pb_style = ProgressStyle::default_bar()
    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
    .unwrap()
    .progress_chars("#>-");

    let bar = ProgressBar::new(files_count as u64);
    bar.set_style(pb_style);
    
    for entry in fs::read_dir(input_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("tar") {
            let tar_file = File::open(&path)?;
            let mut archive = Archive::new(tar_file);

            let zip_file_name = path.file_stem().unwrap().to_str().unwrap().to_owned() + ".zip";
            let zip_file_path = output_path.join(zip_file_name);
            let zip_file = File::create(zip_file_path)?;
            let mut zip = ZipWriter::new(zip_file);

            for file in archive.entries()? {
                let mut file = file?;
                let options = FileOptions::default()
                    .compression_method(Stored)
                    .unix_permissions(file.header().mode().unwrap_or(0o755));
                zip.start_file(file.path()?.to_str().unwrap(), options)?;
                io::copy(&mut file, &mut zip)?;
            }

            zip.finish()?;
            bar.inc(1);
        }
    }

    bar.finish();

    Ok(())
}