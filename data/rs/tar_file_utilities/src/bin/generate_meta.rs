use std::collections::HashSet;
use std::env;
use std::ffi::OsStr;
use std::path::Path;
use std::fs::File;
use tar::Archive;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let fname = &args[1];

    let file = File::open(fname)?;
    let mut archive = Archive::new(file);

    let mut files = HashSet::new();
    let image_exts = vec!["jpg", "jpeg", "png", "gif", "bmp", "tiff", "ico", "jfif", "webp"];

    for file in archive.entries()? {
        let mut file = file?;
        let outpath = file.path()?;

        if let Some(stem) = outpath.file_stem() {
            if let Some(stem_str) = stem.to_str() {
                if let Some(extension) = outpath.extension() {
                    if let Some(extension_str) = extension.to_str() {
                        let lower_extension_str = extension_str.to_lowercase();
                        if image_exts.contains(&lower_extension_str.as_str()) || lower_extension_str == "json" || lower_extension_str == "txt" {
                            files.insert(stem_str.to_string());
                        }
                    }
                }
            }
        }
    }

    //print archive file length
    println!("Archive file length: {}", files.len());

    let mut count = 0;
    for base_name in &files {
        if files.contains(&(base_name.clone() + ".json")) && files.contains(&(base_name.clone() + ".txt")) {
            count += 1;
        }
    }

    println!("Total number of valid triplets: {}", count);

    Ok(())
}