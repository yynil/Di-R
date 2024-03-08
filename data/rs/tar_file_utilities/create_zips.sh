#!/bin/bash
#check if we have two arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input directory> <output directory>" 
    exit 1
fi
# iterate over the directories of the input directory
for dir in $1/*; do
    # check if the directory is a directory
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        # get the name of the directory
        dir_name=$(basename "$dir")
        # making zip the directory to the output directory
        echo "making zip to : $2/$dir_name"
        ./target/release/main $dir $2/$dir_name
        echo "zip directory $2/$dir_name created"
    else
        echo "Skipping file: $dir"
        continue
    fi
done