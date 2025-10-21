# Stable_Diffusion

## General Dataset Build (For ControlNet)

1. Create a folder called `data`
2. Put all the images inside `data/original`
3. Run the script [fast_stippling_generator](fast_stippling_generator.py) to build the dataset like `fill50k`.
4. Copy the folder to `ControlNet` training path for training.


## How to create each dataset

### Lattice

1. Put the `.tar.gz` files on the dir `downloads`
2. Run the scripts in the following order:
    - [extract_tar_gz](extraction_tools/extract_tar_gz.py)
    - [extract_npz](extraction_tools/extract_npz.py)
    - [convert_npy_to_2d](extraction_tools/convert_npy_to_2d.py)
3. See section (General Dataset Build).
