from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Simple recursive inversion script.
# Edit INPUT_DIR / OUTPUT_DIR / OVERWRITE below as needed.

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def invert_file(src: Path, dst: Path):
    img = Image.open(src).convert("L")  # assume already grayscale, enforce
    arr = np.array(img, dtype=np.uint8)
    inv = 255 - arr
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Pillow infers 'L' for 2D uint8 arrays; avoid deprecated 'mode' arg
    Image.fromarray(inv).save(dst)


def invert_folders(input_dir: str, output_dir: str, overwrite: bool = True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for src in tqdm(list(input_dir.rglob("*"))):
        if not src.is_file():
            continue
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        if (not overwrite) and dst.exists():
            continue
        invert_file(src, dst)  # let underlying libs raise if issues


def main():
    # === Editable configuration ===
    INPUT_DIR = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\data_gradients_x\target"
    OUTPUT_DIR = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\data_gradients_x\target_inverted"
    OVERWRITE = True  # Set False to skip if file already exists
    
    invert_folders(INPUT_DIR, OUTPUT_DIR, OVERWRITE)


if __name__ == "__main__":
    main()
