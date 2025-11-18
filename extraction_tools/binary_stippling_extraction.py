from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Simple recursive inversion script.
# Edit INPUT_DIR / OUTPUT_DIR / OVERWRITE below as needed.

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def stipple_file(src: Path, dst: Path, threshold: int = 128, option: int = 0, point_color: int = 255):
    """
    Apply binary stippling to a grayscale image and save the result.
    Args:
        src: Source image path
        dst: Destination image path
        threshold: Grayscale threshold (0-255)
        option: Stippling option (0: simple, 1: NN-based)
        point_color: Color for stippled points (default 255 for white)
    """
    img = Image.open(src).convert("L")  # assume already grayscale, enforce
    arr = np.array(img, dtype=np.uint8)

    # Simple stippling: pixels below threshold -> 0, else 255
    if option == 0:
        result = np.where(arr < threshold, 0, 255).astype(np.uint8)
    
    # NN-based stippling: similar to option 0, but with simple noise reduction
    elif option == 1:
        result = np.where(arr < threshold, 0, 255).astype(np.uint8)

        # Simple noise reduction: if a white pixel has >4 white neighbors, keep it white
        padded = np.pad(result, pad_width=1, mode='constant', constant_values=0)
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                if padded[i, j] == point_color:
                    neighbors = padded[i-1:i+2, j-1:j+2]
                    if np.sum(neighbors == point_color) <= 4:
                        result[i-1, j-1] = 0
    else:
        raise ValueError(f"Unknown stippling option: {option}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Pillow infers 'L' for 2D uint8 arrays; avoid deprecated 'mode' arg
    Image.fromarray(result).save(dst)


def stipple_folders(input_dir: str, output_dir: str, threshold: int = 128, overwrite: bool = True):
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
        stipple_file(src, dst, threshold)  # let underlying libs raise if issues


def main():
    # === Editable configuration ===
    INPUT_DIR = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\test_stipples\input"
    OUTPUT_DIR = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\test_stipples\output"
    THRESHOLD = 128  # Stippling threshold (0-255)
    OVERWRITE = True  # Set False to skip if file already exists
    
    stipple_folders(INPUT_DIR, OUTPUT_DIR, THRESHOLD, OVERWRITE)


if __name__ == "__main__":
    main()
