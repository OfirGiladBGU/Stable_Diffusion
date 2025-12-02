import torch
from pathlib import Path
from PIL import Image
import numpy as np
from model import UNet2D


def load_image_grayscale(path: Path, size=(512,512)):
    img = Image.open(path).convert('L').resize(size, Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return t


def save_tensor_png(t: torch.Tensor, out_path: Path):
    t = t.detach().cpu().clamp(0,1).squeeze(0).squeeze(0)  # (H,W)
    arr = (t.numpy()*255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def main():
    # === Editable configuration ===
    INPUTS_DIR = Path(r".\test_stipples\data_grads_v3_sample\source")
    CHECKPOINT = Path(r".\unet_2d_outputs\unet2d_epoch_5.pt")
    OUTPUT_DIR = Path(r".\unet_2d_outputs\predictions")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SIZE = (512,512)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build model (match training params)
    model = UNet2D(in_channels=1, out_channels=1, fmaps=32, depth=4, norm='batch', residual=True, final_activation='sigmoid').to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # Iterate over input images
    image_paths = sorted(list(INPUTS_DIR.glob('*.png')) + list(INPUTS_DIR.glob('*.jpg')))
    print(f"Predicting {len(image_paths)} images...")

    with torch.no_grad():
        for p in image_paths:
            inp = load_image_grayscale(p, size=SIZE).to(DEVICE)
            pred = model(inp)
            out_path = OUTPUT_DIR / f"pred_{p.stem}.png"
            save_tensor_png(pred, out_path)
            print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
