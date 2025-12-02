import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from model import UNet2D
from dataset import ImageToImageDataset
import numpy as np
from PIL import Image


def main():
    # Utility: save single-channel tensor [0,1] as PNG
    def tensor_to_image(t: torch.Tensor, out_path: Path):
        t = t.detach().cpu().clamp(0,1)
        if t.dim() == 3:
            t = t.squeeze(0)  # (C,H,W) -> (H,W)
        arr = (t.numpy()*255).astype(np.uint8)
        Image.fromarray(arr).save(out_path)

    # === Editable configuration ===
    INPUTS_DIR = r".\test_stipples\data_grads_v3_sample\source"
    TARGETS_DIR = r".\test_stipples\data_grads_v3_sample\target"
    OUT_DIR = Path(r".\unet_2d_outputs")
    BATCH_SIZE = 4
    LR = 1e-3
    EPOCHS = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset + loader
    ds = ImageToImageDataset(INPUTS_DIR, TARGETS_DIR, size=(512,512))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model (3DUNet-style 2D config)
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        fmaps=32,
        depth=4,
        norm='batch',
        activation='relu',
        dropout=0.0,
        residual=True,
        final_activation='sigmoid'
    ).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train loop
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        for inp, tar, _ in pbar:
            inp = inp.to(DEVICE)
            tar = tar.to(DEVICE)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inp.size(0)
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), OUT_DIR / f"unet2d_epoch_{epoch+1}.pt")

    # Quick eval on a few samples (save PNGs via utils)
    model.eval()
    with torch.no_grad():
        for i in range(3):
            inp, tar, name = ds[i]
            inp = inp.unsqueeze(0).to(DEVICE)
            pred = model(inp).squeeze(0)
            png_out = OUT_DIR / f"pred_{name}.png"
            tensor_to_image(pred, png_out)

    print("Training complete.")


if __name__ == "__main__":
    main()
