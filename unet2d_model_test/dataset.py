from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

class ImageToImageDataset(Dataset):
    """
    Simple paired dataset: inputs and targets folders.
    Assumes matching filenames in both folders.
    Loads grayscale images and returns tensors normalized to [0,1].
    """
    def __init__(self, inputs_dir: str, targets_dir: str, size: Tuple[int,int]=(512,512)):
        self.inputs_dir = Path(inputs_dir)
        self.targets_dir = Path(targets_dir)
        self.size = size
        self.items = []
        input_paths = sorted([p for p in self.inputs_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTS])
        for ip in input_paths:
            tp = self.targets_dir / ip.name
            if tp.exists():
                self.items.append((ip, tp))
        if not self.items:
            raise RuntimeError("No matching input/target pairs found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip, tp = self.items[idx]
        inp = Image.open(ip).convert("L").resize(self.size, Image.BICUBIC)
        tar = Image.open(tp).convert("L").resize(self.size, Image.BICUBIC)
        inp_arr = np.array(inp, dtype=np.float32)/255.0
        tar_arr = np.array(tar, dtype=np.float32)/255.0
        inp_t = torch.from_numpy(inp_arr).unsqueeze(0)
        tar_t = torch.from_numpy(tar_arr).unsqueeze(0)
        return inp_t, tar_t, ip.name
