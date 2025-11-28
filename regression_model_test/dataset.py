import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, List
import glob


class PointSetDataset(Dataset):
    """
    Dataset for image-to-pointset regression.
    
    Loads paired images:
    - Source: grayscale input images from source_dir
    - Target: ground truth point locations stored as PNG images (black pixels = points)
    
    Returns:
        image: (1, H, W) normalized float tensor
        points: (N, 2) normalized float tensor in [0, 1], where N is number of black pixels
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        normalize_images: bool = True,
        black_threshold: int = 128,  # Pixels below this value are considered "black" (points)
    ):
        """
        Args:
            source_dir: Directory containing input grayscale images
            target_dir: Directory containing ground truth point images (black pixels = points)
            img_size: Expected image size (H, W)
            normalize_images: Whether to normalize images to [0, 1]
            black_threshold: Threshold for detecting black pixels (points) in target images
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.img_size = img_size
        self.normalize_images = normalize_images
        self.black_threshold = black_threshold
        
        # Get all PNG files from source directory
        self.source_files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
        
        if len(self.source_files) == 0:
            raise ValueError(f"No PNG files found in {source_dir}")
        
        # Verify corresponding target files exist
        self._verify_pairs()
        
        print(f"Loaded {len(self.source_files)} image pairs")
    
    def _verify_pairs(self):
        """Verify that all source files have corresponding target files"""
        missing = []
        for src_path in self.source_files:
            filename = os.path.basename(src_path)
            target_path = os.path.join(self.target_dir, filename)
            if not os.path.exists(target_path):
                missing.append(filename)
        
        if missing:
            raise ValueError(f"Missing target files for: {missing[:10]}... ({len(missing)} total)")
    
    def _extract_points_from_image(self, target_img: np.ndarray) -> np.ndarray:
        """
        Extract point coordinates from target image by finding black pixels.
        
        Args:
            target_img: (H, W) grayscale image array
            
        Returns:
            points: (N, 2) array of (x, y) coordinates normalized to [0, 1]
        """
        # Find black pixels (points)
        mask = target_img < self.black_threshold
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) == 0:
            # No points found - return a single point at center as fallback
            print(f"Warning: No black pixels found in target image")
            x_coords = np.array([self.img_size[1] // 2])
            y_coords = np.array([self.img_size[0] // 2])
        
        # Normalize to [0, 1]
        x_norm = x_coords / (self.img_size[1] - 1)
        y_norm = y_coords / (self.img_size[0] - 1)
        
        # Stack as (N, 2) with (x, y) format
        points = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
        
        return points
    
    def __len__(self) -> int:
        return len(self.source_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: (1, H, W) tensor
            points: (N, 2) tensor of normalized coordinates
        """
        # Load source image
        src_path = self.source_files[idx]
        src_img = Image.open(src_path).convert('L')  # Convert to grayscale
        src_img = src_img.resize(self.img_size[::-1])  # PIL uses (W, H)
        src_array = np.array(src_img, dtype=np.float32)
        
        # Normalize to [0, 1]
        if self.normalize_images:
            src_array = src_array / 255.0
        
        # Convert to tensor (1, H, W)
        image_tensor = torch.from_numpy(src_array).unsqueeze(0)
        
        # Load target image and extract points
        filename = os.path.basename(src_path)
        target_path = os.path.join(self.target_dir, filename)
        target_img = Image.open(target_path).convert('L')
        target_img = target_img.resize(self.img_size[::-1])
        target_array = np.array(target_img)
        
        # Extract points from target image
        points = self._extract_points_from_image(target_array)
        points_tensor = torch.from_numpy(points)
        
        return image_tensor, points_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Custom collate function to handle variable number of points per sample.
    
    Args:
        batch: List of (image, points) tuples
        
    Returns:
        images: (B, 1, H, W) batched images
        points_list: List of (N_i, 2) point tensors (variable length)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    points_list = [item[1] for item in batch]
    
    return images, points_list


def collate_fn_fixed_size(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], 
    num_points: int = 5000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function that samples/pads to a fixed number of points.
    
    Args:
        batch: List of (image, points) tuples
        num_points: Fixed number of points to return per sample
        
    Returns:
        images: (B, 1, H, W) batched images
        points: (B, num_points, 2) batched points
        masks: (B, num_points) boolean mask (True = valid point, False = padding)
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    batch_size = len(batch)
    
    points_batch = torch.zeros(batch_size, num_points, 2)
    masks = torch.zeros(batch_size, num_points, dtype=torch.bool)
    
    for i, (_, points) in enumerate(batch):
        n_points = points.shape[0]
        
        if n_points >= num_points:
            # Randomly sample if we have more points than needed
            indices = torch.randperm(n_points)[:num_points]
            points_batch[i] = points[indices]
            masks[i] = True
        else:
            # Use all points and pad with zeros
            points_batch[i, :n_points] = points
            masks[i, :n_points] = True
            # Remaining entries stay as zeros with mask=False
    
    return images, points_batch, masks


if __name__ == "__main__":
    # Test the dataset
    source_dir = r".\data_grads_v3\source"
    target_dir = r".\data_grads_v3\target"
    
    dataset = PointSetDataset(source_dir, target_dir)
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    image, points = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Number of points: {points.shape[0]}")
    print(f"  Points shape: {points.shape}")
    print(f"  Points range: [{points.min():.3f}, {points.max():.3f}]")
    print(f"  First 5 points:\n{points[:5]}")
    
    # Test collate functions
    from torch.utils.data import DataLoader
    
    print("\n" + "="*50)
    print("Testing variable-length collate function:")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    images, points_list = next(iter(loader))
    print(f"  Batch images shape: {images.shape}")
    print(f"  Number of samples: {len(points_list)}")
    print(f"  Points per sample: {[p.shape[0] for p in points_list]}")
    
    print("\n" + "="*50)
    print("Testing fixed-size collate function:")
    loader_fixed = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=lambda b: collate_fn_fixed_size(b, num_points=5000)
    )
    images, points, masks = next(iter(loader_fixed))
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch points shape: {points.shape}")
    print(f"  Batch masks shape: {masks.shape}")
    print(f"  Valid points per sample: {masks.sum(dim=1).tolist()}")
