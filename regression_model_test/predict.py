import os
# Fix for OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional, List
import glob

import pathlib
import sys
ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

from regression_model_test.model import ImageToPointSet, normalize_to_pixels


def load_model(checkpoint_path: str, num_points: int = 5000, 
               base_channels: int = 32, latent_dim: int = 512,
               device: torch.device = None) -> ImageToPointSet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        num_points: Number of points the model predicts
        base_channels: Base channels in encoder
        latent_dim: Latent dimension
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = ImageToPointSet(
        num_points=num_points,
        base_channels=base_channels,
        latent_dim=latent_dim
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.6f}")
    
    return model


def preprocess_image(image_path: str, img_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path: Path to image file
        img_size: Target image size (H, W)
        
    Returns:
        image_tensor: (1, 1, H, W) preprocessed image tensor
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(img_size[::-1])  # PIL uses (W, H)
    
    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


@torch.no_grad()
def predict_points(model: ImageToPointSet, image_path: str, 
                   img_size: Tuple[int, int] = (512, 512),
                   device: torch.device = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict point coordinates from an image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        img_size: Image size (H, W)
        device: Device to run inference on
        
    Returns:
        points_norm: (N, 2) normalized coordinates in [0, 1]
        points_px: (N, 2) pixel coordinates
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Preprocess image
    img_tensor = preprocess_image(image_path, img_size).to(device)
    
    # Predict
    points_norm = model(img_tensor)  # (1, N, 2)
    points_norm = points_norm[0].cpu().numpy()  # (N, 2)
    
    # Convert to pixel coordinates
    points_px_tensor = normalize_to_pixels(
        torch.from_numpy(points_norm).unsqueeze(0), 
        img_size
    )
    points_px = points_px_tensor[0].numpy()
    
    return points_norm, points_px


def visualize_prediction(image_path: str, points_px: np.ndarray, 
                        output_path: Optional[str] = None,
                        title: str = "Predicted Points",
                        point_size: float = 0.5,
                        point_color: str = 'red',
                        show: bool = True,
                        pred_img_size: Optional[Tuple[int, int]] = None):
    """
    Visualize predicted points overlaid on the input image.
    
    Args:
        image_path: Path to input image
        points_px: (N, 2) predicted points in pixel coordinates
        output_path: Optional path to save visualization
        title: Plot title
        point_size: Size of points in visualization
        point_color: Color of points
        show: Whether to display the plot
    """
    # Load image in original format without conversion
    img = Image.open(image_path)
    # If a prediction image size is provided (e.g., 512x512), resize the image for display
    if pred_img_size is not None:
        img = img.resize((pred_img_size[1], pred_img_size[0]))  # PIL uses (W,H)
    img_array = np.array(img)
    
    # Get original dimensions
    height, width = img_array.shape[:2] if img_array.ndim > 1 else img_array.shape + (1,)
    
    # Create figure sized to the display image
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(width/dpi, height/dpi), dpi=dpi)

    # Display image with original colors and no interpolation
    if img_array.ndim == 2:  # Grayscale
        ax.imshow(img_array, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    else:  # Color
        ax.imshow(img_array, interpolation='nearest')
    
    # Overlay points
    ax.scatter(points_px[:, 0], points_px[:, 1], 
              c=point_color, s=point_size, alpha=0.8, edgecolors='none')
    
    ax.set_title(f"{title}\n({len(points_px)} points)", fontsize=10, pad=5)
    ax.axis('off')
    
    # Remove all padding
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved visualization to: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def create_stipple_image(points_px: np.ndarray, img_size: Tuple[int, int] = (512, 512),
                        background: int = 255, point_value: int = 0) -> np.ndarray:
    """
    Create a stipple image from point coordinates.
    
    Args:
        points_px: (N, 2) points in pixel coordinates
        img_size: Image size (H, W)
        background: Background pixel value
        point_value: Point pixel value
        
    Returns:
        stipple_img: (H, W) stipple image
    """
    stipple_img = np.full(img_size, background, dtype=np.uint8)
    
    # Convert points to integer coordinates and clip to image bounds
    points_int = np.round(points_px).astype(int)
    points_int[:, 0] = np.clip(points_int[:, 0], 0, img_size[1] - 1)
    points_int[:, 1] = np.clip(points_int[:, 1], 0, img_size[0] - 1)
    
    # Set point pixels
    stipple_img[points_int[:, 1], points_int[:, 0]] = point_value
    
    return stipple_img


def save_points(points: np.ndarray, output_path: str, format: str = 'npy'):
    """
    Save predicted points to file.
    
    Args:
        points: (N, 2) point coordinates
        output_path: Output file path
        format: Format to save ('npy', 'txt', 'csv')
    """
    if format == 'npy':
        np.save(output_path, points)
    elif format == 'txt':
        np.savetxt(output_path, points, fmt='%.6f', delimiter=' ')
    elif format == 'csv':
        np.savetxt(output_path, points, fmt='%.6f', delimiter=',', 
                  header='x,y', comments='')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved points to: {output_path}")


def batch_predict(model: ImageToPointSet, image_dir: str, output_dir: str,
                 img_size: Tuple[int, int] = (512, 512),
                 save_visualizations: bool = True,
                 save_stipple_images: bool = True,
                 save_points_format: str = 'npy',
                 device: torch.device = None):
    """
    Run batch prediction on all images in a directory.
    
    Args:
        model: Trained model
        image_dir: Directory containing input images
        output_dir: Directory to save outputs
        img_size: Image size
        save_visualizations: Whether to save visualization overlays
        save_stipple_images: Whether to save stipple images
        save_points_format: Format to save point coordinates
        device: Device to run inference on
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    if save_stipple_images:
        stipple_dir = os.path.join(output_dir, 'stipple_images')
        os.makedirs(stipple_dir, exist_ok=True)
    if save_points_format:
        points_dir = os.path.join(output_dir, 'points')
        os.makedirs(points_dir, exist_ok=True)
    
    # Get all image files
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    image_paths += sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    image_paths += sorted(glob.glob(os.path.join(image_dir, '*.jpeg')))
    
    print(f"\nProcessing {len(image_paths)} images from {image_dir}")
    print(f"Output directory: {output_dir}\n")
    
    from tqdm import tqdm
    for img_path in tqdm(image_paths, desc="Predicting"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Predict points
        points_norm, points_px = predict_points(model, img_path, img_size, device)
        
        # Save visualization
        if save_visualizations:
            vis_path = os.path.join(vis_dir, f"{basename}_vis.png")
            visualize_prediction(img_path, points_px, vis_path, 
                               title=basename, show=False)
        
        # Save stipple image
        if save_stipple_images:
            stipple_img = create_stipple_image(points_px, img_size)
            stipple_path = os.path.join(stipple_dir, f"{basename}_stipple.png")
            Image.fromarray(stipple_img).save(stipple_path)
        
        # Save points
        if save_points_format:
            points_path = os.path.join(points_dir, f"{basename}_points.{save_points_format}")
            save_points(points_px, points_path, save_points_format)
    
    print(f"\n✅ Batch prediction completed!")
    print(f"Results saved to: {output_dir}")


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # === Editable configuration ===
    checkpoint = r"./regression_outputs/pointset_regression_20251203_010248/checkpoints/best_model.pth"  # set your checkpoint path
    # Choose either single image or directory
    image = None  # e.g., r"./data_grads_v3_sample/source/sample.png" or None
    image_dir = r"./data_grads_v3_sample/source"  # used for batch mode when image is None
    output = './regression_outputs/pointset_regression_20251203_010248/predictions'
    # Model and preprocessing
    num_points = 5000
    base_channels = 32
    latent_dim = 512
    img_size = 512
    # Visualization
    no_show = True
    point_size = 0.5
    point_color = 'red'
    # Outputs control
    save_stipple = True
    save_points = True
    points_format = 'npy'  # 'npy' | 'txt' | 'csv'
    save_visualizations = True  # batch mode overlays
    
    # Load model
    model = load_model(
        checkpoint,
        num_points=num_points,
        base_channels=base_channels,
        latent_dim=latent_dim,
        device=device
    )
    
    # Single image or batch prediction
    if image:
        # Single image prediction
        print(f"\nPredicting points for: {image}")
        
        points_norm, points_px = predict_points(
            model, image, 
            img_size=(img_size, img_size),
            device=device
        )
        
        print(f"\nPredicted {len(points_px)} points")
        print(f"  Normalized range: [{points_norm.min():.3f}, {points_norm.max():.3f}]")
        print(f"  Pixel range: [{points_px.min():.1f}, {points_px.max():.1f}]")
        
        # Visualize
        output_path = None
        if output:
            os.makedirs(output, exist_ok=True)
            basename = os.path.splitext(os.path.basename(image))[0]
            output_path = os.path.join(output, f"{basename}_prediction.png")
        
        visualize_prediction(
            image, points_px, 
            output_path=output_path,
            point_size=point_size,
            point_color=point_color,
            show=not no_show
        )
        
        # Save stipple image
        if save_stipple:
            stipple_img = create_stipple_image(points_px, (img_size, img_size))
            stipple_path = os.path.join(output, f"{basename}_stipple.png")
            Image.fromarray(stipple_img).save(stipple_path)
            print(f"Saved stipple image to: {stipple_path}")
        
        # Save points
        if save_points:
            points_path = os.path.join(output, f"{basename}_points.{points_format}")
            save_points(points_px, points_path, points_format)
    
    elif image_dir:
        # Batch prediction
        batch_predict(
            model, image_dir, output,
            img_size=(img_size, img_size),
            save_visualizations=save_visualizations,
            save_stipple_images=save_stipple,
            save_points_format=points_format if save_points else None,
            device=device
        )
    
    else:
        print("Error: Must specify either --image or --image_dir")


if __name__ == "__main__":
    main()
