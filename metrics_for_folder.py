import numpy as np
import PIL
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from metrics import log_power_spectrum_2d, radial_profile_2d, radial_profiles_difference_2d


def save_log_power_spectrum(image: np.ndarray, output_path: Path):
    """Compute and save log power spectrum for a single image."""
    spectrum = log_power_spectrum_2d(image, plot=False)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrum, cmap='gray', origin='lower')
    plt.colorbar(label='Log Power')
    plt.title('2D Power Spectrum')
    plt.xlabel('Frequency (kx)')
    plt.ylabel('Frequency (ky)')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_radial_profile(image: np.ndarray, output_path: Path):
    """Compute and save radial profile for a single image."""
    profile = radial_profile_2d(image, normalize=True, plot=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(profile)
    plt.title('Radial Profile of Power Spectrum')
    plt.xlabel('Radius')
    plt.ylabel('Average Power')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_profile_difference(image1: np.ndarray, image2: np.ndarray, output_path: Path):
    """Compute and save radial profile difference between two images."""
    diff = radial_profiles_difference_2d(image1, image2, normalize=True, plot=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(diff)
    plt.title('Radial Profile Difference')
    plt.xlabel('Radius')
    plt.ylabel('Absolute Difference')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_folder(folder_path: Path):
    """Process a folder: compute and save metrics for each image."""
    image_paths = sorted(folder_path.glob("*.png")) + sorted(folder_path.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {folder_path}")
        return {}, []
    
    folder_name = folder_path.name
    parent_dir = folder_path.parent
    
    # Create output directories
    log_power_dir = parent_dir / f"{folder_name}_log_power"
    radial_profile_dir = parent_dir / f"{folder_name}_radial_profile"
    log_power_dir.mkdir(exist_ok=True)
    radial_profile_dir.mkdir(exist_ok=True)
    
    images_dict = {}
    image_names = []
    
    print(f"Processing {len(image_paths)} images from {folder_name}...")
    for img_path in tqdm(image_paths):
        img_arr = np.array(PIL.Image.open(img_path).convert("L"))
        img_name = img_path.stem
        image_names.append(img_name)
        images_dict[img_name] = img_arr
        
        # Save log power spectrum
        log_power_path = log_power_dir / f"{img_name}.png"
        save_log_power_spectrum(img_arr, log_power_path)
        
        # Save radial profile
        radial_profile_path = radial_profile_dir / f"{img_name}.png"
        save_radial_profile(img_arr, radial_profile_path)
    
    return images_dict, image_names


def main():
    # === Editable configuration ===
    BASE_DIR = Path(r".\test_stipples\data_grads_v3_sample")
    
    FOLDERS = [
        "output_5_stippled",
        "output_10_stippled",
        "output_50_stippled",
        "target"
    ]
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Processing {len(FOLDERS)} folders...\n")
    
    # Process all folders and store images by name
    folder_data = {}
    for folder_name in FOLDERS:
        folder_path = BASE_DIR / folder_name
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*60}")
        images_dict, image_names = process_folder(folder_path)
        folder_data[folder_name] = {"images": images_dict, "names": image_names}
    
    # Compute differences between stippled folders and target
    target_data = folder_data.get("target")
    if not target_data or not target_data["images"]:
        print("\nWarning: No target images found, skipping difference computation.")
        return
    
    target_images = target_data["images"]
    target_names = set(target_data["names"])
    
    print(f"\n{'='*60}")
    print("Computing profile differences vs target...")
    print(f"{'='*60}")
    
    for folder_name in FOLDERS:
        if "stippled" in folder_name:
            print(f"\nComputing difference: {folder_name} vs target")
            stippled_data = folder_data[folder_name]
            stippled_images = stippled_data["images"]
            
            # Create diff output directory
            folder_path = BASE_DIR / folder_name
            diff_dir = folder_path.parent / f"{folder_name}_profiles_diff"
            diff_dir.mkdir(exist_ok=True)
            
            # Process each image that has a matching target
            common_names = [name for name in stippled_data["names"] if name in target_names]
            print(f"Processing {len(common_names)} matching image pairs...")
            
            for img_name in tqdm(common_names):
                stippled_img = stippled_images[img_name]
                target_img = target_images[img_name]
                diff_path = diff_dir / f"{img_name}.png"
                save_profile_difference(stippled_img, target_img, diff_path)
    
    print(f"\n{'='*60}")
    print("All metrics computed and saved!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
