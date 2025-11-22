import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from typing import Tuple
import pathlib
from tqdm import tqdm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Papers:
# https://resources.mpi-inf.mpg.de/ProjectiveBlueNoise/ProjectiveBlueNoise.pdf


########################
# Single Image Metrics #
########################

# https://stackoverflow.com/questions/19867279/how-to-compute-power-spectrum-from-2d-fft
def log_power_spectrum_2d(numpy_2d: np.ndarray, 
                          epsilon: float = 1.0,
                          plot: bool = True) -> np.ndarray:
    """
    Compute the log power spectrum of a 2D image array.
    """    
    # Convert the numpy array to a PyTorch tensor
    tensor_2d = torch.from_numpy(numpy_2d).float().to(DEVICE)

    # DC (Direct Current) component removal
    tensor_2d -= tensor_2d.mean()

    # Compute the 2D Fourier Transform
    fourier_image = torch.fft.fft2(tensor_2d)

    # Shift the zero frequency component to the center
    fourier_image = torch.fft.fftshift(fourier_image)

    # Compute the power spectrum
    power_spectrum = torch.abs(fourier_image) ** 2

    # Add a small constant to avoid log(0)
    log_power_spectrum = torch.log(power_spectrum + epsilon)

    # Plot the log power spectrum if requested
    if plot:
        plt.imshow(log_power_spectrum.cpu().numpy(), cmap='gray', origin='lower')
        plt.colorbar(label='Log Power')
        plt.title('2D Power Spectrum')
        plt.xlabel('Frequency (kx)')
        plt.ylabel('Frequency (ky)')
        plt.show()
    else:
        return log_power_spectrum.cpu().numpy()


# https://stackoverflow.com/questions/29178635/how-to-calculate-1d-power-spectrum-from-2d-noise-power-spectrum-by-radial-averag
def radial_profile_2d(numpy_2d: np.ndarray, 
                      center: Tuple[int, int]=None, normalize: bool = True,
                      plot: bool = True) -> np.ndarray:
    """
    Compute the radial profile of a 2D power spectrum.
    """
    power_spectrum = log_power_spectrum_2d(numpy_2d, plot=False)

    if center is None:
        center = (power_spectrum.shape[1] // 2, power_spectrum.shape[0] // 2)
    
    y, x = np.indices(power_spectrum.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int32)

    tbin = np.bincount(r.ravel(), weights=power_spectrum.ravel())
    rbin = np.bincount(r.ravel())
    radial_profile = tbin / rbin

    if normalize:
        radial_profile = radial_profile / np.max(radial_profile)

    if plot:
        plt.plot(radial_profile)
        plt.title('Radial Profile of Power Spectrum')
        plt.xlabel('Radius')
        plt.ylabel('Average Power')
        plt.show()
    else:
        return radial_profile


def radial_profiles_difference_2d(numpy_2d_1: np.ndarray, numpy_2d_2: np.ndarray, 
                                  normalize: bool = True,
                                  plot: bool = True) -> np.ndarray:
    """
    Compute the absolute difference between two radial profiles.
    """
    radial_profile1 = radial_profile_2d(numpy_2d_1, normalize=normalize, plot=False)
    radial_profile2 = radial_profile_2d(numpy_2d_2, normalize=normalize, plot=False)

    min_length = min(len(radial_profile1), len(radial_profile2))
    profile_diff = np.abs(radial_profile1[:min_length] - radial_profile2[:min_length])

    if plot:
        plt.plot(profile_diff)
        plt.title('Radial Difference')
        plt.xlabel('Radius')
        plt.ylabel('Absolute Difference')
        plt.show()
    else:
        return profile_diff


# Test
def single_image_metrics():
    image1_path = fr".\test_stipples\input\GT.png"
    image2_path = fr".\test_stipples\input\PRED.png"

    # Open images
    image1_arr = np.array(PIL.Image.open(image1_path).convert("L"))
    image2_arr = np.array(PIL.Image.open(image2_path).convert("L"))

    # Plot spectra
    # log_power_spectrum_2d(image1_arr, plot=True)

    # Plot radial profile
    # radial_profile_2d(image1_arr, normalize=True, plot=True)

    # Difference between two images
    radial_profiles_difference_2d(image1_arr, image2_arr, normalize=True, plot=True)


##########################
# Multiple Image Metrics #
##########################

def mean_radial_profile_difference_2d(numpy_2d_list_1: list, numpy_2d_list_2: list, 
                                      normalize: bool = True, 
                                      plot: bool = True) -> np.ndarray:
    """
    Compute the mean radial profile difference between two lists of 2D images.
    """
    
    lists_length = min(len(numpy_2d_list_1), len(numpy_2d_list_2))
    max_length = 0
    profile_diff_list = []
    for i in tqdm(range(lists_length)):
        image1_arr = numpy_2d_list_1[i]
        image2_arr = numpy_2d_list_2[i]

        profile_diff = radial_profiles_difference_2d(image1_arr, image2_arr, normalize=normalize, plot=False)
        profile_diff_list.append(profile_diff)
        if len(profile_diff) > max_length:
            max_length = len(profile_diff)

    padded_profile_diff_list = np.array([np.pad(prof_diff, (0, max_length - len(prof_diff)), 'edge') for prof_diff in profile_diff_list])
    avg_profile_diff = np.mean(padded_profile_diff_list, axis=0)
    
    if plot:
        plt.plot(avg_profile_diff)
        plt.title('Radial Difference')
        plt.xlabel('Radius')
        plt.ylabel('Absolute Difference')
        plt.show()
    else:
        return avg_profile_diff


# Test
def multi_image_metrics():
    image_folder_path1 = fr".\test_stipples\data_grads_v3\source"
    image_folder_path2 = fr".\test_stipples\data_grads_v3\output_beta"

    image_paths1 = sorted(pathlib.Path(image_folder_path1).glob("*.*"))
    image_paths2 = sorted(pathlib.Path(image_folder_path2).glob("*.*"))

    image_arr_list1 = []
    image_arr_list2 = []
    images_count = min(len(image_paths1), len(image_paths2))
    for i in range(images_count):
        print(f"Processing image pair {i+1}/{images_count}")
        image1_arr = np.array(PIL.Image.open(image_paths1[i]).convert("L"))
        image2_arr = np.array(PIL.Image.open(image_paths2[i]).convert("L"))
        image_arr_list1.append(image1_arr)
        image_arr_list2.append(image2_arr)

    mean_radial_profile_difference_2d(image_arr_list1, image_arr_list2, normalize=True, plot=True)


def main():
    # single_image_metrics()
    multi_image_metrics()
    

if __name__ == "__main__":
    main()
