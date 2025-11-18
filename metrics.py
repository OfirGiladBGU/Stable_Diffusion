import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from typing import Tuple


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Papers:
# https://resources.mpi-inf.mpg.de/ProjectiveBlueNoise/ProjectiveBlueNoise.pdf

# https://stackoverflow.com/questions/19867279/how-to-compute-power-spectrum-from-2d-fft
def log_power_spectrum_2d(numpy_2d: np.ndarray, plot: bool = True) -> np.ndarray:
    """
    Compute the log power spectrum of a 2D image array.
    """    
    # Convert the numpy array to a PyTorch tensor
    tensor_2d = torch.from_numpy(numpy_2d).float().to(DEVICE)

    # Compute the 2D Fourier Transform
    fourier_image = torch.fft.fft2(tensor_2d)

    # Shift the zero frequency component to the center
    fourier_image = torch.fft.fftshift(fourier_image)

    # Compute the power spectrum
    power_spectrum = torch.abs(fourier_image) ** 2

    # Add a small constant to avoid log(0)
    log_power_spectrum = torch.log1p(power_spectrum)

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
def radial_profile(numpy_2d: np.ndarray, center: Tuple[int, int]=None, plot: bool = True) -> np.ndarray:
    """
    Compute the radial profile of a 2D power spectrum.
    """
    power_spectrum = log_power_spectrum_2d(numpy_2d, plot=False)

    if center is None:
        center = (power_spectrum.shape[1] // 2, power_spectrum.shape[0] // 2)
    
    y, x = np.indices(power_spectrum.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int16)

    tbin = np.bincount(r.ravel(), weights=power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    if plot:
        plt.plot(radialprofile)
        plt.title('Radial Profile of Power Spectrum')
        plt.xlabel('Radius')
        plt.ylabel('Average Power')
        plt.show()
    else:
        return radialprofile


def radial_difference(numpy_2d_1: np.ndarray, numpy_2d_2: np.ndarray, plot: bool = True) -> np.ndarray:
    """
    Compute the absolute difference between two radial profiles.
    """
    profile1 = radial_profile(numpy_2d_1, plot=False)
    profile2 = radial_profile(numpy_2d_2, plot=False)

    min_length = min(len(profile1), len(profile2))
    profile_diff = np.abs(profile1[:min_length] - profile2[:min_length])
    if plot:
        plt.plot(profile_diff)
        plt.title('Radial Difference')
        plt.xlabel('Radius')
        plt.ylabel('Absolute Difference')
        plt.show()
    else:
        return profile_diff


def main():
    image1_path = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\test_stipples\input\GT.png"
    image2_path = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\test_stipples\input\PRED.png"

    # Open images
    image1_arr = np.array(PIL.Image.open(image1_path).convert("L"))
    image2_arr = np.array(PIL.Image.open(image2_path).convert("L"))

    # Plot spectra
    # log_power_spectrum_2d(image1_arr, plot=True)

    # Plot radial profile
    # radial_profile(image1_arr, plot=True)

    # Difference between two images
    radial_difference(image1_arr, image2_arr, plot=True)



if __name__ == "__main__":
    main()