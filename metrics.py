import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_power_spectrum_2d(numpy_2d: np.ndarray) -> np.ndarray:
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
    log_power_spectrum = torch.log(power_spectrum + 1.0)

    # Plot the log power spectrum
    plt.imshow(log_power_spectrum.cpu().numpy(), cmap='gray', origin='lower')
    plt.colorbar(label='Log Power')
    plt.title('2D Power Spectrum')
    plt.xlabel('Frequency (kx)')
    plt.ylabel('Frequency (ky)')
    plt.show()


def main():
    image1_path = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\test_stipples\input\GT.png"
    # image2_path = r"D:\AllProjects\PycharmProjects\Stable_Diffusion\test_stipples\input\PRED.png"

    image1_arr = np.array(PIL.Image.open(image1_path).convert("L"))
    # image2_arr = np.array(PIL.Image.open(image2_path).convert("L"))

    log_power_spectrum_2d(image1_arr)
    # log_power_spectrum_2d(image2_arr)


if __name__ == "__main__":
    main()