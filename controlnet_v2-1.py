# =============================================
# Colab: Blue-noise → Inversion → SD 1.5 + ControlNet
# =============================================

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ------------------------------------------------------
# 1. Load ControlNet (Canny) + Stable Diffusion v2.1
# ------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "multimodalart/controlnet-sd21-canny-diffusers",  # has diffusion_pytorch_model.safetensors
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_xformers_memory_efficient_attention()

# ------------------------------------------------------
# 2. Load your blue-noise mask (replace with your own)
# ------------------------------------------------------
# Example: generate dummy blue-noise as random dots (replace with your 512x512 image)
res = 512
blue_noise = np.ones((res, res), dtype=np.float32) * 255
yy, xx = np.random.randint(0,res,5000), np.random.randint(0,res,5000)
blue_noise[yy,xx] = 0

img = Image.fromarray(blue_noise.astype(np.uint8)).convert("RGB")

# Invert the mask
img_inverted = Image.fromarray(255 - blue_noise.astype(np.uint8)).convert("RGB")

# ------------------------------------------------------
# 3. Run SD 2.1 with ControlNet using the inverted mask
# ------------------------------------------------------
# prompt = "highly detailed organic texture, cinematic lighting"
prompt = ""

result = pipe(
    prompt=prompt,
    image=img_inverted,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# ------------------------------------------------------
# 4. Plot input vs inverted vs generated result
# ------------------------------------------------------
fig, axes = plt.subplots(1,3, figsize=(15,5))
axes[0].imshow(img); axes[0].set_title("Original Blue Noise"); axes[0].axis("off")
axes[1].imshow(img_inverted); axes[1].set_title("Inverted Mask"); axes[1].axis("off")
axes[2].imshow(result); axes[2].set_title("Stable Diffusion Result"); axes[2].axis("off")
plt.show()
plt.savefig("blue_noise_inversion_results.png")
