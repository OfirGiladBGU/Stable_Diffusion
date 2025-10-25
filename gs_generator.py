import numpy as np
from PIL import Image
import random
import math
import argparse
from scipy.ndimage import gaussian_filter

# === Gradient modes from math-equ-image-demo ===

def linear_gradient(X, Y, params):
    return np.clip((X + Y) * 0.5 + params.get('offset',0.0), 0.0, 1.0)

def sinusoidal_gradient(X, Y, params):
    return 0.5 + 0.5 * np.sin(params['freq'] * (X*X + Y*Y) + params['phase'])

def cosine_gradient(X, Y, params):
    return 0.5 + 0.5 * np.cos(params['freq'] * (X * Y) + params['phase'])

def radial_sinusoidal_gradient(X, Y, params):
    r = np.sqrt(X*X + Y*Y)
    return 0.5 + 0.5 * np.sin(params['freq'] * r + params['phase'])

def radial_cosine_gradient(X, Y, params):
    r = np.sqrt(X*X + Y*Y)
    return 0.5 + 0.5 * np.cos(params['freq'] * r + params['phase'])

# === Additional modes (waves, shapes, noise) ===

def wave_func(X, Y, params):
    return params['baseline'] + params['amplitude'] * np.cos(
        2 * math.pi * (params['freq_x'] * X + params['freq_y'] * Y + params['phase'])
    )

def radial_wave_func(X, Y, params):
    cx = params.get('cx', 0.5)
    cy = params.get('cy', 0.5)
    dx = X - cx
    dy = Y - cy
    r  = np.sqrt(dx*dx + dy*dy)
    return params['baseline'] + params['amplitude'] * np.cos(
        2 * math.pi * (params['freq_r'] * r + params['phase_r'])
    )

def noise_func(X, Y, params):
    val = np.random.rand(*X.shape)
    val = gaussian_filter(val, sigma=params['noise_blur_sigma'])
    return params['baseline'] + params['amplitude'] * (val - 0.5) * 2.0

def combined_shape_func(X, Y, params):
    cx = params['cx']
    cy = params['cy']
    r  = params['r']
    blend_width = params.get('blend_width', r*0.3)
    dx = X - cx
    dy = Y - cy
    dist = np.sqrt(dx*dx + dy*dy)
    t = np.clip((dist - r) / blend_width, 0.0, 1.0)
    mask = 1.0 - t
    inside_vals  = wave_func(X, Y, params['inside_params'])
    outside_vals = radial_wave_func(X, Y, params['outside_params'])
    return mask * inside_vals + (1.0 - mask) * outside_vals

# === Map function names to implementations ===

FUNC_MAP = {
    'Linear Gradient'            : linear_gradient,
    'Sinusoidal Gradient'       : sinusoidal_gradient,
    'Cosine Gradient'            : cosine_gradient,
    'Radial Sinusoidal Gradient': radial_sinusoidal_gradient,
    'Radial Cosine Gradient'     : radial_cosine_gradient,
    'Wave'                      : wave_func,
    'Radial Wave'               : radial_wave_func,
    'Noise'                     : noise_func,
    'Combined Shape'            : combined_shape_func
}

# === Parameter pickers ===

def pick_random_params(func_name, width, height):
    p = {}
    if func_name == 'Linear Gradient':
        p['offset'] = random.uniform(-0.2, 0.2)
    elif func_name in ('Sinusoidal Gradient', 'Cosine Gradient',
                       'Radial Sinusoidal Gradient', 'Radial Cosine Gradient'):
        p['freq']  = random.uniform(2.0, 8.0)
        p['phase'] = random.uniform(0.0, 2*math.pi)
    elif func_name == 'Wave':
        p['freq_x']   = random.uniform(1.0, 4.0)
        p['freq_y']   = random.uniform(1.0, 4.0)
        p['phase']    = random.uniform(0.0, 2*math.pi)
        p['amplitude']= random.uniform(0.05, 0.25)
        p['baseline'] = random.uniform(0.4, 0.6)
    elif func_name == 'Radial Wave':
        p['cx']       = random.uniform(0.3, 0.7)
        p['cy']       = random.uniform(0.3, 0.7)
        p['freq_r']   = random.uniform(1.5, 4.0)
        p['phase_r']  = random.uniform(0.0, 2*math.pi)
        p['amplitude']= random.uniform(0.05, 0.20)
        p['baseline'] = random.uniform(0.4, 0.6)
    elif func_name == 'Noise':
        p['noise_blur_sigma'] = random.uniform(width * 0.002, width * 0.01)
        p['amplitude']        = random.uniform(0.1, 0.35)
        p['baseline']         = random.uniform(0.3, 0.7)
    elif func_name == 'Combined Shape':
        p['cx'] = random.uniform(0.3, 0.7)
        p['cy'] = random.uniform(0.3, 0.7)
        p['r']  = random.uniform(0.2, 0.4)
        p['blend_width'] = random.uniform(p['r']*0.2, p['r']*0.5)
        inside = {
            'freq_x'   : random.uniform(1.0, 3.0),
            'freq_y'   : random.uniform(1.0, 3.0),
            'phase'    : random.uniform(0.0, 2*math.pi),
            'amplitude': random.uniform(0.05, 0.2),
            'baseline' : random.uniform(0.45, 0.55)
        }
        outside = {
            'cx'       : random.uniform(0.3, 0.7),
            'cy'       : random.uniform(0.3, 0.7),
            'freq_r'   : random.uniform(1.0, 3.0),
            'phase_r'  : random.uniform(0.0, 2*math.pi),
            'amplitude': random.uniform(0.02, 0.15),
            'baseline' : random.uniform(0.4, 0.6)
        }
        p['inside_params']  = inside
        p['outside_params'] = outside
    else:
        raise ValueError(f"Unknown function name: {func_name}")
    return p

# === Generate image ===

def generate_image(width=512, height=512, seed=None, blur_sigma=None, flags=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    x = np.linspace(0, 1, width,  endpoint=False)
    y = np.linspace(0, 1, height, endpoint=False)
    X, Y = np.meshgrid(x, y)

    allowed = [fn for fn, enabled in flags.items() if enabled]
    if not allowed:
        raise ValueError("No function modes enabled.")
    func_name = random.choice(allowed)
    params    = pick_random_params(func_name, width, height)
    print(f"Mode: '{func_name}' with params: {params}")

    val = FUNC_MAP[func_name](X, Y, params)

    if blur_sigma is None:
        blur_sigma = width * 0.003
    val = gaussian_filter(val, sigma=blur_sigma)

    val = np.clip(val, 0.0, 1.0)
    img_uint8 = (val * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='L')
    return img, func_name, params

# === Main execution ===

def main():
    parser = argparse.ArgumentParser(description="Generate grayscale images with combined gradient + form modes.")
    parser.add_argument('--width',      type=int,   default=512)
    parser.add_argument('--height',     type=int,   default=512)
    parser.add_argument('--seed',       type=int,   default=None)
    parser.add_argument('--count',      type=int,   default=10)
    parser.add_argument('--outprefix',  type=str, default='gen_gray')
    # Flags for all modes
    parser.add_argument('--enable_linear',                 action='store_true', default=True)
    parser.add_argument('--enable_sinusoidal',             action='store_true', default=True)
    parser.add_argument('--enable_cosine',                  action='store_true', default=True)
    parser.add_argument('--enable_radial_sinusoidal',      action='store_true', default=True)
    parser.add_argument('--enable_radial_cosine',          action='store_true', default=True)
    parser.add_argument('--enable_wave',                    action='store_true', default=True)
    parser.add_argument('--enable_radial_wave',             action='store_true', default=True)
    parser.add_argument('--enable_noise',                   action='store_true', default=True)
    parser.add_argument('--enable_combined_shape',          action='store_true', default=True)

    args = parser.parse_args()

    flags = {
        'Linear Gradient'            : args.enable_linear,
        'Sinusoidal Gradient'       : args.enable_sinusoidal,
        'Cosine Gradient'            : args.enable_cosine,
        'Radial Sinusoidal Gradient': args.enable_radial_sinusoidal,
        'Radial Cosine Gradient'     : args.enable_radial_cosine,
        'Wave'                      : args.enable_wave,
        'Radial Wave'               : args.enable_radial_wave,
        'Noise'                     : args.enable_noise,
        'Combined Shape'            : args.enable_combined_shape
    }

    for i in range(args.count):
        seed_val = args.seed if args.seed is not None else random.randint(0, 2**31-1)
        img, mode, params = generate_image(args.width, args.height, seed=seed_val, flags=flags)
        safe_mode = mode.replace(' ', '_')
        fname = f"{args.outprefix}_{safe_mode}_{seed_val}_{i:02d}.png"
        img.save(fname)
        print(f"Saved {fname}")

if __name__ == '__main__':
    # https://github.com/OfirGiladBGU/math-equ-image-demo
    main()
