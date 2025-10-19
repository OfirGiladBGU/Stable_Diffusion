# High-performance, GPU-parallel, density-controlled blue-noise sampling
# =============================================================================
# Academic-quality, readable, single-cell implementation for Colab/PyTorch.
#
# Key ideas
# ---------
# • You provide ANY density as a Python lambda:  rho(u, v) on [0,1]×[0,1].
# • We generate a quasi-blue-noise *uniform* set via stratified jittered grid (fully vectorized).
# • We then **select** a subset of points *according to your density* using the Gumbel-Top-K trick
#   (parallel, without replacement), which preserves good spacing from the stratified pool.
# • No O(N^2) loops, no per-point neighbor scans. Everything is batched and GPU-friendly.
#
# What you get
# ------------
# • `sample_blue_noise_density(...)` returns integer pixel coords (n_points×2) aligned to `res×res`.
# • Side-by-side plot: grayscale ramp image (your density) and the sampled points.
#
# Notes
# -----
# • For “ramp” densities (linear, vertical, radial, etc.), just pass a lambda. Examples below.
# • If later you require stricter minimal-distance enforcement, add a single grid-based culling pass;
#   the current method already provides very good spacing for training data due to stratification.

import os
import math
import sys

# --- Environment fixes for OpenMP / MKL on Windows ---
# Some Windows Python packages (MKL, OpenMP-enabled libraries) can cause this error:
#   OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# To avoid that at runtime, set thread limits for BLAS/OpenMP runtimes and allow a
# duplicate OpenMP runtime as a last-resort fallback. Placing these env vars before
# importing heavy numeric libraries (torch, numpy, etc.) prevents the crash.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
# Unsafe but pragmatic fallback on Windows when multiple OpenMP runtimes exist
# (see script output message about KMP_DUPLICATE_LIB_OK). If you prefer not to
# allow duplicates, remove or set this to 'FALSE' and resolve conflicting packages.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
print("=== ALL IMPORTS DONE ===")
sys.stdout.flush()


# --- Core Logic ---
@torch.no_grad()
def _make_jittered_grid(n_points: int, oversample: float) -> torch.Tensor:
    """
    Stratified jittered grid in [0,1]^2.
    Generates M = ceil(n_points * oversample) points, then shuffles.

    Returns: (M, 2) tensor on `device`, coordinates in [0,1].
    """
    M = int(math.ceil(n_points * max(1.0, oversample)))
    G = int(math.ceil(math.sqrt(M)))
    ix = torch.arange(G, device=device, dtype=torch.float32)
    iy = torch.arange(G, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(ix, iy, indexing="xy")

    # Stratified: one random jitter per cell
    mesh_grid = torch.stack([X, Y], dim=-1)
    jitter = torch.rand((G, G, 2), device=device)
    pts = (mesh_grid + jitter) / G
    pts = pts.view(-1, 2)
    idx = torch.randperm(pts.shape[0], device=device)[:M]
    return pts[idx]


@torch.no_grad()
def _eval_density_on_grid(res: int, density_fn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate a user-supplied density function ρ(u,v) over a res×res grid on [0,1]^2.

    density_fn must accept tensors U,V (broadcastable) and return nonnegative values.
    Returns: ρ as (res,res) tensor on `device`, normalized to sum=1 (with epsilon for safety).
    """
    # Grid of normalized coordinates
    xs = torch.linspace(0.0, 1.0, res, device=device)
    ys = torch.linspace(0.0, 1.0, res, device=device)
    U, V = torch.meshgrid(xs, ys, indexing="xy")
    rho = torch.clamp(density_fn(U, V), min=0.0)
    rho_normalized = rho / (rho.sum() + 1e-12)
    return U, V, rho_normalized


@torch.no_grad()
def _eval_density_on_points(pts_uv: torch.Tensor, density_fn) -> torch.Tensor:
    """
    Evaluate ρ(u,v) for a (N,2) tensor of points in [0,1]^2.

    Returns: (N,) nonnegative, normalized weights (sum=1).
    """
    w = torch.clamp(density_fn(pts_uv[:, 0], pts_uv[:, 1]), min=0.0)
    w_normalized = w / (w.sum() + 1e-12)
    return w_normalized


@torch.no_grad()
def _gumbel_topk_sample_without_replacement(weights: torch.Tensor, k: int) -> torch.Tensor:
    """
    Parallel sampling w/out replacement using Gumbel-Top-K.
    We sample top-k indices according to 'weights' (already nonnegative, not all zero).

    Returns: (k,) indices (dtype long).
    """
    # logw + Gumbel noise → top-k
    logw = torch.log(weights + 1e-20)
    g = -torch.log(-torch.log(torch.rand_like(logw) + 1e-20) + 1e-20)
    g_indices = torch.topk(logw + g, k=k, largest=True).indices
    return g_indices    


@torch.no_grad()
def sample_blue_noise_density(
    res: int = 512,
    n_points: int = 3000,
    density_fn = None,
    oversample: float = 2.0,
    plot: bool = True,
    fig_name: str = None,
    idx: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    High-performance, GPU-parallel sampling of n_points distributed according to ρ(u,v),
    while preserving good spacing via stratified jittered grid + Gumbel-Top-K selection.

    Returns
    -------
    coords : torch.IntTensor (n_points, 2) on CPU
        Integer pixel coordinates (x,y) in [0, res-1].
    """
    assert callable(density_fn), "Please provide a density_fn: lambda U,V → rho(U,V)."
    U, V, rho_grid = _eval_density_on_grid(res, density_fn)
    pool_uv = _make_jittered_grid(n_points, oversample)
    w = _eval_density_on_points(pool_uv, density_fn)
    pts_uv = pool_uv[_gumbel_topk_sample_without_replacement(w, n_points)]
    coords = (pts_uv * res).clamp(0, res - 1 - 1e-6).floor().to(torch.int32)
    if plot:
        _plot_density_and_points(
            rho_grid, 
            coords, 
            res, 
            n_points, 
            fig_name
        )
    return (U, V, rho_grid, coords)


# --- Image-based Initial Sampler Class ---
class ImageBasedSampler:
    def __init__(self, images_path, dataset_paths=None, white_threshold: float = 0.95):
        self.images_path = images_path
        self.image_files = [f for f in os.listdir(self.images_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        # self.image_files.sort(key=lambda x: int(x.split('.')[0]))
        self.image_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else str(x.split('.')[0]))
        self.idx = -1  # Start before the first image

        # Used for Dataset Export
        self.dataset_paths = dataset_paths
        self.json_data = []
        self.export_json_every_n = 100  # Export every image
        # Background detection settings (simple white-threshold)
        # Pixels with grayscale value >= white_threshold (in [0,1]) will be treated as empty (zero density)
        self.white_threshold = float(white_threshold)

    def __call__(self, res=512, n_points=3000, density_fn=None, oversample=2.0, plot=True, fig_name=None, idx=-1):
        if idx != -1:
            # Can support multi threading if idx is provided
            curr_idx = idx
        else:
            # Can only be called sequentially
            self.idx += 1
            curr_idx = self.idx

        if curr_idx >= len(self.image_files):
            raise IndexError("No more images left in folder.")
        image_path = os.path.join(self.images_path, self.image_files[curr_idx])

        # Load, resize image and convert [0,255] to [0,1] for direct use
        img = Image.open(image_path).convert("L").resize((res, res), Image.BICUBIC)
        img_arr = torch.from_numpy(np.array(img)).float().to(device)
        img_arr = img_arr / 255.0
        # Simple background detection (white-threshold only)
        threshold = float(self.white_threshold)
        mask = img_arr >= threshold
        mask = mask.to(device)
        
        # Add small constant to avoid zero density regions
        # img_arr = img_arr + 0.01
        
        def density_fn_img(U, V):
            x_idx = (U * (res - 1)).clamp(0, res - 1).round().long()
            y_idx = (V * (res - 1)).clamp(0, res - 1).round().long()
            # img_arr is (height, width) = (y, x), so index as [y, x]
            vals = img_arr[y_idx, x_idx]
            # Apply computed background mask: set background pixels to zero density
            vals = torch.where(mask[y_idx, x_idx], torch.tensor(0.0, device=device), vals)
            return vals  # Return the direct grayscale density (with white threshold applied)

        return sample_blue_noise_density(
            res=res,
            n_points=n_points,
            density_fn=density_fn_img,
            oversample=oversample,
            plot=plot,
            fig_name=fig_name
        )

    def export_sample(self, rho_grid, coords_relaxed, res, idx=-1):
        if idx != -1:
            curr_idx = idx
        else:
            curr_idx = self.idx

        rho_grid_cpu = rho_grid.detach().cpu()
        rho_grid_cpu = (rho_grid_cpu / (rho_grid_cpu.max() + 1e-12)).float()
        rho_grid_cpu = (rho_grid_cpu * 255).to(torch.uint8).numpy()

        # Listen! The coords_relaxed are x and y coordinates of the points
        # We need to create white image with black points, and then resize to res x res
        # coords_relaxed is float tensor, convert to integer pixel coordinates
        coords_relaxed_cpu = coords_relaxed.detach().cpu()
        coords_relaxed_indices = (coords_relaxed_cpu).clamp(0, res - 1 - 1e-6).floor().to(torch.int32)
        stipple_img = torch.ones((res, res), dtype=torch.float32)
        stipple_img[coords_relaxed_indices[:,1], coords_relaxed_indices[:,0]] = 0.0  # Set points to black
        coords_relaxed_cpu = (stipple_img * 255).to(torch.uint8).numpy()

        image_base_name = os.path.basename(self.image_files[curr_idx])
        Image.fromarray(rho_grid_cpu).save(
            os.path.join(self.dataset_paths['source_path'], image_base_name)
        )
        Image.fromarray(coords_relaxed_cpu).save(
            os.path.join(self.dataset_paths['target_path'], image_base_name)
        )
        self.json_data.append({
            "source": f"source/{image_base_name}",
            "target": f"target/{image_base_name}",
            "prompt": f"Stippling"
        })

        if curr_idx % self.export_json_every_n == 0:
            self.export_json()

    def export_json(self):
        with open(self.dataset_paths['json_path'], 'w') as f:
            for item in self.json_data:
                data_line = str(item).replace("\'", "\"")
                f.write(f"{data_line}\n")


@torch.no_grad()
def cc_lloyd_relaxation_wrapper(
    base_sampler,
    density_fn,
    out_res: int = 512,
    n_points: int = 3000,
    iters: int = 5,
    plot: bool = True,
    fig_name: str = None,
    debug: bool = True,
    idx: int = -1
) -> torch.Tensor:
    """
    Wraps an existing blue-noise sampler with capacity-constrained Lloyd relaxation.

    Returns
    -------
    coords_relaxed : torch.Tensor
        (n_points,2) final relaxed coordinates on CPU.
    """
    U, V, rho_grid, coords0 = base_sampler(
        res=out_res, n_points=n_points, density_fn=density_fn, plot=False, idx=idx
    )
    coords = coords0.clone().float().to(device)
    grid = torch.stack([U*out_res, V*out_res], dim=-1)
    for _ in range(iters):
        d2 = torch.cdist(grid.view(-1,2), coords)
        assign = d2.argmin(dim=1)
        rho_flat = rho_grid.view(-1)
        num = torch.zeros((n_points,2), device=device)
        den = torch.zeros((n_points,), device=device)
        num.index_add_(0, assign, grid.view(-1,2) * rho_flat[:,None])
        den.index_add_(0, assign, rho_flat)
        coords = num / (den[:,None] + 1e-12)
    coords_relaxed = coords
    if debug:
        if plot:
            _plot_density_samples_relaxed(
                rho_grid, 
                coords0, 
                coords_relaxed, 
                out_res, 
                n_points, 
                iters, 
                fig_name
            )
    else:
        base_sampler.export_sample(rho_grid, coords_relaxed, out_res, idx)
    return coords_relaxed


@torch.no_grad()
def cc_lloyd_multires_wrapper(
    base_sampler,
    density_fn,
    n_points: int = 3000,
    out_res: int = 512,
    iters_per_level = (2, 1),
    levels = (128, 512),
    plot: bool = True,
    fig_name: str = None,
    debug: bool = True,
    idx: int = -1
) -> torch.Tensor:
    """
    Multi-resolution capacity-constrained Lloyd relaxation (discretized).
    Starts from base_sampler, refines sites on coarse then fine grids.

    Returns
    -------
    coords_relaxed : (n_points,2) tensor on CPU
    """
    U, V, rho_grid, coords0 = base_sampler(
        res=out_res, n_points=n_points, density_fn=density_fn, plot=False, idx=idx
    )
    coords = coords0.clone().float().to(device)
    rho_flat = rho_grid.view(-1)
    for level_res, n_iter in zip(levels, iters_per_level):
        pixels = torch.stack([U*level_res, V*level_res], dim=-1).view(-1,2)
        for _ in range(n_iter):
            d2 = torch.cdist(pixels, coords)
            assign = d2.argmin(dim=1)
            num = torch.zeros((n_points,2), device=device)
            den = torch.zeros((n_points,), device=device)
            num.index_add_(0, assign, pixels * rho_flat[:,None])
            den.index_add_(0, assign, rho_flat)
            coords = num / (den[:,None] + 1e-12)
    coords_relaxed = (coords / levels[-1] * out_res).clamp(0, out_res-1)
    if debug:
        if plot:
            _plot_density_samples_relaxed(
                rho_grid, 
                coords0, 
                coords_relaxed, 
                out_res, 
                n_points, 
                iters_per_level, 
                fig_name
            )
    else:
        base_sampler.export_sample(rho_grid, coords_relaxed, out_res, idx)
    
    return coords_relaxed


# --- Plotting helpers ---
def _plot_density_and_points(rho_grid, coords, res, n_points, fig_name):
    rho_grid_cpu = rho_grid.detach().cpu()
    rho_grid_cpu = (rho_grid_cpu / (rho_grid_cpu.max() + 1e-12)).float()
    coords_cpu = coords.detach().cpu()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rho_grid_cpu.numpy(), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    axes[0].set_title("Density (grayscale)")
    axes[0].axis("off")
    # axes[1].imshow(torch.ones((res, res)), cmap="gray", origin="upper")
    axes[1].scatter(coords_cpu[:,0], coords_cpu[:,1], s=2, c="black", marker=".")
    axes[1].set_title(f"Initial samples ({n_points} points)")
    axes[1].set_xlim(0,res); axes[1].set_ylim(res,0); axes[1].axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_name)


def _plot_density_samples_relaxed(rho_grid, coords0, coords_relaxed, res, n_points, iters, fig_name):
    """
    Plots three panels:
    1. Density (grayscale)
    2. Initial samples
    3. CC Lloyd relaxed
    """
    rho_grid_cpu = rho_grid.detach().cpu()
    rho_grid_cpu = (rho_grid_cpu / (rho_grid_cpu.max() + 1e-12)).float()
    coords0_cpu = coords0.detach().cpu()
    coords_relaxed_cpu = coords_relaxed.detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Density (Grayscale)
    axes[0].imshow(rho_grid_cpu.numpy(), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    axes[0].set_title("Density (grayscale)")
    axes[0].axis("off")
    # Initial samples
    # axes[1].imshow(rho_img.numpy(), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    axes[1].scatter(coords0_cpu[:,0], coords0_cpu[:,1], s=2, c="black", marker=".")
    axes[1].set_title(f"Initial samples ({n_points} points)")
    axes[1].set_xlim(0,res); axes[1].set_ylim(res,0); axes[1].axis("off")
    # CC Lloyd relaxed
    # axes[2].imshow(rho_img.numpy(), cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    axes[2].scatter(coords_relaxed_cpu[:,0], coords_relaxed_cpu[:,1], s=2, c="black", marker=".")
    axes[2].set_title(f"CC Lloyd relaxed ({iters} iters)")
    axes[2].set_xlim(0,res); axes[2].set_ylim(res,0); axes[2].axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_name)


# --- Dataset Generation Function ---
def generate_stippling_dataset(
    N: int,
    base_sampler = None,
    output_path: str = "output",
    debug: bool = True
):
    """
    Generate N stippling images using sample_blue_noise_density, saving each to output_path.
    Each image uses a randomly generated density function (random grayscale map).
    """
    # TODO: Add parallel processing for faster dataset generation

    os.makedirs(output_path, exist_ok=True)
    if N == -1:
        N = len(base_sampler.image_files)

    for idx in tqdm(range(N)):
        input_base_name = os.path.basename(base_sampler.image_files[idx])
        img_path = os.path.join(output_path,input_base_name)
        cc_lloyd_multires_wrapper(
            base_sampler=base_sampler,
            density_fn=rho_linear,
            n_points=4000,
            out_res=512,
            iters_per_level=(9, 1),
            levels=(512, 512),
            plot=True,
            fig_name=img_path,
            debug=debug,
            idx=idx
        )
    if not debug:
        base_sampler.export_json()


# --- Examples ---
def example1():
    fig_name = "StipplingSampler.png"
    _ = sample_blue_noise_density(
        res=512,
        n_points=4000,
        density_fn=rho_linear,
        oversample=2.0,
        plot=True,
        fig_name=fig_name
    )


def example2():
    fig_name = "FastStipplingV1.png"
    coords_relaxed = cc_lloyd_relaxation_wrapper(
        base_sampler=sample_blue_noise_density,
        density_fn=rho_linear,
        out_res=512,
        n_points=4000,
        iters=10,
        plot=True,
        fig_name=fig_name
    )
    print("Final relaxed coords shape:", coords_relaxed.shape)


def example3():
    fig_name = "FastStipplingV2.png"
    coords_relaxed = cc_lloyd_multires_wrapper(
        base_sampler=sample_blue_noise_density,
        density_fn=rho_linear,
        n_points=4000,
        out_res=512,
        iters_per_level=(9, 1),
        levels=(512, 512),
        plot=True,
        fig_name=fig_name
    )
    print("Final relaxed coords shape:", coords_relaxed.shape)


def debug_dataset_generator():
    WHITE_THRESHOLD = 0.95  # Threshold for white pixels

    IMAGES_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "original")
    OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    img_sampler = ImageBasedSampler(
        images_path=IMAGES_PATH,
        white_threshold=WHITE_THRESHOLD
    )
    generate_stippling_dataset(
        N=10,
        base_sampler=img_sampler,
        output_path=OUTPUT_PATH,
        debug=True
    )


def true_dataset_generator():
    WHITE_THRESHOLD = 0.9  # Threshold for white pixels

    SOURCE_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "source")
    TARGET_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "target")
    JSON_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "prompt.json")
    IMAGES_PATH = os.path.join(ROOT_PATH, DATA_FOLDER, "original")
    OUTPUT_PATH = os.path.join(ROOT_PATH, "output")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # N = 10
    N = -1  # Set to -1 to process all images in the folder

    dataset_paths = dict(
        source_path=SOURCE_PATH,
        target_path=TARGET_PATH,
        json_path=JSON_PATH
    )
    os.makedirs(os.path.dirname(dataset_paths['json_path']), exist_ok=True)
    os.makedirs(dataset_paths['source_path'], exist_ok=True)
    os.makedirs(dataset_paths['target_path'], exist_ok=True)

    img_sampler = ImageBasedSampler(
        images_path=IMAGES_PATH,
        dataset_paths=dataset_paths,
        white_threshold=WHITE_THRESHOLD
    )
    generate_stippling_dataset(
        N=N,
        base_sampler=img_sampler,
        output_path=OUTPUT_PATH,
        debug=False
    )


if __name__ == "__main__":
    print("Starting fast stippling generator...")
    sys.stdout.flush()
    DATA_FOLDER = "data"
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

    # Set CUDA parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    seed = 42
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Define density function
    rho_linear = lambda U, V: U
    # rho_vertical_quad = lambda U, V: V**2
    # rho_radial = lambda U, V: (1.0 - torch.sqrt((U-0.5)**2 + (V-0.5)**2) / 0.7071).clamp(min=0.0)

    # example1()
    # example2()
    # example3()

    # Generate stippling dataset with your images
    # debug_dataset_generator()
    true_dataset_generator()
