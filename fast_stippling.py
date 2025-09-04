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


import math
import torch
import matplotlib.pyplot as plt


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
    fig_name: str = None
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


@torch.no_grad()
def cc_lloyd_relaxation_wrapper(
    base_sampler,
    density_fn,
    out_res: int = 512,
    n_points: int = 3000,
    iters: int = 5,
    plot: bool = True,
    fig_name: str = None
) -> torch.Tensor:
    """
    Wraps an existing blue-noise sampler with capacity-constrained Lloyd relaxation.

    Returns
    -------
    coords_relaxed : torch.Tensor
        (n_points,2) final relaxed coordinates on CPU.
    """
    U, V, rho_grid, coords0 = base_sampler(
        res=out_res, n_points=n_points, density_fn=density_fn, plot=False
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
    coords_relaxed = coords.cpu()
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
    fig_name: str = None
) -> torch.Tensor:
    """
    Multi-resolution capacity-constrained Lloyd relaxation (discretized).
    Starts from base_sampler, refines sites on coarse then fine grids.

    Returns
    -------
    coords_relaxed : (n_points,2) tensor on CPU
    """
    U, V, rho_grid, coords0 = base_sampler(
        res=out_res, n_points=n_points, density_fn=density_fn, plot=False
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
    coords_relaxed = (coords / levels[-1] * out_res).clamp(0, out_res-1).cpu()
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
    axes[2].scatter(coords_relaxed[:,0], coords_relaxed[:,1], s=2, c="black", marker=".")
    axes[2].set_title(f"CC Lloyd relaxed ({iters} iters)")
    axes[2].set_xlim(0,res); axes[2].set_ylim(res,0); axes[2].axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_name)


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


if __name__ == "__main__":
    # Set CUDA parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # seed = 12345
    seed = 42
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    # Run the multi-resolution CC Lloyd wrapper
    rho_linear = lambda U, V: U
    # rho_vertical_quad = lambda U, V: V**2
    # rho_radial = lambda U, V: (1.0 - torch.sqrt((U-0.5)**2 + (V-0.5)**2) / 0.7071).clamp(min=0.0)

    # example1()
    # example2()
    example3()
