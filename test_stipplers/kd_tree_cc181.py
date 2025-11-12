"""
Weighted Voronoi Stippling (Python port of Coding Train / d3-delaunay approach)

- Input: grayscale image path (set IMAGE_PATH below)
- Output: stippled PNG (stippling_result.png)

Algorithm:
  * Initialize many points, rejecting bright pixels with probability ~ brightness
  * Iterate:
      - Assign each pixel to its nearest point (KDTree)
      - Compute density-weighted centroids (weight = 1 - gray/255)
      - Lerp points toward their centroids
  * Draw points on white canvas

This follows the spirit of the linked JS sketch which uses `d3.Delaunay.find` to
map pixels to Voronoi cells; here we use a KDTree nearest-neighbor search which
is equivalent for assigning pixels to the closest site.

Dependencies: numpy, opencv-python, scipy
"""

import math
import numpy as np
import cv2
from scipy.spatial import cKDTree

# --------------------------- Config ---------------------------
IMAGE_PATH = fr'gloria_pickle.jpg'  # set your input image path
NUM_POINTS = 6000           # similar to JS example
ITERS = 50                  # relaxation iterations
LERP = 0.1                  # interpolation factor toward centroid
DOT_RADIUS = 1              # radius in pixels when rendering
# MAX_SIZE = 800              # optional: max width/height cap for speed
BATCH_PIXELS = 120_000      # process this many pixels per KDTree batch
RANDOM_SEED = 1             # for reproducibility
# --------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Couldn't read image: {path}")
    # Optionally cap size for speed (preserve aspect)
    # h, w = img.shape
    # scale = min(1.0, MAX_SIZE / max(h, w))
    # if scale < 1.0:
    #     img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def generate_random_points(n, w, h, weights):
    """Reject bright areas. `weights` is 1 - gray/255 over the whole image.
    We sample uniformly over pixels then accept with prob ~ weight.
    """
    pts = []
    # Precompute a flattened alias for speed
    for _ in range(n):
        while True:
            x = rng.integers(0, w)
            y = rng.integers(0, h)
            if rng.random() < weights[y, x]:
                pts.append([float(x), float(y)])
                break
    return np.asarray(pts, dtype=np.float32)


def build_pixel_grid(w, h):
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.column_stack((xx.ravel(), yy.ravel()))  # (Npix, 2) as (x,y)
    return coords


def relax(points, weights, coords, w, h, iters=20, alpha=0.1, batch=BATCH_PIXELS):
    N = len(points)
    pts = points.copy().astype(np.float32)

    # Precompute per-pixel weight array (flattened)
    Wpix = weights.ravel().astype(np.float32)

    for it in range(iters):
        tree = cKDTree(pts)
        sum_w = np.zeros(N, dtype=np.float64)
        sum_xw = np.zeros(N, dtype=np.float64)
        sum_yw = np.zeros(N, dtype=np.float64)

        # Process pixels in batches to keep memory modest
        total_pix = len(coords)
        for start in range(0, total_pix, batch):
            end = min(start + batch, total_pix)
            chunk = coords[start:end]
            w_chunk = Wpix[start:end]
            # nearest site index for each pixel in chunk
            _, idx = tree.query(chunk, k=1, workers=-1)
            # accumulate with bincount
            np.add.at(sum_w, idx, w_chunk)
            np.add.at(sum_xw, idx, chunk[:, 0] * w_chunk)
            np.add.at(sum_yw, idx, chunk[:, 1] * w_chunk)

        # Compute centroids
        centroids = pts.copy()
        mask = sum_w > 1e-12
        centroids[mask, 0] = sum_xw[mask] / sum_w[mask]
        centroids[mask, 1] = sum_yw[mask] / sum_w[mask]
        # If a site gets zero weight, keep it where it is

        # Lerp toward centroids
        pts += (centroids - pts) * float(alpha)

    return pts


def render_points(points, w, h, radius=1):
    img = np.full((h, w), 255, np.uint8)
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img, (xi, yi), radius, 0, -1, lineType=cv2.LINE_AA)
    return img


if __name__ == "__main__":
    gray = load_gray(IMAGE_PATH)
    h, w = gray.shape
    # weight = darkness
    weights = 1.0 - (gray.astype(np.float32) / 255.0)

    # init points similar to JS sketch
    points0 = generate_random_points(NUM_POINTS, w, h, weights)

    # build pixel grid once
    coords = build_pixel_grid(w, h)

    # relax
    points = relax(points0, weights, coords, w, h, iters=ITERS, alpha=LERP, batch=BATCH_PIXELS)

    # render
    out = render_points(points, w, h, radius=DOT_RADIUS)
    cv2.imwrite("kd_tree_cc181_result.png", out)
    print("Saved kd_tree_cc181_result.png", out.shape, f"points={len(points)}")

# IMAGE_PATH = fr'D:\AllProjects\PycharmProjects\Stable_Diffusion\data_gradients_x\original\gen_gray_Combined_Shape_1610272128_13.png'  # Replace with your image path
