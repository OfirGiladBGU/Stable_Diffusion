"""
Paper-faithful Weighted Voronoi Stippling (Secord, 2002) - Hardcoded Args
"""

import math
import sys
from typing import List, Tuple

import cv2
import numpy as np
from scipy.spatial import Voronoi

# ---------------- Configuration ----------------
IMAGE_PATH = fr'gloria_pickle.jpg'  # set your input image path
NUM_POINTS = 6000
MAX_ITERS = 50
TOLERANCE = 0.5
DOT_RADIUS = 1
SEED = 1
MAX_SIDE = 0  # resize so max(width,height)=MAX_SIDE (0 = keep)
DAMPING = 1.0  # fraction of move toward centroid per iteration
# ------------------------------------------------

def sutherland_hodgman_clip(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    """Clip a polygon to the rectangle [0,W) x [0,H) (float coords).
    poly: (M,2) float array. Returns (K,2) float array (may be empty).
    """
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=np.float32)

    def clip_edge(vertices: List[Tuple[float, float]], edge: str):
        out: List[Tuple[float, float]] = []
        if not vertices:
            return out
        def inside(p):
            x, y = p
            if edge == 'left':
                return x >= 0.0
            if edge == 'right':
                return x < float(W)
            if edge == 'top':
                return y >= 0.0
            return y < float(H)  # bottom
        def intersect(p1, p2):
            x1, y1 = p1; x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            if edge == 'left':
                x = 0.0
                if dx == 0: return (x, y1)
                t = (x - x1) / dx
                y = y1 + t * dy
            elif edge == 'right':
                x = float(W) - 1e-6
                if dx == 0: return (x, y1)
                t = (x - x1) / dx
                y = y1 + t * dy
            elif edge == 'top':
                y = 0.0
                if dy == 0: return (x1, y)
                t = (y - y1) / dy
                x = x1 + t * dx
            else:  # bottom
                y = float(H) - 1e-6
                if dy == 0: return (x1, y)
                t = (y - y1) / dy
                x = x1 + t * dx
            return (x, y)
        prev = vertices[-1]
        prev_in = inside(prev)
        for cur in vertices:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    verts = [(float(x), float(y)) for x, y in poly]
    for e in ('left', 'right', 'top', 'bottom'):
        verts = clip_edge(verts, e)
        if not verts:
            break
    return np.asarray(verts, dtype=np.float32)


def polygon_scanlines(poly: np.ndarray) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Yield scanline segments for an arbitrary polygon.
    Returns a list of (y, [(xL,xR), ...]) with integer pixel spans inclusive.
    """
    if len(poly) < 3:
        return []
    y_min = int(np.ceil(np.min(poly[:, 1])))
    y_max = int(np.floor(np.max(poly[:, 1])))
    if y_max < y_min:
        return []
    segs_per_row = []
    n = len(poly)
    for y in range(y_min, y_max + 1):
        xs = []
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # consider edges that cross the horizontal line y
            if (y1 <= y < y2) or (y2 <= y < y1):
                if y2 == y1:  # should not happen due to condition
                    continue
                t = (y - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))
        xs.sort()
        spans = []
        for i in range(0, len(xs), 2):
            if i + 1 >= len(xs):
                break
            L = int(np.ceil(xs[i]))
            R = int(np.floor(xs[i + 1]))
            if R >= L:
                spans.append((L, R))
        if spans:
            segs_per_row.append((y, spans))
    return segs_per_row


# --------------------------- Centroid integration ---------------------------

def build_prefix(density: np.ndarray):
    """Build per-row prefix sums P and Qx (ρ and x·ρ)."""
    H, W = density.shape
    x = np.arange(W, dtype=np.float32)
    P = np.cumsum(density, axis=1, dtype=np.float64)
    Qx = np.cumsum(density * x[None, :], axis=1, dtype=np.float64)
    return P, Qx


def segment_integrals(P: np.ndarray, Qx: np.ndarray, y: int, xL: int, xR: int):
    """Compute ∫ ρ and ∫ x·ρ over [xL,xR] at row y using prefixes."""
    if xL == 0:
        mass = P[y, xR]
        mx = Qx[y, xR]
    else:
        mass = P[y, xR] - P[y, xL - 1]
        mx = Qx[y, xR] - Qx[y, xL - 1]
    return mass, mx


def weighted_centroid_of_cell(poly: np.ndarray, density: np.ndarray, P: np.ndarray, Qx: np.ndarray) -> Tuple[float, float, float]:
    """Compute (cx, cy, mass) of a polygonal cell under scalar density."""
    H, W = density.shape
    poly_c = sutherland_hodgman_clip(poly, W, H)
    if len(poly_c) < 3:
        # fallback to polygon centroid (unweighted) if degenerate
        return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1])), 0.0

    mass_total = 0.0
    Mx_total = 0.0
    My_total = 0.0

    for y, spans in polygon_scanlines(poly_c):
        for xL, xR in spans:
            xL = max(0, min(xL, W - 1))
            xR = max(0, min(xR, W - 1))
            if xR < xL:
                continue
            mass, mx = segment_integrals(P, Qx, y, xL, xR)
            mass_total += mass
            Mx_total += mx
            My_total += y * mass

    if mass_total <= 1e-12:
        # return unweighted centroid to avoid NaNs; zero mass signals caller
        return float(np.mean(poly_c[:, 0])), float(np.mean(poly_c[:, 1])), 0.0

    cx = Mx_total / mass_total
    cy = My_total / mass_total
    return float(cx), float(cy), float(mass_total)


# --------------------------- Stippling core ---------------------------

def init_points_by_density(N: int, density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H, W = density.shape
    pts = []
    while len(pts) < N:
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        if rng.random() < float(density[y, x]):
            pts.append([float(x), float(y)])
    return np.asarray(pts, dtype=np.float32)


def mirror_points(points: np.ndarray, W: int, H: int) -> np.ndarray:
    """8-neighbor mirroring for bounded Voronoi."""
    P = points
    left = np.stack([-P[:, 0], P[:, 1]], axis=1)
    right = np.stack([2 * W - P[:, 0], P[:, 1]], axis=1)
    top = np.stack([P[:, 0], -P[:, 1]], axis=1)
    bottom = np.stack([P[:, 0], 2 * H - P[:, 1]], axis=1)
    tl = np.stack([-P[:, 0], -P[:, 1]], axis=1)
    tr = np.stack([2 * W - P[:, 0], -P[:, 1]], axis=1)
    bl = np.stack([-P[:, 0], 2 * H - P[:, 1]], axis=1)
    br = np.stack([2 * W - P[:, 0], 2 * H - P[:, 1]], axis=1)
    return np.vstack([P, left, right, top, bottom, tl, tr, bl, br]).astype(np.float32)


def iterate_lloyd(points: np.ndarray, density: np.ndarray, max_iters: int, tol: float, damping: float = 1.0) -> np.ndarray:
    H, W = density.shape
    P, Qx = build_prefix(density.astype(np.float32))
    pts = points.copy().astype(np.float32)
    for it in range(max_iters):
        all_pts = mirror_points(pts, W, H)
        vor = Voronoi(all_pts)
        new_pts = np.empty_like(pts)
        max_shift = 0.0
        for i in range(len(pts)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                new_pts[i] = pts[i]
                continue
            polygon = vor.vertices[region]
            cx, cy, mass = weighted_centroid_of_cell(polygon, density, P, Qx)
            if mass <= 1e-12:
                new_pts[i] = pts[i]
            else:
                target = np.array([cx, cy], dtype=np.float32)
                moved = pts[i] + (target - pts[i]) * float(damping)
                new_pts[i] = moved
                max_shift = max(max_shift, float(np.linalg.norm(moved - pts[i])))

        pts = new_pts
        # Early exit if small movement
        if max_shift < tol:
            # print(f"Converged at iter {it+1}, max shift {max_shift:.3f}")
            break

    return pts


def render(points: np.ndarray, W: int, H: int, radius: int) -> np.ndarray:
    img = np.full((H, W), 255, np.uint8)
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(img, (xi, yi), radius, 0, -1, lineType=cv2.LINE_AA)
    return img

if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Could not read image: {IMAGE_PATH}")
        sys.exit(1)

    H0, W0 = gray.shape
    if MAX_SIDE and max(W0, H0) > MAX_SIDE:
        scale = MAX_SIDE / float(max(W0, H0))
        gray = cv2.resize(gray, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
    density = 1.0 - (gray.astype(np.float32) / 255.0)
    H, W = density.shape
    pts0 = init_points_by_density(NUM_POINTS, density, rng)
    pts = iterate_lloyd(pts0, density, MAX_ITERS, TOLERANCE, DAMPING)
    out = render(pts, W, H, DOT_RADIUS)
    cv2.imwrite('voronoi_cc181_result.png', out)
    print(f'Saved voronoi_cc181_result.png  size=({W}x{H})  points={len(pts)}')

# IMAGE_PATH = fr'D:\AllProjects\PycharmProjects\Stable_Diffusion\data_gradients_x\original\gen_gray_Combined_Shape_1610272128_13.png'  # Replace with your image path