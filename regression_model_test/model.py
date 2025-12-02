import torch
import torch.nn as nn
from typing import Tuple


class PointDecoder(nn.Module):
    """
    Shared per-point decoder:
    latent (B, latent_dim) -> points (B, num_points, 2)
    """

    def __init__(
        self,
        num_points: int,
        latent_dim: int,
        global_dim: int = 512,
        point_embed_dim: int = 32,
        mlp_hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_points = num_points

        # Map latent to a global feature
        self.global_mlp = nn.Sequential(
            nn.Linear(latent_dim, global_dim),
            nn.ReLU(inplace=True),
        )

        # Learned embedding for each point index (adds a notion of "slot")
        self.point_embedding = nn.Embedding(num_points, point_embed_dim)

        in_dim = global_dim + point_embed_dim

        # Shared per-point MLP
        self.per_point_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 2),
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim)

        Returns:
            points: (B, num_points, 2) in [0,1]
        """
        B = latent.size(0)
        device = latent.device

        # (B, latent_dim) -> (B, global_dim)
        global_feat = self.global_mlp(latent)  # (B, G)

        # (N,) point indices
        idx = torch.arange(self.num_points, device=device)
        # (N, E)
        point_emb = self.point_embedding(idx)

        # Expand to (B, N, G) and (B, N, E)
        global_feat_exp = global_feat.unsqueeze(1).expand(B, self.num_points, -1)
        point_emb_exp = point_emb.unsqueeze(0).expand(B, -1, -1)

        # Concatenate -> (B, N, G+E)
        feats = torch.cat([global_feat_exp, point_emb_exp], dim=-1)

        # Flatten -> (B*N, G+E)
        feats_flat = feats.view(B * self.num_points, -1)

        # Shared MLP -> (B*N, 2)
        out_flat = self.per_point_mlp(feats_flat)
        out_flat = self.output_activation(out_flat)

        # Reshape back -> (B, N, 2)
        points = out_flat.view(B, self.num_points, 2)
        return points


class ImageToPointSet(nn.Module):
    """
    Direct regression model with upgraded decoder:
    Grayscale image -> fixed-size set of 2D points (normalized to [0,1]^2).

    Input:
        x: (B, 1, H, W)

    Output:
        points: (B, num_points, 2)  in [0, 1]
    """

    def __init__(
        self,
        num_points: int = 5000,
        base_channels: int = 32,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.num_points = num_points

        # --- Encoder: CNN that produces a global feature vector ---
        self.encoder = nn.Sequential(
            # (B, 1, H, W) -> (B, base_channels, H/2, W/2)
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # -> (B, 2*base_channels, H/4, W/4)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # -> (B, 4*base_channels, H/8, W/8)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Global average pool -> (B, 4*base_channels, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        enc_out_channels = base_channels * 4

        # --- Project encoder output to latent vector ---
        self.to_latent = nn.Sequential(
            nn.Flatten(),  # (B, C, 1, 1) -> (B, C)
            nn.Linear(enc_out_channels, latent_dim),
            nn.ReLU(inplace=True),
        )

        # --- Upgraded decoder ---
        self.decoder = PointDecoder(
            num_points=num_points,
            latent_dim=latent_dim,
            global_dim=512,
            point_embed_dim=32,
            mlp_hidden_dim=512,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.encoder(x)        # (B, C, 1, 1)
        latent = self.to_latent(feat_map) # (B, latent_dim)
        points = self.decoder(latent)     # (B, num_points, 2)
        return points


def normalize_to_pixels(points: torch.Tensor, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Convert normalized [0,1] coords to pixel coords
    (x in [0,W-1], y in [0,H-1])

    Args:
        points: (B, N, 2) in [0,1]
        img_size: (H, W)

    Returns:
        points_px: (B, N, 2) in pixel coordinates
    """
    H, W = img_size
    points_px = points.clone()
    points_px[..., 0] = points_px[..., 0] * (W - 1)
    points_px[..., 1] = points_px[..., 1] * (H - 1)
    return points_px


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 2
    img_size = (512, 512)
    num_points = 5000

    model = ImageToPointSet(
        num_points=num_points,
        base_channels=32,
        latent_dim=512,
    ).to(device)

    dummy_input = torch.randn(batch_size, 1, img_size[0], img_size[1], device=device)

    with torch.no_grad():
        points_norm = model(dummy_input)
        points_px = normalize_to_pixels(points_norm, img_size)

    print("Input shape:          ", dummy_input.shape)
    print("Output shape (norm):  ", points_norm.shape)
    print("Output shape (pixels):", points_px.shape)

    print("\nExample normalized points (batch 0, first 5):")
    print(points_norm[0, :5])

    print("\nExample pixel-space points (batch 0, first 5):")
    print(points_px[0, :5])

    assert points_norm.shape == (batch_size, num_points, 2)
    assert (points_norm >= 0.0).all() and (points_norm <= 1.0).all()
    print("\nSanity checks passed: correct shape and points in [0,1].")


if __name__ == "__main__":
    main()
