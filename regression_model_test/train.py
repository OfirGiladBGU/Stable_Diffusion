import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
from datetime import datetime
import json

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from regression_model import ImageToPointSet
from dataset import PointSetDataset, collate_fn_fixed_size


# Global variable to store num_points for collate function
_NUM_POINTS = 5000


def collate_wrapper(batch):
    """Wrapper for collate_fn_fixed_size to avoid lambda pickling issues"""
    return collate_fn_fixed_size(batch, num_points=_NUM_POINTS)


def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between predicted and ground truth point sets.
    
    Chamfer Distance = mean(min_dist(pred->gt)) + mean(min_dist(gt->pred))
    
    Args:
        pred_points: (B, N, 2) predicted points
        gt_points: (B, M, 2) ground truth points
        
    Returns:
        loss: scalar Chamfer distance
    """
    # Compute pairwise squared distances: (B, N, M)
    # ||pred_i - gt_j||^2
    pred_expanded = pred_points.unsqueeze(2)  # (B, N, 1, 2)
    gt_expanded = gt_points.unsqueeze(1)      # (B, 1, M, 2)
    distances = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # (B, N, M)
    
    # Forward: For each predicted point, find nearest GT point
    min_dist_to_gt, _ = distances.min(dim=2)  # (B, N)
    forward_loss = min_dist_to_gt.mean()
    
    # Backward: For each GT point, find nearest predicted point
    min_dist_to_pred, _ = distances.min(dim=1)  # (B, M)
    backward_loss = min_dist_to_pred.mean()
    
    # Total Chamfer distance
    chamfer_loss = forward_loss + backward_loss
    
    return chamfer_loss


def masked_l2_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, 
                   gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 loss between predicted and ground truth points where mask is valid.
    
    Args:
        pred_points: (B, N, 2) predicted points
        gt_points: (B, N, 2) ground truth points
        gt_mask: (B, N) boolean mask for valid ground truth points
        
    Returns:
        loss: scalar L2 loss
    """
    # Compute squared differences
    squared_diff = torch.sum((pred_points - gt_points) ** 2, dim=-1)  # (B, N)
    
    # Apply mask and compute mean over valid points
    masked_diff = squared_diff * gt_mask.float()
    num_valid = gt_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
    
    # Average over valid points per batch, then over batch
    loss_per_sample = masked_diff.sum(dim=1) / (num_valid.squeeze() + 1e-8)
    loss = loss_per_sample.mean()
    
    return loss


def repulsion_loss(pred_points: torch.Tensor, min_dist: float) -> torch.Tensor:
    """
    Simple repulsion term to discourage point collapse.
    Penalizes pairs of predicted points that are closer than `min_dist`.

    Args:
        pred_points: (B, N, 2)
        min_dist: minimum desired distance in normalized coords [0,1]

    Returns:
        loss: scalar repulsion penalty
    """
    # Compute pairwise distances between predicted points: (B, N, N)
    pp_i = pred_points.unsqueeze(2)  # (B, N, 1, 2)
    pp_j = pred_points.unsqueeze(1)  # (B, 1, N, 2)
    d2 = torch.sum((pp_i - pp_j) ** 2, dim=-1)  # squared distances
    # Avoid self-distances by masking diagonal
    B, N = pred_points.size(0), pred_points.size(1)
    eye = torch.eye(N, device=pred_points.device).unsqueeze(0).expand(B, -1, -1)
    d2 = d2 + eye * 1e9
    # Margin hinge: penalize distances below min_dist
    margin = min_dist ** 2
    penalty = torch.clamp(margin - d2, min=0.0)
    # Average penalty per sample
    loss = penalty.mean()
    return loss


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device, epoch: int, writer, global_step: int,
                loss_mode: str, repel_weight: float, repel_min_dist: float) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, gt_points, gt_masks) in enumerate(progress_bar):
        images = images.to(device)
        gt_points = gt_points.to(device)
        gt_masks = gt_masks.to(device)
        
        # Forward pass
        pred_points = model(images)  # (B, num_points, 2)
        
        # Compute losses with selectable modes
        chamfer_loss = chamfer_distance(pred_points, gt_points)
        l2_loss = masked_l2_loss(pred_points, gt_points, gt_masks)
        repel_loss = repulsion_loss(pred_points, repel_min_dist) if repel_weight > 0 else torch.tensor(0.0, device=device)

        if loss_mode == 'chamfer':
            loss = chamfer_loss
        elif loss_mode == 'chamfer_l2':
            loss = chamfer_loss + l2_loss
        elif loss_mode == 'chamfer_repel':
            loss = chamfer_loss + repel_weight * repel_loss
        elif loss_mode == 'chamfer_l2_repel':
            loss = chamfer_loss + l2_loss + repel_weight * repel_loss
        else:
            loss = chamfer_loss + l2_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.6f}', 'avg_loss': f'{avg_loss:.6f}'})
        
        # Log to tensorboard (if available)
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/batch_loss', loss.item(), global_step)
            global_step += 1
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, epoch: int,
             loss_mode: str, repel_weight: float, repel_min_dist: float) -> float:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for images, gt_points, gt_masks in progress_bar:
            images = images.to(device)
            gt_points = gt_points.to(device)
            gt_masks = gt_masks.to(device)
            
            # Forward pass
            pred_points = model(images)
            
            # Compute losses with selectable modes
            chamfer_loss = chamfer_distance(pred_points, gt_points)
            l2_loss = masked_l2_loss(pred_points, gt_points, gt_masks)
            repel_loss = repulsion_loss(pred_points, repel_min_dist) if repel_weight > 0 else torch.tensor(0.0, device=device)

            if loss_mode == 'chamfer':
                loss = chamfer_loss
            elif loss_mode == 'chamfer_l2':
                loss = chamfer_loss + l2_loss
            elif loss_mode == 'chamfer_repel':
                loss = chamfer_loss + repel_weight * repel_loss
            elif loss_mode == 'chamfer_l2_repel':
                loss = chamfer_loss + l2_loss + repel_weight * repel_loss
            else:
                loss = chamfer_loss + l2_loss
            
            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'val_loss': f'{avg_loss:.6f}'})
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   loss: float, checkpoint_dir: str, is_best: bool = False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Always update latest
    latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: str, device: torch.device):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    return epoch, loss


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.run_name}_{timestamp}" if args.run_name else f"run_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Initialize tensorboard writer (if enabled and available)
    writer = None
    if not args.no_tensorboard:
        if TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled: {log_dir}")
        else:
            print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
    else:
        print("TensorBoard logging disabled")
    
    # Load dataset
    print(f"\nLoading dataset from:")
    print(f"  Source: {args.source_dir}")
    print(f"  Target: {args.target_dir}")
    
    dataset = PointSetDataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        img_size=(args.img_size, args.img_size),
        normalize_images=True,
        black_threshold=args.black_threshold
    )
    
    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Set global num_points for collate function
    global _NUM_POINTS
    _NUM_POINTS = args.num_points
    
    # Create data loaders (use collate_wrapper to avoid lambda pickling issues on Windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper
    )
    
    # Initialize model
    print(f"\nInitializing model:")
    print(f"  Number of points: {args.num_points}")
    print(f"  Base channels: {args.base_channels}")
    print(f"  Latent dim: {args.latent_dim}")
    
    model = ImageToPointSet(
        num_points=args.num_points,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Load checkpoint if resume
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    
    if args.resume:
        resume_path = os.path.join(args.resume, 'latest_model.pth')
        start_epoch, _ = load_checkpoint(model, optimizer, resume_path, device)
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, global_step,
            loss_mode=args.loss_mode, repel_weight=args.repel_weight, repel_min_dist=args.repel_min_dist
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, epoch,
            loss_mode=args.loss_mode, repel_weight=args.repel_weight, repel_min_dist=args.repel_min_dist
        )
        
        # Log to tensorboard (if available)
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6e}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🎉 New best validation loss: {best_val_loss:.6f}")
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, is_best=is_best)
        
        print("="*80)
    
    # Final save
    save_checkpoint(model, optimizer, args.epochs - 1, val_loss, checkpoint_dir, is_best=False)
    
    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    if writer is not None:
        print(f"Tensorboard logs: {log_dir}")
        print(f"\nTo view tensorboard logs, run:")
        print(f"  tensorboard --logdir {log_dir}")
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ImageToPointSet model")
    
    # Data arguments
    parser.add_argument('--source_dir', type=str, 
                       default=r".\data_grads_v3\source",
                       help='Directory containing source images')
    parser.add_argument('--target_dir', type=str,
                       default=r".\data_grads_v3\target",
                       help='Directory containing target images')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--run_name', type=str, default='pointset_regression',
                       help='Name for this training run')
    
    # Dataset arguments
    parser.add_argument('--img_size', type=int, default=512,
                       help='Image size (assumes square images)')
    parser.add_argument('--black_threshold', type=int, default=128,
                       help='Threshold for detecting black pixels (points) in target images')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--num_points', type=int, default=5000,
                       help='Number of points to predict')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base number of channels in encoder')
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--scheduler_step', type=int, default=30,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler')
    
    # Checkpoint arguments
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint directory to resume training')
    
    # Logging arguments
    parser.add_argument('--no_tensorboard', action='store_true', default=True,
                       help='Disable TensorBoard logging')

    # Loss arguments
    parser.add_argument('--loss_mode', type=str, default='chamfer_l2',
                       choices=['chamfer', 'chamfer_l2', 'chamfer_repel', 'chamfer_l2_repel'],
                       help='Select loss composition')
    parser.add_argument('--repel_weight', type=float, default=0.0,
                       help='Weight of repulsion term (0 disables)')
    parser.add_argument('--repel_min_dist', type=float, default=0.01,
                       help='Minimum desired distance between points in normalized coords [0,1]')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)
