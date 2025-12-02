import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
from datetime import datetime
import json
from pathlib import Path

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from regression_model_test.model import ImageToPointSet
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


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    # === Editable configuration ===
    # Data arguments
    source_dir = r"/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/source"
    target_dir = r"/groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3/target"
    output_root = Path(r"./regression_outputs")
    
    run_name = 'pointset_regression'
    img_size = 512
    black_threshold = 128
    val_split = 0.1
    num_workers = 4
    # Model arguments
    num_points = 5000
    base_channels = 32
    latent_dim = 512
    # Training arguments
    epochs = 5
    batch_size = 8
    lr = 1e-4
    weight_decay = 1e-5
    seed = 42
    # Scheduler arguments
    scheduler_type = 'cosine'  # 'step' | 'cosine' | 'plateau' | 'none'
    scheduler_step = 30
    scheduler_gamma = 0.1
    # Checkpoint arguments
    save_freq = 5
    resume = None  # path to checkpoint dir or None
    # Logging arguments
    no_tensorboard = True
    # Loss arguments
    loss_mode = 'chamfer_l2'  # 'chamfer'|'chamfer_l2'|'chamfer_repel'|'chamfer_l2_repel'
    repel_weight = 0.0
    repel_min_dist = 0.01

    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{run_name}_{timestamp}" if run_name else f"run_{timestamp}"
    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'source_dir': source_dir,
        'target_dir': target_dir,
        'output_root': str(output_root),
        'run_name': run_name,
        'img_size': img_size,
        'black_threshold': black_threshold,
        'val_split': val_split,
        'num_workers': num_workers,
        'num_points': num_points,
        'base_channels': base_channels,
        'latent_dim': latent_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'seed': seed,
        'scheduler': scheduler_type,
        'scheduler_step': scheduler_step,
        'scheduler_gamma': scheduler_gamma,
        'save_freq': save_freq,
        'resume': resume,
        'no_tensorboard': no_tensorboard,
        'loss_mode': loss_mode,
        'repel_weight': repel_weight,
        'repel_min_dist': repel_min_dist,
    }
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Initialize tensorboard writer (if enabled and available)
    writer = None
    if not no_tensorboard:
        if TENSORBOARD_AVAILABLE:
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled: {log_dir}")
        else:
            print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
    else:
        print("TensorBoard logging disabled")
    
    # Load dataset
    print(f"\nLoading dataset from:")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    
    dataset = PointSetDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        img_size=(img_size, img_size),
        normalize_images=True,
        black_threshold=black_threshold
    )
    
    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Set global num_points for collate function
    global _NUM_POINTS
    _NUM_POINTS = num_points
    
    # Create data loaders (use collate_wrapper to avoid lambda pickling issues on Windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper
    )
    
    # Initialize model
    print(f"\nInitializing model:")
    print(f"  Number of points: {num_points}")
    print(f"  Base channels: {base_channels}")
    print(f"  Latent dim: {latent_dim}")
    
    model = ImageToPointSet(
        num_points=num_points,
        base_channels=base_channels,
        latent_dim=latent_dim
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Load checkpoint if resume
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    
    if resume:
        resume_path = os.path.join(resume, 'latest_model.pth')
        start_epoch, _ = load_checkpoint(model, optimizer, resume_path, device)
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("="*80)
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, global_step,
            loss_mode=loss_mode, repel_weight=repel_weight, repel_min_dist=repel_min_dist
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, epoch,
            loss_mode=loss_mode, repel_weight=repel_weight, repel_min_dist=repel_min_dist
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
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  🎉 New best validation loss: {best_val_loss:.6f}")
        
        if (epoch + 1) % save_freq == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, is_best=is_best)
        
        print("="*80)
    
    # Final save
    save_checkpoint(model, optimizer, epochs - 1, val_loss, checkpoint_dir, is_best=False)
    
    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    if writer is not None:
        print(f"Tensorboard logs: {log_dir}")
        print(f"\nTo view tensorboard logs, run:")
        print(f"  tensorboard --logdir {log_dir}")
        writer.close()


if __name__ == "__main__":
    main()
