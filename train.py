"""
Training script for the Hybrid Quantum-Classical Dental Health Classifier.
"""

import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data_loader import create_dataloaders
from src.hybrid_model import create_hybrid_model, ModelCheckpoint


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The hybrid quantum-classical model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move data to device (CPU for quantum)
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100 * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    return epoch_loss, epoch_accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Validate the model.
    
    Args:
        model: The hybrid quantum-classical model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (val_loss, val_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validating")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (len(pbar))
            accuracy = 100 * correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
    
    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    return val_loss, val_accuracy


def plot_training_history(history: dict, save_path: str):
    """
    Plot and save training history.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")


def main(args):
    """Main training function."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    
    # Device configuration (must be CPU for quantum circuits)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    print("\n" + "="*60)
    print("DENTAL HEALTH QUANTUM ML TRAINING")
    print("="*60)
    print(f"Healthy images folder: {args.healthy_dir}")
    print(f"Deficiency images folder: {args.deficiency_dir}")
    print(f"Feature dimension (qubits): {args.feature_dim}")
    print(f"Quantum layers: {args.num_layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("="*60 + "\n")
    
    # Create data loaders
    print("Loading and preparing dataset...")
    train_loader, val_loader = create_dataloaders(
        healthy_dir=args.healthy_dir,
        deficiency_dir=args.deficiency_dir,
        batch_size=args.batch_size,
        test_size=args.test_split
    )
    
    # Create model
    print("\nCreating hybrid quantum-classical model...")
    model = create_hybrid_model(
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        pretrained_cnn=args.pretrained,
        device=device
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print("\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                args.output_dir, 'checkpoints', 'best_model.pth'
            )
            ModelCheckpoint.save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, val_acc, checkpoint_path
            )
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save latest checkpoint
        latest_path = os.path.join(
            args.output_dir, 'checkpoints', 'latest_model.pth'
        )
        ModelCheckpoint.save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss, val_acc, latest_path
        )
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60 + "\n")
    
    # Plot and save training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Hybrid Quantum-Classical Dental Health Classifier"
    )
    
    # Data arguments
    parser.add_argument(
        '--healthy_dir',
        type=str,
        default='healthy images',
        help='Path to healthy teeth images folder'
    )
    parser.add_argument(
        '--deficiency_dir',
        type=str,
        default='vitamin c deficiency',
        help='Path to vitamin C deficiency images folder'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='Fraction of data for validation (default: 0.2)'
    )
    
    # Model arguments
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=8,
        help='Feature dimension / number of qubits (default: 8)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of quantum variational layers (default: 2)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained CNN backbone'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for training (default: 4, small for quantum)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Directory to save outputs (default: output)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.healthy_dir):
        raise ValueError(f"Healthy images directory not found: {args.healthy_dir}")
    if not os.path.exists(args.deficiency_dir):
        raise ValueError(f"Deficiency images directory not found: {args.deficiency_dir}")
    
    main(args)

