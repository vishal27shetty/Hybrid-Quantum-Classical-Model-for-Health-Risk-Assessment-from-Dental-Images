"""
Data loading and preprocessing module for dental health image classification.
Handles loading images from healthy and deficiency folders.
"""

import os
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DentalHealthDataset(Dataset):
    """Custom dataset for dental health images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        """
        Args:
            image_paths: List of paths to image files
            labels: List of labels (0 = healthy, 1 = deficiency)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label


def load_image_paths(healthy_dir: str, deficiency_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load all image paths from the two directories.
    
    Args:
        healthy_dir: Path to healthy teeth images folder
        deficiency_dir: Path to vitamin c deficiency images folder
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # Load healthy images (label = 0)
    healthy_path = Path(healthy_dir)
    if healthy_path.exists():
        for img_file in healthy_path.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(0)  # Low Risk
        print(f"Loaded {len([l for l in labels if l == 0])} healthy images")
    else:
        print(f"Warning: Healthy images directory not found: {healthy_dir}")
    
    # Load deficiency images (label = 1)
    deficiency_path = Path(deficiency_dir)
    if deficiency_path.exists():
        for img_file in deficiency_path.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(1)  # High Risk
        print(f"Loaded {len([l for l in labels if l == 1])} deficiency images")
    else:
        print(f"Warning: Deficiency images directory not found: {deficiency_dir}")
    
    return image_paths, labels


def get_data_transforms(img_size: int = 224):
    """
    Get standard data transforms for training and validation.
    
    Args:
        img_size: Target image size (default 224 for most CNNs)
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def create_dataloaders(
    healthy_dir: str,
    deficiency_dir: str,
    batch_size: int = 16,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        healthy_dir: Path to healthy teeth images
        deficiency_dir: Path to vitamin c deficiency images
        batch_size: Batch size for dataloaders
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load all image paths and labels
    image_paths, labels = load_image_paths(healthy_dir, deficiency_dir)
    
    if len(image_paths) == 0:
        raise ValueError("No images found in the specified directories!")
    
    print(f"\nTotal images loaded: {len(image_paths)}")
    print(f"Class distribution: Healthy={labels.count(0)}, Deficiency={labels.count(1)}")
    
    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Get transforms
    transforms_dict = get_data_transforms()
    
    # Create datasets
    train_dataset = DentalHealthDataset(train_paths, train_labels, transforms_dict['train'])
    val_dataset = DentalHealthDataset(val_paths, val_labels, transforms_dict['val'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader


def preprocess_single_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size
        
    Returns:
        Preprocessed image tensor with batch dimension
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

