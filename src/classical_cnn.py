"""
Classical CNN Feature Extractor for dental health images.
Uses a pretrained ResNet18 as the backbone to extract meaningful visual features.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CNNFeatureExtractor(nn.Module):
    """
    Classical CNN that extracts a low-dimensional feature vector from images.
    Uses a pretrained ResNet18 backbone with a custom feature reduction layer.
    """
    
    def __init__(self, feature_dim: int = 8, pretrained: bool = True):
        """
        Args:
            feature_dim: Dimension of output feature vector (should be small for quantum circuit)
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet18 outputs 512 features before the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom layers to reduce to desired feature dimension
        self.feature_reducer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, feature_dim),
            nn.Tanh()  # Output in range [-1, 1] for better quantum encoding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature vector of shape (batch_size, feature_dim)
        """
        # Extract features using ResNet backbone
        features = self.backbone(x)
        
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        # Reduce to target feature dimension
        reduced_features = self.feature_reducer(features)
        
        return reduced_features
    
    def freeze_backbone(self):
        """Freeze the ResNet backbone to only train the feature reducer."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze the ResNet backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class HybridCNNClassifier(nn.Module):
    """
    A hybrid classifier that combines CNN feature extraction with a classical head.
    This can be used for initial training before switching to quantum classification.
    """
    
    def __init__(self, feature_dim: int = 8, num_classes: int = 2):
        """
        Args:
            feature_dim: Dimension of feature vector from CNN
            num_classes: Number of output classes (default 2 for binary classification)
        """
        super(HybridCNNClassifier, self).__init__()
        
        self.feature_extractor = CNNFeatureExtractor(feature_dim=feature_dim)
        
        # Classical classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 4),
            nn.ReLU(),
            nn.Linear(4, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid classical model.
        
        Args:
            x: Input image tensor
            
        Returns:
            Class logits
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract just the features without classification."""
        return self.feature_extractor(x)


def create_cnn_feature_extractor(
    feature_dim: int = 8,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> CNNFeatureExtractor:
    """
    Factory function to create a CNN feature extractor.
    
    Args:
        feature_dim: Dimension of output features
        pretrained: Whether to use pretrained weights
        device: Device to place the model on
        
    Returns:
        Configured CNNFeatureExtractor model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNFeatureExtractor(feature_dim=feature_dim, pretrained=pretrained)
    model = model.to(device)
    
    print(f"Created CNN Feature Extractor with {feature_dim} output dimensions")
    print(f"Model placed on device: {device}")
    
    return model


if __name__ == "__main__":
    # Test the feature extractor
    print("Testing CNN Feature Extractor...")
    
    model = create_cnn_feature_extractor(feature_dim=8)
    
    # Create a dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("\nTest passed!")

