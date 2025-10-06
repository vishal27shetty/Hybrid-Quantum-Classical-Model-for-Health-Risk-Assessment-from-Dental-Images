"""
Hybrid Quantum-Classical Model for Dental Health Risk Classification.
Integrates CNN feature extraction with Variational Quantum Classification.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.classical_cnn import CNNFeatureExtractor
from src.quantum_classifier import VariationalQuantumClassifier


class DentalHealthQMLModel(nn.Module):
    """
    Complete hybrid quantum-classical model for dental health risk assessment.
    
    Architecture:
    1. Classical CNN extracts features from dental images
    2. Quantum VQC processes features for classification
    3. Output layer maps quantum expectation to risk categories
    """
    
    def __init__(
        self,
        feature_dim: int = 8,
        num_layers: int = 2,
        pretrained_cnn: bool = True
    ):
        """
        Args:
            feature_dim: Dimension of feature vector (number of qubits)
            num_layers: Number of variational layers in quantum circuit
            pretrained_cnn: Whether to use pretrained CNN backbone
        """
        super(DentalHealthQMLModel, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Classical feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            feature_dim=feature_dim,
            pretrained=pretrained_cnn
        )
        
        # Quantum classifier
        self.quantum_classifier = VariationalQuantumClassifier(
            num_qubits=feature_dim,
            num_layers=num_layers,
            use_parity=False
        )
        
        # Post-quantum processing layer
        # Maps quantum expectation value [-1, 1] to binary class logits
        self.output_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        print(f"\n{'='*60}")
        print("Hybrid Quantum-Classical Model Initialized")
        print(f"{'='*60}")
        print(f"Feature Dimension: {feature_dim}")
        print(f"Quantum Layers: {num_layers}")
        print(f"CNN Pretrained: {pretrained_cnn}")
        print(f"{'='*60}\n")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete hybrid pipeline.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Class logits of shape (batch_size, 2)
        """
        # Step 1: Extract features using classical CNN
        features = self.cnn_extractor(x)
        
        # Step 2: Process through quantum circuit
        quantum_output = self.quantum_classifier(features)
        
        # Ensure correct shape
        if quantum_output.dim() == 1:
            quantum_output = quantum_output.unsqueeze(-1)
        
        # Step 3: Map to class logits
        logits = self.output_layer(quantum_output)
        
        return logits
    
    def predict_risk(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Predict risk category for input image(s).
        
        Args:
            x: Input image tensor
            
        Returns:
            Tuple of (risk_category, confidence)
            risk_category: "Low Risk" or "High Risk"
            confidence: Probability of predicted class
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            
            # Get prediction for first image in batch
            pred_class = torch.argmax(probs[0]).item()
            confidence = probs[0, pred_class].item()
            
            risk_category = "Low Risk" if pred_class == 0 else "High Risk"
            
        return risk_category, confidence
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features only (useful for analysis).
        
        Args:
            x: Input image tensor
            
        Returns:
            Feature tensor
        """
        return self.cnn_extractor(x)
    
    def freeze_cnn(self):
        """Freeze CNN parameters to only train quantum circuit."""
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False
        print("CNN parameters frozen")
    
    def unfreeze_cnn(self):
        """Unfreeze CNN parameters for end-to-end training."""
        for param in self.cnn_extractor.parameters():
            param.requires_grad = True
        print("CNN parameters unfrozen")


def create_hybrid_model(
    feature_dim: int = 8,
    num_layers: int = 2,
    pretrained_cnn: bool = True,
    device: Optional[torch.device] = None
) -> DentalHealthQMLModel:
    """
    Factory function to create a hybrid quantum-classical model.
    
    Args:
        feature_dim: Number of features/qubits
        num_layers: Number of quantum layers
        pretrained_cnn: Use pretrained CNN
        device: Device to place the model on
        
    Returns:
        Configured DentalHealthQMLModel
    """
    if device is None:
        # Quantum circuits require CPU
        device = torch.device('cpu')
        print("Note: Quantum circuits run on CPU")
    
    model = DentalHealthQMLModel(
        feature_dim=feature_dim,
        num_layers=num_layers,
        pretrained_cnn=pretrained_cnn
    )
    
    # Move CNN to device (quantum part stays on CPU)
    model.cnn_extractor = model.cnn_extractor.to(device)
    model.output_layer = model.output_layer.to(device)
    
    return model


class ModelCheckpoint:
    """Utility class for saving and loading model checkpoints."""
    
    @staticmethod
    def save_checkpoint(
        model: DentalHealthQMLModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        filepath: str
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'feature_dim': model.feature_dim,
            'num_layers': model.num_layers
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    @staticmethod
    def load_checkpoint(
        filepath: str,
        device: Optional[torch.device] = None
    ) -> Tuple[DentalHealthQMLModel, dict]:
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate model with saved configuration
        model = create_hybrid_model(
            feature_dim=checkpoint['feature_dim'],
            num_layers=checkpoint['num_layers'],
            pretrained_cnn=False,
            device=device
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Validation Accuracy: {checkpoint['val_accuracy']:.2f}%")
        
        return model, checkpoint


if __name__ == "__main__":
    # Test the hybrid model
    print("Testing Hybrid Quantum-Classical Model...")
    
    model = create_hybrid_model(
        feature_dim=4,
        num_layers=1,
        pretrained_cnn=False
    )
    
    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits:\n{output}")
    
    # Test prediction
    risk, confidence = model.predict_risk(dummy_input)
    print(f"\nPredicted Risk: {risk}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nTest passed!")

