"""
Variational Quantum Classifier (VQC) for dental health risk classification.
Uses Qiskit to implement a parameterized quantum circuit for binary classification.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class QuantumCircuitBuilder:
    """Builds parameterized quantum circuits for classification."""
    
    def __init__(self, num_qubits: int, num_layers: int = 2):
        """
        Args:
            num_qubits: Number of qubits (should match feature dimension)
            num_layers: Number of variational layers in the circuit
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
    def create_feature_map(self) -> QuantumCircuit:
        """
        Create a feature map circuit to encode classical data into quantum states.
        Uses ZZ Feature Map for entanglement.
        """
        feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=1,
            entanglement='linear'
        )
        return feature_map
    
    def create_ansatz(self) -> QuantumCircuit:
        """
        Create a variational ansatz (parameterized quantum circuit).
        Uses RealAmplitudes ansatz with full entanglement.
        """
        ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.num_layers,
            entanglement='full'
        )
        return ansatz
    
    def create_full_circuit(self) -> QuantumCircuit:
        """
        Create the complete quantum circuit by composing feature map and ansatz.
        """
        feature_map = self.create_feature_map()
        ansatz = self.create_ansatz()
        
        # Combine feature map and ansatz
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        return qc


class VariationalQuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier using Qiskit's EstimatorQNN.
    Compatible with PyTorch for end-to-end training.
    """
    
    def __init__(
        self,
        num_qubits: int = 8,
        num_layers: int = 2,
        use_parity: bool = False
    ):
        """
        Args:
            num_qubits: Number of qubits (should match CNN feature dimension)
            num_layers: Number of variational layers
            use_parity: If True, use parity measurement; otherwise use single-qubit measurement
        """
        super(VariationalQuantumClassifier, self).__init__()
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # Build quantum circuit
        circuit_builder = QuantumCircuitBuilder(num_qubits, num_layers)
        self.qc = circuit_builder.create_full_circuit()
        
        # Define observable for measurement
        if use_parity:
            # Parity measurement on all qubits
            pauli_string = 'Z' * num_qubits
            self.observable = SparsePauliOp([pauli_string], coeffs=[1.0])
        else:
            # Measure first qubit
            pauli_string = 'Z' + 'I' * (num_qubits - 1)
            self.observable = SparsePauliOp([pauli_string], coeffs=[1.0])
        
        # Create the Quantum Neural Network
        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=[self.observable],
            input_params=self.qc.parameters[:num_qubits],  # Feature map parameters
            weight_params=self.qc.parameters[num_qubits:],  # Ansatz parameters
            estimator=estimator
        )
        
        # Wrap QNN with TorchConnector for PyTorch compatibility
        self.quantum_layer = TorchConnector(self.qnn)
        
        print(f"Quantum Circuit created with {num_qubits} qubits and {num_layers} layers")
        print(f"Total parameters: {len(self.qc.parameters)}")
        print(f"Feature parameters: {num_qubits}")
        print(f"Weight parameters: {len(self.qc.parameters) - num_qubits}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.
        
        Args:
            x: Input feature tensor of shape (batch_size, num_qubits)
            
        Returns:
            Quantum circuit output (expectation values)
        """
        # Ensure input is in correct range [-1, 1] for feature map
        x = torch.clamp(x, -1, 1)
        
        # Pass through quantum layer
        output = self.quantum_layer(x)
        
        return output


class HybridQuantumClassifier(nn.Module):
    """
    Hybrid model that combines classical preprocessing with quantum classification.
    This is a standalone quantum classifier that can accept features directly.
    """
    
    def __init__(
        self,
        num_qubits: int = 8,
        num_layers: int = 2,
        num_classes: int = 2
    ):
        """
        Args:
            num_qubits: Number of qubits
            num_layers: Number of variational layers
            num_classes: Number of output classes
        """
        super(HybridQuantumClassifier, self).__init__()
        
        self.vqc = VariationalQuantumClassifier(
            num_qubits=num_qubits,
            num_layers=num_layers
        )
        
        # Post-processing layer to convert quantum output to class probabilities
        # Quantum output is an expectation value in range [-1, 1]
        # We map this to class logits
        self.output_layer = nn.Linear(1, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid quantum classifier.
        
        Args:
            x: Input feature tensor
            
        Returns:
            Class logits
        """
        # Get quantum expectation value
        quantum_output = self.vqc(x)
        
        # Ensure correct shape (batch_size, 1)
        if quantum_output.dim() == 1:
            quantum_output = quantum_output.unsqueeze(-1)
        
        # Convert to class logits
        logits = self.output_layer(quantum_output)
        
        return logits


def create_quantum_classifier(
    num_qubits: int = 8,
    num_layers: int = 2,
    device: Optional[torch.device] = None
) -> HybridQuantumClassifier:
    """
    Factory function to create a quantum classifier.
    
    Args:
        num_qubits: Number of qubits
        num_layers: Number of variational layers
        device: Device to place the model on (quantum circuits run on CPU)
        
    Returns:
        Configured HybridQuantumClassifier
    """
    if device is None:
        # Quantum circuits run on CPU
        device = torch.device('cpu')
    
    model = HybridQuantumClassifier(
        num_qubits=num_qubits,
        num_layers=num_layers,
        num_classes=2
    )
    model = model.to(device)
    
    print(f"Quantum classifier created and placed on device: {device}")
    
    return model


if __name__ == "__main__":
    # Test the quantum classifier
    print("Testing Variational Quantum Classifier...")
    
    model = create_quantum_classifier(num_qubits=4, num_layers=1)
    
    # Create dummy input (batch_size=2, features=4)
    dummy_input = torch.randn(2, 4) * 0.5  # Small values
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input:\n{dummy_input}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output logits:\n{output}")
    
    # Convert to probabilities
    probs = torch.softmax(output, dim=1)
    print(f"\nOutput probabilities:\n{probs}")
    
    print("\nTest passed!")

