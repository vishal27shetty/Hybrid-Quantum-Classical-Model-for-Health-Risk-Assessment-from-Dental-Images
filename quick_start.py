"""
Quick Start Script for Dental Health Quantum ML
Simplified interface for training and testing the model.
"""

import os
import sys
import argparse


def check_environment():
    """Check if required packages are installed."""
    try:
        import torch
        import qiskit
        import PIL
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False


def check_data():
    """Check if data directories exist."""
    healthy_exists = os.path.exists("healthy images")
    deficiency_exists = os.path.exists("vitamin c deficiency")
    
    if healthy_exists and deficiency_exists:
        # Count images
        healthy_count = len([f for f in os.listdir("healthy images") 
                           if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        deficiency_count = len([f for f in os.listdir("vitamin c deficiency") 
                               if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        print(f"✓ Data directories found")
        print(f"  - Healthy images: {healthy_count}")
        print(f"  - Deficiency images: {deficiency_count}")
        return True
    else:
        print("✗ Data directories not found")
        if not healthy_exists:
            print("  Missing: healthy images/")
        if not deficiency_exists:
            print("  Missing: vitamin c deficiency/")
        return False


def train_model(quick=False):
    """Train the model."""
    print("\n" + "="*60)
    print("TRAINING HYBRID QUANTUM-CLASSICAL MODEL")
    print("="*60 + "\n")
    
    if quick:
        # Quick training for testing
        cmd = "python train.py --batch_size 4 --epochs 5 --feature_dim 4"
        print("Running quick training (5 epochs, 4 qubits)...")
    else:
        # Full training
        cmd = "python train.py --pretrained --batch_size 4 --epochs 20"
        print("Running full training (20 epochs, pretrained CNN)...")
    
    print(f"Command: {cmd}\n")
    os.system(cmd)


def test_prediction():
    """Test prediction on a sample image."""
    print("\n" + "="*60)
    print("TESTING PREDICTION")
    print("="*60 + "\n")
    
    # Find a sample image
    sample_image = None
    if os.path.exists("healthy images"):
        images = [f for f in os.listdir("healthy images") 
                 if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if images:
            sample_image = os.path.join("healthy images", images[0])
    
    if not sample_image:
        print("No sample images found for testing")
        return
    
    print(f"Testing with image: {sample_image}")
    
    cmd = f'python predict.py "{sample_image}" --visualize --output_dir predictions'
    print(f"Command: {cmd}\n")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Quick Start for Dental Health Quantum ML"
    )
    parser.add_argument(
        'action',
        choices=['check', 'train', 'quick-train', 'predict', 'all'],
        help='Action to perform'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DENTAL HEALTH QUANTUM ML - QUICK START")
    print("="*60 + "\n")
    
    if args.action == 'check':
        # Check environment and data
        print("Checking environment...")
        env_ok = check_environment()
        print("\nChecking data...")
        data_ok = check_data()
        
        if env_ok and data_ok:
            print("\n✓ All checks passed! Ready to train.")
        else:
            print("\n✗ Some checks failed. Please fix the issues above.")
    
    elif args.action == 'train':
        # Full training
        if not check_environment():
            return
        if not check_data():
            return
        train_model(quick=False)
    
    elif args.action == 'quick-train':
        # Quick training for testing
        if not check_environment():
            return
        if not check_data():
            return
        train_model(quick=True)
    
    elif args.action == 'predict':
        # Test prediction
        if not check_environment():
            return
        if not os.path.exists("output/checkpoints/best_model.pth"):
            print("✗ No trained model found!")
            print("Please train the model first:")
            print("  python quick_start.py train")
            return
        test_prediction()
    
    elif args.action == 'all':
        # Run everything
        print("Running complete workflow...\n")
        
        # Check
        print("Step 1: Checking environment and data")
        if not check_environment() or not check_data():
            return
        
        # Train
        print("\nStep 2: Training model")
        response = input("Run full training (20 epochs) or quick test (5 epochs)? [full/quick]: ")
        if response.lower().startswith('q'):
            train_model(quick=True)
        else:
            train_model(quick=False)
        
        # Predict
        print("\nStep 3: Testing prediction")
        test_prediction()
        
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE!")
        print("="*60)
        print("\nTo train again: python quick_start.py train")
        print("To make predictions: python predict.py <image_path>")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nUsage: python quick_start.py [action]")
        print("\nActions:")
        print("  check       - Check environment and data")
        print("  train       - Train the model (full)")
        print("  quick-train - Quick training (5 epochs)")
        print("  predict     - Test prediction on sample image")
        print("  all         - Run complete workflow")
        print("\nExample: python quick_start.py check")
        sys.exit(0)
    
    main()

