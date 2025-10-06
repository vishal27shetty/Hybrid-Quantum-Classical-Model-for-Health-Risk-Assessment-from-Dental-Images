"""
Inference script for the Hybrid Quantum-Classical Dental Health Classifier.
Makes predictions on new dental images and outputs risk assessment.
"""

import os
import argparse
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data_loader import preprocess_single_image
from src.hybrid_model import ModelCheckpoint


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device
) -> tuple:
    """
    Predict risk category for a single image.
    
    Args:
        model: Trained hybrid quantum-classical model
        image_path: Path to the image file
        device: Device to run inference on
        
    Returns:
        Tuple of (risk_category, confidence, probabilities)
    """
    # Preprocess image
    image_tensor = preprocess_single_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # Map to risk category
        risk_category = "Low Risk" if predicted_class == 0 else "High Risk"
        
        # Get both class probabilities
        low_risk_prob = probabilities[0, 0].item()
        high_risk_prob = probabilities[0, 1].item()
    
    return risk_category, confidence, (low_risk_prob, high_risk_prob)


def visualize_prediction(
    image_path: str,
    risk_category: str,
    confidence: float,
    probabilities: tuple,
    save_path: str = None
):
    """
    Visualize the prediction result.
    
    Args:
        image_path: Path to the input image
        risk_category: Predicted risk category
        confidence: Confidence of prediction
        probabilities: Tuple of (low_risk_prob, high_risk_prob)
        save_path: Optional path to save visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    image = Image.open(image_path)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Display prediction
    categories = ['Low Risk\n(Healthy)', 'High Risk\n(Deficiency)']
    colors = ['green' if risk_category == "Low Risk" else 'lightgray',
              'red' if risk_category == "High Risk" else 'lightgray']
    
    bars = ax2.barh(categories, probabilities, color=colors)
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Risk Assessment', fontsize=14, fontweight='bold')
    
    # Add probability labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', fontweight='bold')
    
    # Add prediction text
    result_color = 'green' if risk_category == "Low Risk" else 'red'
    fig.suptitle(
        f'Prediction: {risk_category} (Confidence: {confidence:.1%})',
        fontsize=16, fontweight='bold', color=result_color
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    """Main inference function."""
    
    print("\n" + "="*60)
    print("DENTAL HEALTH QUANTUM ML - INFERENCE")
    print("="*60)
    
    # Device configuration (must be CPU for quantum)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at: {args.model_path}")
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, checkpoint = ModelCheckpoint.load_checkpoint(args.model_path, device)
    model.eval()
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image prediction
        print(f"\nProcessing image: {args.input}")
        
        risk_category, confidence, probabilities = predict_single_image(
            model, str(input_path), device
        )
        
        # Print results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Image: {input_path.name}")
        print(f"Risk Category: {risk_category}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nDetailed Probabilities:")
        print(f"  Low Risk (Healthy):        {probabilities[0]:.2%}")
        print(f"  High Risk (Deficiency):    {probabilities[1]:.2%}")
        print("="*60 + "\n")
        
        # Visualize if requested
        if args.visualize:
            save_path = None
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(
                    args.output_dir,
                    f"prediction_{input_path.stem}.png"
                )
            
            visualize_prediction(
                str(input_path), risk_category, confidence,
                probabilities, save_path
            )
    
    elif input_path.is_dir():
        # Batch prediction on directory
        print(f"\nProcessing images in directory: {args.input}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if len(image_files) == 0:
            print("No image files found in directory!")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Create output directory if needed
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each image
        results = []
        for img_path in image_files:
            try:
                risk_category, confidence, probabilities = predict_single_image(
                    model, str(img_path), device
                )
                
                results.append({
                    'filename': img_path.name,
                    'risk': risk_category,
                    'confidence': confidence,
                    'low_risk_prob': probabilities[0],
                    'high_risk_prob': probabilities[1]
                })
                
                print(f"  {img_path.name}: {risk_category} ({confidence:.1%})")
                
                # Visualize if requested
                if args.visualize and args.output_dir:
                    save_path = os.path.join(
                        args.output_dir,
                        f"prediction_{img_path.stem}.png"
                    )
                    visualize_prediction(
                        str(img_path), risk_category, confidence,
                        probabilities, save_path
                    )
            
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(results)}")
        low_risk_count = sum(1 for r in results if r['risk'] == 'Low Risk')
        high_risk_count = sum(1 for r in results if r['risk'] == 'High Risk')
        print(f"Low Risk predictions: {low_risk_count}")
        print(f"High Risk predictions: {high_risk_count}")
        print("="*60 + "\n")
        
        # Save results to CSV if output directory specified
        if args.output_dir:
            import csv
            csv_path = os.path.join(args.output_dir, 'predictions.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['filename', 'risk', 'confidence',
                               'low_risk_prob', 'high_risk_prob']
                )
                writer.writeheader()
                writer.writerows(results)
            print(f"Results saved to {csv_path}")
    
    else:
        raise ValueError(f"Input path is neither a file nor directory: {args.input}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions using trained Quantum ML model"
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to input image or directory of images'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='output/checkpoints/best_model.pth',
        help='Path to trained model checkpoint (default: output/checkpoints/best_model.pth)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save prediction results and visualizations'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of predictions'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input):
        raise ValueError(f"Input path not found: {args.input}")
    
    main(args)

