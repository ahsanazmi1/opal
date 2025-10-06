"""
Training script for Opal ML models.

This script trains and saves the fraud detection and value scoring models.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opal.ml.fraud_detection import FraudDetectionModel


def train_fraud_detection_model():
    """Train the fraud detection model."""
    print("🔧 Training fraud detection model...")
    
    model = FraudDetectionModel()
    model.save_model()
    
    print("✅ Fraud detection model trained and saved")


def train_value_scoring_model():
    """Train the value scoring model."""
    print("🔧 Training value scoring model...")
    
    # The value scoring model is already trained in its initialization
    # This is just a placeholder for future enhancements
    from opal.ml.value_scoring import get_value_scorer
    
    scorer = get_value_scorer()
    print("✅ Value scoring model ready")


def main():
    """Main training function."""
    print("🚀 Starting Opal ML model training...")
    
    try:
        train_fraud_detection_model()
        train_value_scoring_model()
        
        print("🎉 All ML models trained successfully!")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
