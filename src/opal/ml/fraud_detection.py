"""
ML-powered fraud detection for wallet transactions.

This module provides machine learning models to detect fraudulent transactions
based on behavioral patterns, velocity checks, and anomaly detection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FraudDetectionFeatures(BaseModel):
    """Features for fraud detection."""

    # Transaction features
    amount: float = Field(..., description="Transaction amount", gt=0.0)
    velocity_24h: int = Field(..., description="Number of transactions in last 24h", ge=0)
    velocity_amount_24h: float = Field(..., description="Total amount in last 24h", ge=0.0)

    # Behavioral features
    merchant_category: str = Field(..., description="Merchant category code")
    channel: str = Field(..., description="Transaction channel")
    time_of_day: int = Field(..., description="Hour of day (0-23)")
    day_of_week: int = Field(..., description="Day of week (0-6)")

    # Account features
    account_age_days: int = Field(..., description="Account age in days", ge=0)
    avg_transaction_amount: float = Field(..., description="Average transaction amount", ge=0.0)
    transaction_frequency: float = Field(..., description="Transactions per day", ge=0.0)

    # Risk indicators
    new_merchant: bool = Field(..., description="Is this a new merchant for user")
    unusual_amount: bool = Field(..., description="Is amount unusual for user")
    cross_border: bool = Field(..., description="Is this a cross-border transaction")

    # Derived features
    amount_to_avg_ratio: float = Field(..., description="Amount / average amount ratio", ge=0.0)
    velocity_risk_score: float = Field(..., description="Velocity-based risk score", ge=0.0, le=1.0)


class FraudDetectionResult(BaseModel):
    """Result of fraud detection."""

    is_fraud: bool = Field(..., description="Fraud prediction")
    fraud_probability: float = Field(..., description="Fraud probability", ge=0.0, le=1.0)
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    confidence: float = Field(..., description="Model confidence", ge=0.0, le=1.0)
    model_type: str = Field(..., description="Model type used")
    model_version: str = Field(..., description="Model version")
    features_used: List[str] = Field(..., description="Features used in detection")
    risk_factors: List[str] = Field(..., description="Key risk factors identified")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")


class FraudDetectionModel:
    """ML model for fraud detection using Random Forest."""

    def __init__(self, model_dir: str = "models/fraud_detection"):
        """Initialize the fraud detection model."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[RandomForestClassifier] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self._load_model()

    def _load_model(self):
        """Load or create the fraud detection model."""
        model_path = self.model_dir / "fraud_detection_model.pkl"
        scaler_path = self.model_dir / "fraud_detection_model_scaler.pkl"
        metadata_path = self.model_dir / "fraud_detection_model_metadata.json"

        if model_path.exists() and scaler_path.exists() and metadata_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)

                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)

                self.feature_names = self.metadata.get("feature_names", [])
                self.is_loaded = True
                logger.info(f"✅ Loaded fraud detection model from {self.model_dir}")
            except Exception as e:
                logger.error(f"Failed to load fraud detection model: {e}")
                self._create_model()
        else:
            self._create_model()

    def _create_model(self):
        """Create and train a new fraud detection model."""
        logger.info("Creating new fraud detection model...")

        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 5000

        # Feature names
        self.feature_names = [
            "amount",
            "velocity_24h",
            "velocity_amount_24h",
            "time_of_day",
            "day_of_week",
            "account_age_days",
            "avg_transaction_amount",
            "transaction_frequency",
            "new_merchant",
            "unusual_amount",
            "cross_border",
            "amount_to_avg_ratio",
            "velocity_risk_score",
        ]

        # Generate synthetic features
        X = np.random.rand(n_samples, len(self.feature_names))

        # Realistic feature distributions
        X[:, 0] = np.random.lognormal(3, 1, n_samples)  # amount (log-normal)
        X[:, 1] = np.random.poisson(5, n_samples)  # velocity_24h
        X[:, 2] = X[:, 1] * np.random.lognormal(3, 0.5, n_samples)  # velocity_amount_24h
        X[:, 3] = np.random.randint(0, 24, n_samples)  # time_of_day
        X[:, 4] = np.random.randint(0, 7, n_samples)  # day_of_week
        X[:, 5] = np.random.uniform(30, 3650, n_samples)  # account_age_days
        X[:, 6] = np.random.lognormal(3, 0.8, n_samples)  # avg_transaction_amount
        X[:, 7] = np.random.uniform(0.1, 10, n_samples)  # transaction_frequency
        X[:, 8] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # new_merchant
        X[:, 9] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # unusual_amount
        X[:, 10] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # cross_border
        X[:, 11] = X[:, 0] / (X[:, 6] + 1e-6)  # amount_to_avg_ratio
        X[:, 12] = np.clip(X[:, 1] / 20 + X[:, 11] / 10, 0, 1)  # velocity_risk_score

        # Generate synthetic labels (fraud vs legitimate)
        # Higher fraud probability for unusual patterns
        fraud_prob = (
            (X[:, 1] > 15) * 0.3  # High velocity
            + (X[:, 9] == 1) * 0.4  # Unusual amount
            + (X[:, 10] == 1) * 0.2  # Cross border
            + (X[:, 8] == 1) * 0.1  # New merchant
            + (X[:, 3] < 6) * 0.1  # Late night
            + (X[:, 11] > 5) * 0.3  # High amount ratio
        )

        y = np.random.binomial(1, fraud_prob)

        # Train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )
        self.model.fit(X_scaled, y)

        # Calibrate probabilities
        self.calibrator = CalibratedClassifierCV(self.model, method="isotonic", cv=3)
        self.calibrator.fit(X_scaled, y)

        # Save model
        self.save_model()

        logger.info("✅ Fraud detection model trained and saved")

    def detect_fraud(self, features: FraudDetectionFeatures) -> FraudDetectionResult:
        """Detect fraud in a transaction."""
        start_time = datetime.now()

        try:
            # Prepare features for prediction
            feature_vector = np.array(
                [
                    [
                        features.amount,
                        features.velocity_24h,
                        features.velocity_amount_24h,
                        features.time_of_day,
                        features.day_of_week,
                        features.account_age_days,
                        features.avg_transaction_amount,
                        features.transaction_frequency,
                        float(features.new_merchant),
                        float(features.unusual_amount),
                        float(features.cross_border),
                        features.amount_to_avg_ratio,
                        features.velocity_risk_score,
                    ]
                ]
            )

            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Predict fraud probability
            if self.calibrator is not None:
                fraud_probability = self.calibrator.predict_proba(feature_vector_scaled)[0][1]
            else:
                fraud_probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            is_fraud = fraud_probability > 0.5

            # Determine risk level
            if fraud_probability < 0.3:
                risk_level = "low"
            elif fraud_probability < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"

            # Identify risk factors
            risk_factors = []
            if features.velocity_24h > 10:
                risk_factors.append("High transaction velocity")
            if features.unusual_amount:
                risk_factors.append("Unusual transaction amount")
            if features.cross_border:
                risk_factors.append("Cross-border transaction")
            if features.new_merchant:
                risk_factors.append("New merchant")
            if features.time_of_day < 6 or features.time_of_day > 22:
                risk_factors.append("Unusual transaction time")
            if features.amount_to_avg_ratio > 5:
                risk_factors.append("Amount significantly above average")

            prediction_time = (datetime.now() - start_time).total_seconds() * 1000

            return FraudDetectionResult(
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                risk_level=risk_level,
                confidence=0.85,
                model_type="RandomForest",
                model_version="1.0.0",
                features_used=self.feature_names,
                risk_factors=risk_factors,
                prediction_time_ms=prediction_time,
            )

        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            # Return conservative fallback
            return FraudDetectionResult(
                is_fraud=False,
                fraud_probability=0.1,
                risk_level="low",
                confidence=0.5,
                model_type="fallback",
                model_version="1.0.0",
                features_used=self.feature_names,
                risk_factors=[],
                prediction_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    def save_model(self, model_name: str = "fraud_detection_model") -> None:
        """Save the trained model and metadata."""
        model_path = self.model_dir / f"{model_name}.pkl"
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"
        metadata_path = self.model_dir / f"{model_name}_metadata.json"

        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Feature importance (convert numpy types to Python types for JSON serialization)
        feature_importance = dict(
            zip(self.feature_names, [float(x) for x in self.model.feature_importances_])
        )

        # Save metadata
        self.metadata = {
            "model_type": "RandomForestClassifier",
            "version": "1.0.0",
            "trained_on": datetime.now().isoformat(),
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "feature_importance": feature_importance,
            "model_parameters": {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
        }

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"✅ Fraud detection model saved to {self.model_dir}")


# Global model instance
_fraud_model: Optional[FraudDetectionModel] = None


def get_fraud_model() -> FraudDetectionModel:
    """Get the global fraud detection model instance."""
    global _fraud_model
    if _fraud_model is None:
        _fraud_model = FraudDetectionModel()
    return _fraud_model


def detect_transaction_fraud(
    amount: float,
    velocity_24h: int,
    velocity_amount_24h: float,
    merchant_category: str,
    channel: str,
    time_of_day: int,
    day_of_week: int,
    account_age_days: int,
    avg_transaction_amount: float,
    transaction_frequency: float,
    new_merchant: bool = False,
    unusual_amount: bool = False,
    cross_border: bool = False,
) -> FraudDetectionResult:
    """
    Detect fraud in a transaction.

    Args:
        amount: Transaction amount
        velocity_24h: Number of transactions in last 24h
        velocity_amount_24h: Total amount in last 24h
        merchant_category: Merchant category code
        channel: Transaction channel
        time_of_day: Hour of day (0-23)
        day_of_week: Day of week (0-6)
        account_age_days: Account age in days
        avg_transaction_amount: Average transaction amount
        transaction_frequency: Transactions per day
        new_merchant: Is this a new merchant for user
        unusual_amount: Is amount unusual for user
        cross_border: Is this a cross-border transaction

    Returns:
        FraudDetectionResult with fraud prediction
    """
    # Calculate derived features
    amount_to_avg_ratio = amount / (avg_transaction_amount + 1e-6)
    velocity_risk_score = min(1.0, velocity_24h / 20.0 + amount_to_avg_ratio / 10.0)

    features = FraudDetectionFeatures(
        amount=amount,
        velocity_24h=velocity_24h,
        velocity_amount_24h=velocity_amount_24h,
        merchant_category=merchant_category,
        channel=channel,
        time_of_day=time_of_day,
        day_of_week=day_of_week,
        account_age_days=account_age_days,
        avg_transaction_amount=avg_transaction_amount,
        transaction_frequency=transaction_frequency,
        new_merchant=new_merchant,
        unusual_amount=unusual_amount,
        cross_border=cross_border,
        amount_to_avg_ratio=amount_to_avg_ratio,
        velocity_risk_score=velocity_risk_score,
    )

    model = get_fraud_model()
    return model.detect_fraud(features)
