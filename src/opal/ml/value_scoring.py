"""
ML-powered consumer value scoring for payment instruments.

This module provides machine learning models to score consumer payment instruments
based on rewards rate, fees, loyalty bonuses, and card tiers.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConsumerValueFeatures(BaseModel):
    """Features for consumer value scoring."""
    
    rewards_rate: float = Field(..., description="Reward rate (e.g., 0.02 for 2% cashback)", ge=0.0, le=1.0)
    fee_rate: float = Field(..., description="Fee rate in basis points", ge=0.0)
    loyalty_bonus: float = Field(..., description="Loyalty tier bonus multiplier", ge=0.0, le=5.0)
    card_tier: int = Field(..., description="Card tier (1=basic, 2=premium, 3=elite)", ge=1, le=3)
    
    # Additional context features
    transaction_amount: float = Field(..., description="Transaction amount", gt=0.0)
    merchant_category: str = Field(default="general", description="Merchant category code")
    channel: str = Field(default="online", description="Transaction channel")
    
    # Derived features
    net_reward_value: float = Field(..., description="Net reward value after fees", ge=0.0)
    annual_fee: float = Field(default=0.0, description="Annual fee for the card", ge=0.0)


class ValueScoreResult(BaseModel):
    """Result of consumer value scoring."""
    
    value_score: float = Field(..., description="ML-predicted value score", ge=0.0, le=1.0)
    confidence: float = Field(..., description="Model confidence", ge=0.0, le=1.0)
    model_type: str = Field(..., description="Model type used")
    model_version: str = Field(..., description="Model version")
    features_used: List[str] = Field(..., description="Features used in scoring")
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class XGBoostValueScorer:
    """XGBoost-based consumer value scorer."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = [
            "rewards_rate", "fee_rate", "loyalty_bonus", "card_tier",
            "transaction_amount", "net_reward_value", "annual_fee"
        ]
        self.model_type = "xgboost"
        self.model_version = "1.0"
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._create_stub_model()
    
    def _create_stub_model(self):
        """Create a stub XGBoost model for testing."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using deterministic scoring")
            return
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features
        X = np.random.rand(n_samples, len(self.feature_names))
        X[:, 0] = np.random.uniform(0.01, 0.05, n_samples)  # rewards_rate
        X[:, 1] = np.random.uniform(0, 300, n_samples)  # fee_rate (basis points)
        X[:, 2] = np.random.uniform(1.0, 3.0, n_samples)  # loyalty_bonus
        X[:, 3] = np.random.randint(1, 4, n_samples)  # card_tier
        X[:, 4] = np.random.uniform(10, 10000, n_samples)  # transaction_amount
        X[:, 5] = X[:, 0] * X[:, 4] * X[:, 2] - X[:, 1] * X[:, 4] / 10000  # net_reward_value
        X[:, 6] = np.random.uniform(0, 500, n_samples)  # annual_fee
        
        # Generate synthetic target (value score)
        # Higher rewards, lower fees, higher loyalty = higher value
        y = (
            X[:, 0] * 20 +  # rewards rate weight
            (1 - X[:, 1] / 300) * 10 +  # inverse fee weight
            X[:, 2] * 5 +  # loyalty bonus weight
            X[:, 3] * 3 +  # card tier weight
            np.maximum(0, X[:, 5]) / 100 +  # net reward value weight
            (1 - X[:, 6] / 500) * 2  # inverse annual fee weight
        )
        y = np.clip(y / 50, 0, 1)  # Normalize to 0-1
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        self.model.fit(X, y)
        
        logger.info("✅ XGBoost value scorer trained with synthetic data")
    
    def score_consumer_value(self, features: ConsumerValueFeatures) -> ValueScoreResult:
        """Score consumer value for a payment instrument."""
        start_time = datetime.now()
        
        try:
            # Prepare features for prediction
            feature_vector = np.array([[
                features.rewards_rate,
                features.fee_rate,
                features.loyalty_bonus,
                features.card_tier,
                features.transaction_amount,
                features.net_reward_value,
                features.annual_fee
            ]])
            
            if self.model is not None:
                # XGBoost prediction
                value_score = float(self.model.predict(feature_vector)[0])
                confidence = 0.85  # High confidence for XGBoost
            else:
                # Fallback deterministic scoring
                value_score = self._deterministic_value_score(features)
                confidence = 0.7
            
            # Ensure score is within bounds
            value_score = max(0.0, min(1.0, value_score))
            
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ValueScoreResult(
                value_score=value_score,
                confidence=confidence,
                model_type=self.model_type,
                model_version=self.model_version,
                features_used=self.feature_names,
                prediction_time_ms=prediction_time
            )
            
        except Exception as e:
            logger.error(f"Error in value scoring: {e}")
            # Return fallback deterministic score
            return ValueScoreResult(
                value_score=self._deterministic_value_score(features),
                confidence=0.5,
                model_type="deterministic_fallback",
                model_version="1.0",
                features_used=self.feature_names,
                prediction_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _deterministic_value_score(self, features: ConsumerValueFeatures) -> float:
        """Deterministic fallback value scoring."""
        # Simple weighted scoring
        score = (
            features.rewards_rate * 0.3 +
            (1 - features.fee_rate / 300) * 0.25 +
            features.loyalty_bonus * 0.2 +
            features.card_tier * 0.15 +
            max(0, features.net_reward_value) / features.transaction_amount * 0.1
        )
        
        return max(0.0, min(1.0, score))
    
    def load_model(self, model_path: str):
        """Load a trained XGBoost model."""
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            logger.info(f"✅ Loaded XGBoost model from {model_path}")
        else:
            logger.warning("XGBoost not available, using deterministic scoring")


# Global scorer instance
_value_scorer: Optional[XGBoostValueScorer] = None


def get_value_scorer() -> XGBoostValueScorer:
    """Get the global value scorer instance."""
    global _value_scorer
    if _value_scorer is None:
        _value_scorer = XGBoostValueScorer()
    return _value_scorer


def score_consumer_instrument_value(
    rewards_rate: float,
    fee_rate: float,
    loyalty_bonus: float,
    card_tier: int,
    transaction_amount: float,
    net_reward_value: float,
    annual_fee: float = 0.0,
    merchant_category: str = "general",
    channel: str = "online"
) -> ValueScoreResult:
    """
    Score consumer value for a payment instrument.
    
    Args:
        rewards_rate: Reward rate (e.g., 0.02 for 2% cashback)
        fee_rate: Fee rate in basis points
        loyalty_bonus: Loyalty tier bonus multiplier
        card_tier: Card tier (1=basic, 2=premium, 3=elite)
        transaction_amount: Transaction amount
        net_reward_value: Net reward value after fees
        annual_fee: Annual fee for the card
        merchant_category: Merchant category code
        channel: Transaction channel
        
    Returns:
        ValueScoreResult with ML-predicted value score
    """
    features = ConsumerValueFeatures(
        rewards_rate=rewards_rate,
        fee_rate=fee_rate,
        loyalty_bonus=loyalty_bonus,
        card_tier=card_tier,
        transaction_amount=transaction_amount,
        merchant_category=merchant_category,
        channel=channel,
        net_reward_value=net_reward_value,
        annual_fee=annual_fee
    )
    
    scorer = get_value_scorer()
    return scorer.score_consumer_value(features)