"""
ML-enhanced spend controls with fraud detection.

This module integrates traditional spend controls with ML-powered fraud detection
to provide more intelligent transaction authorization decisions.
"""

import os
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import ML models
from .ml.fraud_detection import get_fraud_model, detect_transaction_fraud
from .ml.value_scoring import get_value_scorer

# Import existing controls
from .controls import SpendControls, TransactionRequest, SpendControlResult


class MLEnhancedControls:
    """ML-enhanced spend controls with fraud detection and value scoring."""
    
    def __init__(self, ml_weight: float = 0.7, use_ml: bool = True):
        """Initialize ML-enhanced controls."""
        self.ml_weight = ml_weight
        self.use_ml = use_ml
        self.fraud_model = get_fraud_model()
        self.value_scorer = get_value_scorer()
    
    def evaluate_transaction(self, transaction_request: TransactionRequest) -> SpendControlResult:
        """
        Evaluate transaction with ML-enhanced fraud detection.
        
        Args:
            transaction_request: Transaction request to evaluate
            
        Returns:
            SpendControlResult with ML-enhanced decision
        """
        # First, run traditional spend controls
        traditional_result = SpendControls.evaluate_transaction(transaction_request)
        
        if not self.use_ml:
            return traditional_result
        
        try:
            # Run fraud detection
            fraud_result = self._detect_fraud(transaction_request)
            
            # If fraud detected with high confidence, deny transaction
            if fraud_result.is_fraud and fraud_result.fraud_probability > 0.7:
                return SpendControlResult(
                    allowed=False,
                    token_reference=None,
                    reasons=[
                        "Transaction declined due to fraud risk",
                        f"Fraud probability: {fraud_result.fraud_probability:.1%}",
                        f"Risk level: {fraud_result.risk_level}",
                        f"Risk factors: {', '.join(fraud_result.risk_factors)}"
                    ],
                    limits_applied=traditional_result.limits_applied + ["ml_fraud_detection"],
                    max_amount_allowed=None,
                    control_version="ml-enhanced-v1.0.0"
                )
            
            # If traditional controls deny, but ML says low fraud risk, 
            # consider allowing with reduced amount or additional monitoring
            if not traditional_result.allowed and fraud_result.fraud_probability < 0.3:
                # Allow with reduced amount if fraud risk is low
                reduced_amount = min(
                    float(transaction_request.amount) * 0.8,
                    fraud_result.fraud_probability < 0.1 and float(transaction_request.amount) or 0
                )
                
                if reduced_amount > 0:
                    return SpendControlResult(
                        allowed=True,
                        token_reference=f"ml_monitored_{transaction_request.actor_id}",
                        reasons=[
                            "Traditional limits exceeded but low fraud risk",
                            f"Allowed reduced amount: ${reduced_amount:.2f}",
                            f"Fraud probability: {fraud_result.fraud_probability:.1%}",
                            "Transaction will be monitored"
                        ],
                        limits_applied=traditional_result.limits_applied + ["ml_fraud_detection", "ml_amount_reduction"],
                        max_amount_allowed=Decimal(str(reduced_amount)),
                        control_version="ml-enhanced-v1.0.0"
                    )
            
            # If traditional controls allow, enhance with fraud risk information
            if traditional_result.allowed:
                enhanced_reasons = traditional_result.reasons.copy()
                enhanced_reasons.extend([
                    f"ML fraud probability: {fraud_result.fraud_probability:.1%}",
                    f"ML risk level: {fraud_result.risk_level}"
                ])
                
                if fraud_result.risk_factors:
                    enhanced_reasons.append(f"Risk factors: {', '.join(fraud_result.risk_factors)}")
                
                return SpendControlResult(
                    allowed=True,
                    token_reference=traditional_result.token_reference,
                    reasons=enhanced_reasons,
                    limits_applied=traditional_result.limits_applied + ["ml_fraud_detection"],
                    max_amount_allowed=traditional_result.max_amount_allowed,
                    control_version="ml-enhanced-v1.0.0"
                )
            
            # Default to traditional result with ML information
            return traditional_result
            
        except Exception as e:
            # Fallback to traditional controls if ML fails
            print(f"⚠️ ML fraud detection failed: {e}")
            return traditional_result
    
    def _detect_fraud(self, transaction_request: TransactionRequest) -> Any:
        """Run fraud detection on the transaction."""
        # This is a simplified implementation - in production you'd have
        # access to user behavioral data, transaction history, etc.
        
        # Mock user profile data (in production, this would come from a database)
        user_profile = self._get_user_profile(transaction_request.actor_id)
        
        # Detect fraud
        return detect_transaction_fraud(
            amount=float(transaction_request.amount),
            velocity_24h=user_profile["velocity_24h"],
            velocity_amount_24h=user_profile["velocity_amount_24h"],
            merchant_category=transaction_request.mcc or "general",
            channel=transaction_request.channel,
            time_of_day=user_profile["time_of_day"],
            day_of_week=user_profile["day_of_week"],
            account_age_days=user_profile["account_age_days"],
            avg_transaction_amount=user_profile["avg_transaction_amount"],
            transaction_frequency=user_profile["transaction_frequency"],
            new_merchant=user_profile["new_merchant"],
            unusual_amount=user_profile["unusual_amount"],
            cross_border=user_profile["cross_border"]
        )
    
    def _get_user_profile(self, actor_id: str) -> Dict[str, Any]:
        """Get user profile data for fraud detection (mock implementation)."""
        # In production, this would query a user database
        import random
        random.seed(hash(actor_id) % 1000)  # Deterministic for testing
        
        return {
            "velocity_24h": random.randint(0, 15),
            "velocity_amount_24h": random.uniform(0, 5000),
            "time_of_day": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "account_age_days": random.randint(30, 3650),
            "avg_transaction_amount": random.uniform(50, 500),
            "transaction_frequency": random.uniform(0.5, 5.0),
            "new_merchant": random.choice([True, False]),
            "unusual_amount": random.choice([True, False]),
            "cross_border": random.choice([True, False])
        }
    
    def get_control_limits(self) -> Dict[str, Any]:
        """Get ML-enhanced control limits."""
        traditional_limits = SpendControls.get_control_limits()
        
        ml_limits = {
            "ml_enabled": self.use_ml,
            "ml_weight": self.ml_weight,
            "fraud_detection": {
                "model_type": "RandomForest",
                "version": "1.0.0",
                "features": 13,
                "risk_thresholds": {
                    "low": 0.3,
                    "medium": 0.7,
                    "high": 1.0
                }
            },
            "value_scoring": {
                "model_type": "XGBoost",
                "version": "1.0.0",
                "features": 7
            }
        }
        
        traditional_limits.update(ml_limits)
        return traditional_limits


# Global ML-enhanced controls instance
ml_enhanced_controls = MLEnhancedControls()
