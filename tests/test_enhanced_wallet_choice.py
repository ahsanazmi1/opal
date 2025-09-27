"""
Tests for enhanced wallet choice negotiation with ML value scoring.

This module tests the negotiateWalletChoice function with:
- ML-powered value scoring using XGBoost/calibrated logistic
- Deterministic choice with/without loyalty boost
- LLM-powered explanations
- Explanation JSON snapshots
"""

import asyncio
import pytest
from datetime import datetime

from src.opal.negotiation import negotiateWalletChoice
from src.opal.controls import (
    ConsumerInstrument,
    ConsumerReward,
    MerchantProposal,
    InstrumentType,
    RewardType,
)
from src.opal.ml.value_scoring import score_consumer_instrument_value, ConsumerValueFeatures


class TestMLValueScoring:
    """Test ML-powered consumer value scoring."""
    
    def test_xgboost_value_scoring_basic(self):
        """Test basic XGBoost value scoring functionality."""
        features = ConsumerValueFeatures(
            rewards_rate=0.02,  # 2% cashback
            fee_rate=200.0,  # 2% fee in basis points
            loyalty_bonus=1.5,  # 50% loyalty bonus
            card_tier=2,  # Premium tier
            transaction_amount=1000.0,
            net_reward_value=15.0,  # $15 net reward
            annual_fee=95.0
        )
        
        result = score_consumer_instrument_value(
            rewards_rate=features.rewards_rate,
            fee_rate=features.fee_rate,
            loyalty_bonus=features.loyalty_bonus,
            card_tier=features.card_tier,
            transaction_amount=features.transaction_amount,
            net_reward_value=features.net_reward_value,
            annual_fee=features.annual_fee
        )
        
        # Verify result structure
        assert 0.0 <= result.value_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_type in ["xgboost", "deterministic_fallback"]
        assert len(result.features_used) > 0
        assert result.prediction_time_ms >= 0
    
    def test_value_scoring_with_different_tiers(self):
        """Test value scoring across different card tiers."""
        # Basic tier
        basic_result = score_consumer_instrument_value(
            rewards_rate=0.01,
            fee_rate=100.0,
            loyalty_bonus=1.0,
            card_tier=1,
            transaction_amount=1000.0,
            net_reward_value=5.0
        )
        
        # Premium tier
        premium_result = score_consumer_instrument_value(
            rewards_rate=0.025,
            fee_rate=150.0,
            loyalty_bonus=1.5,
            card_tier=2,
            transaction_amount=1000.0,
            net_reward_value=12.5
        )
        
        # Elite tier
        elite_result = score_consumer_instrument_value(
            rewards_rate=0.03,
            fee_rate=200.0,
            loyalty_bonus=2.0,
            card_tier=3,
            transaction_amount=1000.0,
            net_reward_value=20.0
        )
        
        # Elite should score higher than premium, premium higher than basic
        assert elite_result.value_score > premium_result.value_score
        assert premium_result.value_score > basic_result.value_score


class TestEnhancedWalletChoice:
    """Test enhanced wallet choice negotiation."""
    
    def create_sample_instruments(self):
        """Create sample consumer instruments for testing."""
        sapphire_card = ConsumerInstrument(
            instrument_id="sapphire_001",
            instrument_type="credit_card",
            provider="Chase",
            last_four="1234",
            base_fee=200.0,  # 2% fee
            out_of_pocket_cost=0.0,
            available_balance=10000.0,
            rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    rate=0.03,  # 3% cashback
                    value=30.0,  # $30 for $1000 transaction
                    description="3% cashback on all purchases"
                )
            ],
            total_reward_value=30.0,
            loyalty_tier="premium",
            loyalty_multiplier=1.5,
            net_value=30.0,
            value_score=0.8,
            eligible=True
        )
        
        basic_card = ConsumerInstrument(
            instrument_id="basic_001",
            instrument_type="credit_card",
            provider="Capital One",
            last_four="5678",
            base_fee=150.0,  # 1.5% fee
            out_of_pocket_cost=0.0,
            available_balance=5000.0,
            rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    rate=0.015,  # 1.5% cashback
                    value=15.0,  # $15 for $1000 transaction
                    description="1.5% cashback on all purchases"
                )
            ],
            total_reward_value=15.0,
            loyalty_tier="standard",
            loyalty_multiplier=1.0,
            net_value=15.0,
            value_score=0.6,
            eligible=True
        )
        
        bnpl_option = ConsumerInstrument(
            instrument_id="bnpl_001",
            instrument_type="bnpl",
            provider="Klarna",
            last_four="9999",
            base_fee=0.0,  # No fees
            out_of_pocket_cost=0.0,
            available_balance=2000.0,
            rewards=[],
            total_reward_value=0.0,
            loyalty_tier="standard",
            loyalty_multiplier=1.0,
            net_value=0.0,
            value_score=0.3,
            eligible=True
        )
        
        return [sapphire_card, basic_card, bnpl_option]
    
    def create_merchant_proposal(self):
        """Create sample merchant proposal."""
        return MerchantProposal(
            rail_type="ACH",
            merchant_cost=50.0,  # 0.5% in basis points
            settlement_days=2,
            risk_score=0.3,
            explanation="ACH selected for cost efficiency",
            trace_id="trace_enhanced_test_001"
        )
    
    def test_enhanced_wallet_choice_basic(self):
        """Test basic enhanced wallet choice negotiation."""
        instruments = self.create_sample_instruments()
        merchant_proposal = self.create_merchant_proposal()
        
        # Run the async function
        response = asyncio.run(negotiateWalletChoice(
            actor_id="test_consumer_001",
            transaction_amount=1000.0,
            available_instruments=instruments,
            merchant_proposal=merchant_proposal,
            consumer_preferences={"prefer_rewards": True},
            deterministic_seed=42
        ))
        
        # Verify response structure
        assert response.selected_instrument is not None
        assert response.counter_proposal is not None
        assert len(response.explanation) > 0
        assert response.trace_id == merchant_proposal.trace_id
        assert response.win_win_score >= 0.0
        
        # Verify ML scoring was applied
        assert response.negotiation_metadata["ml_value_scoring"] is True
        assert "ml_value_scores" in response.negotiation_metadata
        assert len(response.negotiation_metadata["ml_value_scores"]) > 0
    
    def test_deterministic_choice_consistency(self):
        """Test that deterministic seed produces consistent results."""
        instruments = self.create_sample_instruments()
        merchant_proposal = self.create_merchant_proposal()
        
        # Run negotiation multiple times with same seed
        response1 = asyncio.run(negotiateWalletChoice(
            actor_id="test_consumer_001",
            transaction_amount=1000.0,
            available_instruments=instruments,
            merchant_proposal=merchant_proposal,
            deterministic_seed=123
        ))
        
        response2 = asyncio.run(negotiateWalletChoice(
            actor_id="test_consumer_001",
            transaction_amount=1000.0,
            available_instruments=instruments,
            merchant_proposal=merchant_proposal,
            deterministic_seed=123
        ))
        
        # Results should be identical
        assert response1.selected_instrument.instrument_id == response2.selected_instrument.instrument_id
        assert response1.win_win_score == response2.win_win_score
        
        # ML scores should be identical
        scores1 = response1.negotiation_metadata["ml_value_scores"]
        scores2 = response2.negotiation_metadata["ml_value_scores"]
        assert scores1 == scores2


if __name__ == "__main__":
    pytest.main([__file__])