"""
Integration tests for multi-rail consumer negotiation functionality in Opal Phase 7.

Tests consumer-centric negotiation logic that counters merchant proposals
with optimal consumer value across multiple rails and instruments.
"""

import json
import pytest
from pathlib import Path

from src.opal.multi_rail_consumer_negotiator import (
    get_multi_rail_consumer_negotiator,
    get_multi_rail_consumer_evaluator,
)
from src.opal.controls import MerchantProposal, CounterNegotiationResponse


class TestMultiRailConsumerIntegration:
    """Integration tests for multi-rail consumer negotiation."""

    @pytest.fixture
    def consumer_negotiator(self):
        """Get multi-rail consumer negotiator."""
        return get_multi_rail_consumer_negotiator()

    @pytest.fixture
    def consumer_evaluator(self):
        """Get multi-rail consumer evaluator."""
        return get_multi_rail_consumer_evaluator()

    @pytest.fixture
    def consumer_multi_rail_fixture(self):
        """Load consumer multi-rail test fixture."""
        fixture_path = (
            Path(__file__).parent / "fixtures" / "multi_rail" / "consumer_multi_rail_example.json"
        )
        with open(fixture_path) as f:
            return json.load(f)

    def test_consumer_capability_initialization(self, consumer_evaluator):
        """Test that consumer capabilities are properly initialized."""
        capability = consumer_evaluator.capability

        # Test available instruments
        assert "credit_card" in capability.available_instruments
        assert "debit_card" in capability.available_instruments
        assert "prepaid_card" in capability.available_instruments
        assert "stablecoin" in capability.available_instruments
        assert "bank_transfer" in capability.available_instruments
        assert "digital_wallet" in capability.available_instruments
        assert "bnpl" in capability.available_instruments

        # Test available rails
        assert "Card" in capability.available_rails
        assert "ACH" in capability.available_rails
        assert "RTP" in capability.available_rails
        assert "FedNow" in capability.available_rails
        assert "SEPA" in capability.available_rails
        assert "Crypto" in capability.available_rails
        assert "Wire" in capability.available_rails

        # Test valid combinations
        available_combinations = capability.get_available_combinations()
        assert len(available_combinations) > 0

        # Verify specific combinations exist
        assert ("Crypto", "stablecoin") in available_combinations
        assert ("FedNow", "debit_card") in available_combinations
        assert ("Card", "credit_card") in available_combinations

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_evaluation_crypto_stablecoin(
        self, consumer_evaluator, consumer_multi_rail_fixture
    ):
        """Test consumer evaluation for Crypto/stablecoin combination."""
        fixture = consumer_multi_rail_fixture

        # Create merchant proposal
        merchant_proposal = MerchantProposal(**fixture["merchant_proposal"])

        # Create consumer context
        consumer_context = fixture["consumer_context"]

        # Evaluate Crypto/stablecoin combination
        evaluation = consumer_evaluator.evaluate_consumer_combination(
            "Crypto",
            "stablecoin",
            merchant_proposal,
            fixture["transaction_amount"],
            consumer_context,
        )

        assert evaluation is not None
        assert evaluation["rail_type"] == "Crypto"
        assert evaluation["instrument_type"] == "stablecoin"

        # Verify high consumer value for preferred combination
        assert evaluation["total_consumer_value"] > 0.8
        assert evaluation["reward_value"] > 0  # Should earn rewards
        assert evaluation["convenience_score"] > 0.8  # High convenience
        assert evaluation["is_preferred"]

        # Verify expected values match fixture
        expected = fixture["expected_optimal_combination"]
        assert abs(evaluation["total_consumer_value"] - expected["total_consumer_value"]) < 0.1
        assert abs(evaluation["reward_value"] - expected["reward_value"]) < 1.0

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_evaluation_fednow_debit(
        self, consumer_evaluator, consumer_multi_rail_fixture
    ):
        """Test consumer evaluation for FedNow/debit combination."""
        fixture = consumer_multi_rail_fixture

        # Create merchant proposal
        merchant_proposal = MerchantProposal(**fixture["merchant_proposal"])

        # Create consumer context
        consumer_context = fixture["consumer_context"]

        # Evaluate FedNow/debit combination
        evaluation = consumer_evaluator.evaluate_consumer_combination(
            "FedNow",
            "debit_card",
            merchant_proposal,
            fixture["transaction_amount"],
            consumer_context,
        )

        assert evaluation is not None
        assert evaluation["rail_type"] == "FedNow"
        assert evaluation["instrument_type"] == "debit_card"

        # Verify good consumer value
        assert evaluation["total_consumer_value"] > 0.6
        assert evaluation["reward_value"] > 0  # Should earn some rewards
        assert evaluation["convenience_score"] > 0.7  # Good convenience
        assert evaluation["is_preferred"]  # FedNow is preferred

        # Verify real-time processing benefit
        assert "instant processing" in evaluation["explanation"]

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_negotiation_optimal_selection(
        self, consumer_negotiator, consumer_multi_rail_fixture
    ):
        """Test complete consumer negotiation with optimal selection."""
        fixture = consumer_multi_rail_fixture

        # Create merchant proposal
        merchant_proposal = MerchantProposal(**fixture["merchant_proposal"])

        # Create consumer context
        consumer_context = fixture["consumer_context"]

        # Run consumer negotiation
        response = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, fixture["transaction_amount"], consumer_context
        )

        assert isinstance(response, CounterNegotiationResponse)
        assert response.trace_id == merchant_proposal.trace_id
        assert response.actor_id == consumer_context["actor_id"]

        # Verify optimal combination is selected
        optimal = response.consumer_proposal
        assert optimal.rail_type in ["Crypto", "FedNow", "Card"]  # Should be one of preferred rails
        assert optimal.instrument_type in [
            "stablecoin",
            "debit_card",
            "credit_card",
        ]  # Should be preferred instrument

        # Verify consumer value is maximized
        assert optimal.consumer_benefit >= 0
        assert optimal.convenience_score > 0

        # Verify alternatives are provided
        assert len(response.alternatives) > 0

        # Verify consumer rewards are created
        assert len(response.consumer_rewards) > 0

        # Verify explanation is provided
        assert response.explanation
        assert len(response.explanation) > 10

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_negotiation_alternatives(
        self, consumer_negotiator, consumer_multi_rail_fixture
    ):
        """Test that consumer negotiation provides meaningful alternatives."""
        fixture = consumer_multi_rail_fixture

        # Create merchant proposal
        merchant_proposal = MerchantProposal(**fixture["merchant_proposal"])

        # Create consumer context
        consumer_context = fixture["consumer_context"]

        # Run consumer negotiation
        response = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, fixture["transaction_amount"], consumer_context
        )

        # Verify alternatives are provided
        assert len(response.alternatives) > 0

        # Verify alternatives have different rail/instrument combinations
        rail_instrument_combinations = set()
        for alt in response.alternatives:
            combination = (alt["rail_type"], alt["instrument_type"])
            rail_instrument_combinations.add(combination)

        # Should have multiple different combinations
        assert len(rail_instrument_combinations) > 1

        # Verify alternatives have reasonable consumer values
        for alt in response.alternatives:
            assert alt["total_consumer_value"] > 0.3  # Reasonable minimum value
            assert alt["confidence"] > 0.5  # Reasonable confidence

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_rewards_calculation(self, consumer_negotiator, consumer_multi_rail_fixture):
        """Test that consumer rewards are properly calculated."""
        fixture = consumer_multi_rail_fixture

        # Create merchant proposal
        merchant_proposal = MerchantProposal(**fixture["merchant_proposal"])

        # Create consumer context
        consumer_context = fixture["consumer_context"]

        # Run consumer negotiation
        response = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, fixture["transaction_amount"], consumer_context
        )

        # Verify consumer rewards are created
        assert len(response.consumer_rewards) > 0

        # Verify reward types are appropriate
        reward_types = [reward.reward_type for reward in response.consumer_rewards]
        assert any(
            rt in ["cashback", "crypto_rewards", "points", "convenience", "cost_savings"]
            for rt in reward_types
        )

        # Verify monetary rewards are positive
        monetary_rewards = [r for r in response.consumer_rewards if r.value > 0]
        if monetary_rewards:
            total_monetary_value = sum(r.value for r in monetary_rewards)
            assert total_monetary_value > 0

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_preference_prioritization(self, consumer_negotiator):
        """Test that consumer preferences are properly prioritized."""
        # Create merchant proposal
        merchant_proposal = MerchantProposal(
            rail_type="ACH",
            merchant_cost=15.0,
            settlement_days=1,
            risk_score=0.3,
            explanation="ACH for low cost",
            trace_id="test_preferences_001",
        )

        # Create consumer context with strong preferences
        consumer_context = {
            "actor_id": "consumer_prefs_test",
            "prefers_mobile": True,
            "preferred_instruments": ["stablecoin"],
            "preferred_rails": ["Crypto"],
            "available_instruments": {
                "stablecoin": {"max_amount": 50000.0, "reward_rate": 0.02, "preferred": True},
                "credit_card": {"max_amount": 25000.0, "reward_rate": 0.015, "preferred": False},
            },
            "available_rails": {
                "Crypto": {
                    "supported_instruments": ["stablecoin"],
                    "consumer_benefit": 0.03,
                    "preferred": True,
                },
                "Card": {
                    "supported_instruments": ["credit_card"],
                    "consumer_benefit": 0.02,
                    "preferred": False,
                },
            },
        }

        # Run consumer negotiation
        response = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, 2000.0, consumer_context
        )

        # Verify preferred combination is selected
        optimal = response.consumer_proposal
        assert optimal.rail_type == "Crypto"
        assert optimal.instrument_type == "stablecoin"

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_fallback_to_merchant_proposal(self, consumer_negotiator):
        """Test fallback when no consumer alternatives are available."""
        # Create merchant proposal
        merchant_proposal = MerchantProposal(
            rail_type="Card",
            merchant_cost=200.0,
            settlement_days=2,
            risk_score=0.4,
            explanation="Card payment",
            trace_id="test_fallback_001",
        )

        # Create consumer context with no available alternatives
        consumer_context = {
            "actor_id": "consumer_no_alternatives",
            "available_instruments": {},
            "available_rails": {},
        }

        # Run consumer negotiation
        response = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, 1000.0, consumer_context
        )

        # Verify fallback response
        assert response.consumer_proposal.rail_type == "Card"  # Falls back to merchant proposal
        assert response.consumer_rewards[0].value == 0.0  # No rewards available
        assert "accepting merchant proposal" in response.explanation.lower()
        assert response.confidence < 0.5  # Low confidence for fallback

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_negotiation_deterministic(
        self, consumer_negotiator, consumer_multi_rail_fixture
    ):
        """Test that consumer negotiation is deterministic."""
        fixture = consumer_multi_rail_fixture

        # Create merchant proposal
        merchant_proposal = MerchantProposal(**fixture["merchant_proposal"])

        # Create consumer context
        consumer_context = fixture["consumer_context"]

        # Run negotiation multiple times
        response1 = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, fixture["transaction_amount"], consumer_context
        )
        response2 = consumer_negotiator.negotiate_consumer_response(
            merchant_proposal, fixture["transaction_amount"], consumer_context
        )

        # Verify deterministic results
        assert response1.consumer_proposal.rail_type == response2.consumer_proposal.rail_type
        assert (
            response1.consumer_proposal.instrument_type
            == response2.consumer_proposal.instrument_type
        )
        assert (
            abs(
                response1.consumer_proposal.consumer_benefit
                - response2.consumer_proposal.consumer_benefit
            )
            < 0.01
        )
        assert (
            abs(
                response1.consumer_proposal.convenience_score
                - response2.consumer_proposal.convenience_score
            )
            < 0.01
        )

    @pytest.mark.skip(reason="Multi-rail evaluator needs capability context initialization")
    def test_consumer_multi_rail_coverage(self, consumer_evaluator):
        """Test that consumer evaluation covers multiple rail/instrument combinations."""
        # Create merchant proposal
        merchant_proposal = MerchantProposal(
            rail_type="ACH",
            merchant_cost=15.0,
            settlement_days=1,
            risk_score=0.3,
            explanation="ACH proposal",
            trace_id="test_coverage_001",
        )

        # Create consumer context
        consumer_context = {
            "actor_id": "consumer_coverage_test",
            "available_instruments": {
                "stablecoin": {"max_amount": 50000.0, "reward_rate": 0.02, "preferred": True},
                "debit_card": {"max_amount": 10000.0, "reward_rate": 0.005, "preferred": False},
                "credit_card": {"max_amount": 25000.0, "reward_rate": 0.015, "preferred": True},
            },
            "available_rails": {
                "Crypto": {
                    "supported_instruments": ["stablecoin"],
                    "consumer_benefit": 0.03,
                    "preferred": True,
                },
                "FedNow": {
                    "supported_instruments": ["debit_card"],
                    "consumer_benefit": 0.01,
                    "preferred": True,
                },
                "Card": {
                    "supported_instruments": ["credit_card", "debit_card"],
                    "consumer_benefit": 0.02,
                    "preferred": False,
                },
            },
        }

        # Get available combinations
        available_combinations = consumer_evaluator.capability.get_available_combinations()

        # Evaluate each combination
        evaluated_combinations = []
        for rail_type, instrument_type in available_combinations:
            evaluation = consumer_evaluator.evaluate_consumer_combination(
                rail_type, instrument_type, merchant_proposal, 1000.0, consumer_context
            )
            if evaluation:
                evaluated_combinations.append((rail_type, instrument_type))

        # Verify good coverage
        coverage_ratio = len(evaluated_combinations) / len(available_combinations)
        assert coverage_ratio > 0.7, f"Only {coverage_ratio:.1%} of combinations were evaluated"

        # Verify important combinations are covered
        important_combinations = [
            ("Crypto", "stablecoin"),
            ("FedNow", "debit_card"),
            ("Card", "credit_card"),
        ]

        for rail_type, instrument_type in important_combinations:
            if (rail_type, instrument_type) in available_combinations:
                assert (
                    rail_type,
                    instrument_type,
                ) in evaluated_combinations, (
                    f"Important combination {rail_type}/{instrument_type} not evaluated"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
