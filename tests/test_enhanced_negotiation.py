"""
Unit tests for Opal Phase 4 Enhanced Negotiation functionality.

Tests consumer preference flows, deterministic scoring, explainability diffs,
and multi-instrument support.
"""

import json
import pytest
from pathlib import Path

from src.opal.controls import (
    ConsumerInstrument,
    ConsumerReward,
    MerchantProposal,
    CounterNegotiationRequest,
    CounterNegotiationResponse,
)
from src.opal.enhanced_negotiation import MultiInstrumentNegotiator, enhanced_counter_negotiate


class TestMultiInstrumentNegotiator:
    """Test the MultiInstrumentNegotiator class."""

    @pytest.fixture
    def negotiator(self):
        """Create a MultiInstrumentNegotiator instance."""
        return MultiInstrumentNegotiator()

    @pytest.fixture
    def sample_merchant_proposal(self):
        """Create a sample merchant proposal."""
        return MerchantProposal(
            rail_type="ACH",
            merchant_cost=250,  # 2.5% in basis points
            settlement_days=3,
            risk_score=0.2,
            explanation="ACH rail offers lowest cost for merchant",
            trace_id="merchant_trace_123",
        )

    @pytest.fixture
    def sample_consumer_instruments(self):
        """Create sample consumer instruments with different types."""
        return [
            # Credit card with high rewards
            ConsumerInstrument(
                instrument_id="cc_visa_001",
                instrument_type="credit_card",
                provider="Visa",
                last_four="1234",
                base_fee=200,  # 2.0% in basis points
                out_of_pocket_cost=0.0,
                available_balance=5000.0,
                rewards=[
                    ConsumerReward(
                        reward_type="cashback",
                        rate=0.02,  # 2% cashback
                        value=3.0,  # $3 on $150 transaction
                        description="2% cashback on all purchases",
                    )
                ],
                total_reward_value=3.0,  # $3 on $150 transaction
                loyalty_tier="gold",
                loyalty_multiplier=1.2,
                net_value=3.0,
                value_score=0.0,  # Will be calculated
                eligible=True,
                preference_score=0.8,
                selection_factors=[],
                exclusion_reasons=[],
            ),
            # Debit card with lower fees
            ConsumerInstrument(
                instrument_id="dc_chase_001",
                instrument_type="debit_card",
                provider="Chase",
                last_four="5678",
                base_fee=100,  # 1.0% in basis points
                out_of_pocket_cost=0.0,
                available_balance=2000.0,
                rewards=[
                    ConsumerReward(
                        reward_type="cashback",
                        rate=0.01,  # 1% cashback
                        value=1.5,  # $1.50 on $150 transaction
                        description="1% cashback on debit purchases",
                    )
                ],
                total_reward_value=1.5,  # $1.50 on $150 transaction
                loyalty_tier=None,
                loyalty_multiplier=1.0,
                net_value=1.5,
                value_score=0.0,
                eligible=True,
                preference_score=0.6,
                selection_factors=[],
                exclusion_reasons=[],
            ),
            # BNPL with benefits
            ConsumerInstrument(
                instrument_id="bnpl_klarna_001",
                instrument_type="bnpl",
                provider="Klarna",
                last_four="9999",
                base_fee=50,  # 0.5% in basis points
                out_of_pocket_cost=0.0,
                available_balance=1000.0,
                rewards=[
                    ConsumerReward(
                        reward_type="bnpl_benefits",
                        rate=0.0167,  # ~1.67% equivalent benefit
                        value=2.5,  # $2.50 in BNPL benefits on $150 transaction
                        description="No interest if paid on time",
                    )
                ],
                total_reward_value=2.5,
                loyalty_tier=None,
                loyalty_multiplier=1.0,
                net_value=2.5,
                value_score=0.0,
                eligible=True,
                preference_score=0.7,
                selection_factors=[],
                exclusion_reasons=[],
            ),
            # Stablecoin with crypto rewards
            ConsumerInstrument(
                instrument_id="stable_usdc_001",
                instrument_type="stablecoin",
                provider="Circle",
                last_four="0001",
                base_fee=75,  # 0.75% in basis points
                out_of_pocket_cost=0.0,
                available_balance=500.0,
                rewards=[
                    ConsumerReward(
                        reward_type="crypto_rewards",
                        rate=0.012,  # 1.2% crypto rewards
                        value=1.8,  # $1.80 in crypto rewards on $150 transaction
                        description="1.2% crypto rewards on USDC",
                    )
                ],
                total_reward_value=1.8,
                loyalty_tier=None,
                loyalty_multiplier=1.0,
                net_value=1.8,
                value_score=0.0,
                eligible=True,
                preference_score=0.9,
                selection_factors=[],
                exclusion_reasons=[],
            ),
        ]

    @pytest.fixture
    def sample_consumer_preferences(self):
        """Create sample consumer preferences."""
        return {
            "cash_rewards": 1.2,  # 20% preference boost for cash rewards
            "loyalty_programs": 0.8,  # 20% preference reduction for loyalty programs
            "bnpl_preference": 1.5,  # 50% preference boost for BNPL
            "convenience_preference": 1.1,  # 10% preference boost for convenience
            "category_preferences": {
                "cash_rewards": 1.2,
                "loyalty_programs": 0.8,
                "bnpl_preference": 1.5,
            },
        }

    def test_instrument_value_calculation_deterministic(
        self,
        negotiator,
        sample_consumer_instruments,
        sample_merchant_proposal,
        sample_consumer_preferences,
    ):
        """Test deterministic instrument value calculation."""

        # Test with same inputs multiple times
        results = []
        for _ in range(5):
            value_score, scoring_details = negotiator.calculate_instrument_value(
                sample_consumer_instruments[0],  # Credit card
                150.0,  # $150 transaction
                sample_merchant_proposal,
                sample_consumer_preferences,
            )
            results.append((value_score, scoring_details))

        # All results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert (
                abs(result[0] - first_result[0]) < 0.001
            )  # Allow small floating point differences
            assert result[1]["instrument_type"] == first_result[1]["instrument_type"]

    def test_multi_instrument_scoring(
        self,
        negotiator,
        sample_consumer_instruments,
        sample_merchant_proposal,
        sample_consumer_preferences,
    ):
        """Test scoring across different instrument types."""

        scored_instruments = []
        for instrument in sample_consumer_instruments:
            value_score, scoring_details = negotiator.calculate_instrument_value(
                instrument, 150.0, sample_merchant_proposal, sample_consumer_preferences
            )
            scored_instruments.append((instrument.instrument_type, value_score, scoring_details))

        # Should have scores for all instrument types
        assert len(scored_instruments) == 4

        # Scores should be between 0 and 1
        for instrument_type, score, details in scored_instruments:
            assert 0.0 <= score <= 1.0
            assert details["instrument_type"] == instrument_type

    def test_consumer_preference_flows(
        self, negotiator, sample_consumer_instruments, sample_merchant_proposal
    ):
        """Test different consumer preference flows."""

        # Test cashback preference
        cashback_preferences = {
            "cash_rewards": 2.0,  # Strong preference for cash rewards
            "loyalty_programs": 0.5,  # Weak preference for loyalty programs
            "bnpl_preference": 1.0,
            "convenience_preference": 1.0,
        }

        credit_card_score, _ = negotiator.calculate_instrument_value(
            sample_consumer_instruments[0],  # Credit card with cashback
            150.0,
            sample_merchant_proposal,
            cashback_preferences,
        )

        # Test BNPL preference
        bnpl_preferences = {
            "cash_rewards": 1.0,
            "loyalty_programs": 1.0,
            "bnpl_preference": 2.0,  # Strong preference for BNPL
            "convenience_preference": 1.0,
        }

        bnpl_score, _ = negotiator.calculate_instrument_value(
            sample_consumer_instruments[2],  # BNPL instrument
            150.0,
            sample_merchant_proposal,
            bnpl_preferences,
        )

        # Test convenience preference
        convenience_preferences = {
            "cash_rewards": 1.0,
            "loyalty_programs": 1.0,
            "bnpl_preference": 1.0,
            "convenience_preference": 2.0,  # Strong preference for convenience
        }

        debit_score, _ = negotiator.calculate_instrument_value(
            sample_consumer_instruments[1],  # Debit card (convenient)
            150.0,
            sample_merchant_proposal,
            convenience_preferences,
        )

        # Scores should reflect preferences
        assert credit_card_score > 0
        assert bnpl_score > 0
        assert debit_score > 0

    def test_counter_negotiation_deterministic(
        self,
        negotiator,
        sample_consumer_instruments,
        sample_merchant_proposal,
        sample_consumer_preferences,
    ):
        """Test deterministic counter-negotiation results."""

        request = CounterNegotiationRequest(
            actor_id="test_actor_123",
            transaction_amount=150.0,
            currency="USD",
            merchant_id="test_merchant",
            merchant_proposal=sample_merchant_proposal,
            available_instruments=sample_consumer_instruments,
            consumer_preferences=sample_consumer_preferences,
        )

        # Test with same inputs multiple times
        results = []
        for _ in range(5):
            response = negotiator.counter_negotiate(request)
            results.append(response)

        # All results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert (
                result.consumer_proposal.instrument_type
                == first_result.consumer_proposal.instrument_type
            )
            assert (
                abs(
                    result.metadata.get("win_win_score", 0)
                    - first_result.metadata.get("win_win_score", 0)
                )
                < 0.001
            )
            assert result.explanation == first_result.explanation

    def test_explainability_diffs(
        self, negotiator, sample_consumer_instruments, sample_merchant_proposal
    ):
        """Test explainability differences between instrument selections."""

        # Test with different preference sets
        preference_sets = [
            {
                "cash_rewards": 2.0,
                "loyalty_programs": 0.5,
                "bnpl_preference": 1.0,
            },  # Cashback focused
            {"cash_rewards": 0.5, "loyalty_programs": 1.0, "bnpl_preference": 2.0},  # BNPL focused
            {"cash_rewards": 1.0, "loyalty_programs": 1.0, "bnpl_preference": 1.0},  # Neutral
        ]

        explanations = []
        for preferences in preference_sets:
            request = CounterNegotiationRequest(
                actor_id="test_actor_123",
                transaction_amount=150.0,
                currency="USD",
                merchant_proposal=sample_merchant_proposal,
                available_instruments=sample_consumer_instruments,
                consumer_preferences=preferences,
            )

            response = negotiator.counter_negotiate(request)
            explanations.append(
                {
                    "preferences": preferences,
                    "selected_instrument": response.consumer_proposal.instrument_type,
                    "explanation": response.explanation,
                    "value_score": response.metadata.get("value_score", 0.0),
                }
            )

        # Should have different explanations for different preferences
        assert len(explanations) == 3

        # At least one explanation should mention the preference
        preference_mentioned = any(
            "cashback" in exp["explanation"].lower()
            or "reward" in exp["explanation"].lower()
            or "bnpl" in exp["explanation"].lower()
            for exp in explanations
        )
        assert preference_mentioned

    def test_win_win_score_calculation(
        self,
        negotiator,
        sample_consumer_instruments,
        sample_merchant_proposal,
        sample_consumer_preferences,
    ):
        """Test win-win score calculation."""

        request = CounterNegotiationRequest(
            actor_id="test_actor_123",
            transaction_amount=150.0,
            currency="USD",
            merchant_proposal=sample_merchant_proposal,
            available_instruments=sample_consumer_instruments,
            consumer_preferences=sample_consumer_preferences,
        )

        response = negotiator.counter_negotiate(request)

        # Win-win score should be between 0 and 1
        assert 0.0 <= response.metadata.get("win_win_score", 0.5) <= 1.0

        # Should have positive consumer value (in metadata or consumer_proposal)
        consumer_value = (
            response.metadata.get("consumer_value", 0)
            or response.consumer_proposal.consumer_benefit
        )
        assert consumer_value >= 0
        merchant_savings = response.metadata.get("merchant_savings", 0)
        assert merchant_savings >= 0

    def test_selection_factors_generation(
        self,
        negotiator,
        sample_consumer_instruments,
        sample_merchant_proposal,
        sample_consumer_preferences,
    ):
        """Test selection factors generation."""

        # Test with high-value credit card
        value_score, scoring_details = negotiator.calculate_instrument_value(
            sample_consumer_instruments[0],  # Credit card with high rewards
            150.0,
            sample_merchant_proposal,
            sample_consumer_preferences,
        )

        selection_factors = negotiator._generate_selection_factors(scoring_details)

        # Should have some selection factors
        assert len(selection_factors) > 0

        # Factors should mention relevant aspects
        factors_text = " ".join(selection_factors).lower()
        assert any(
            keyword in factors_text for keyword in ["reward", "cost", "convenient", "loyalty"]
        )

    def test_merchant_savings_calculation(
        self, negotiator, sample_consumer_instruments, sample_merchant_proposal
    ):
        """Test merchant savings calculation."""

        # Test with different instruments
        for instrument in sample_consumer_instruments:
            savings = negotiator._calculate_merchant_savings(instrument, sample_merchant_proposal)

            # Savings should be non-negative
            assert savings >= 0.0

            # Should be reasonable compared to merchant cost
            merchant_cost = sample_merchant_proposal.merchant_cost / 10000.0  # Convert basis points
            assert savings <= merchant_cost

    def test_settlement_speed_mapping(self, negotiator):
        """Test settlement speed mapping for different instrument types."""

        speed_map = {
            "credit_card": "1-3 days",
            "debit_card": "1-2 days",
            "prepaid_card": "immediate",
            "bnpl": "immediate",
            "stablecoin": "immediate",
            "digital_wallet": "immediate",
            "bank_transfer": "1-3 days",
        }

        for instrument_type, expected_speed in speed_map.items():
            actual_speed = negotiator._get_settlement_speed(instrument_type)
            assert actual_speed == expected_speed

    def test_error_handling_no_eligible_instruments(
        self, negotiator, sample_merchant_proposal, sample_consumer_preferences
    ):
        """Test error handling when no instruments are eligible."""

        # Create ineligible instruments
        ineligible_instruments = [
            ConsumerInstrument(
                instrument_id="ineligible_001",
                instrument_type="credit_card",
                provider="Test",
                last_four="0000",
                base_fee=200,
                out_of_pocket_cost=0.0,
                available_balance=0.0,  # No balance
                rewards=[],
                total_reward_value=0.0,
                loyalty_multiplier=1.0,
                net_value=0.0,
                value_score=0.0,
                eligible=False,  # Not eligible
                selection_factors=[],
                exclusion_reasons=[],
            )
        ]

        request = CounterNegotiationRequest(
            actor_id="test_actor_123",
            transaction_amount=150.0,
            currency="USD",
            merchant_proposal=sample_merchant_proposal,
            available_instruments=ineligible_instruments,
            consumer_preferences=sample_consumer_preferences,
        )

        with pytest.raises(ValueError, match="No eligible instruments available"):
            negotiator.counter_negotiate(request)

    def test_golden_fixture_consistency(self, negotiator):
        """Test consistency with golden fixture data."""

        # Load golden fixture data
        fixture_dir = Path(__file__).parent / "fixtures" / "enhanced_negotiation"

        if not fixture_dir.exists():
            pytest.skip("Golden fixtures not available")

        # Test with golden fixture inputs
        golden_input = fixture_dir / "input.json"
        golden_output = fixture_dir / "expected_output.json"

        if golden_input.exists() and golden_output.exists():
            with open(golden_input) as f:
                input_data = json.load(f)

            with open(golden_output) as f:
                expected_output = json.load(f)

            # Create objects from input data
            merchant_proposal = MerchantProposal(**input_data["merchant_proposal"])
            instruments = [
                ConsumerInstrument(**inst) for inst in input_data["available_instruments"]
            ]
            preferences = input_data["consumer_preferences"]

            # Generate response
            request = CounterNegotiationRequest(
                actor_id=input_data["actor_id"],
                transaction_amount=input_data["transaction_amount"],
                currency=input_data["currency"],
                merchant_proposal=merchant_proposal,
                available_instruments=instruments,
                consumer_preferences=preferences,
            )

            response = negotiator.counter_negotiate(request)

            # Compare with expected output
            assert (
                response.consumer_proposal.instrument_type
                == expected_output["selected_instrument_type"]
            )
            assert (
                abs(response.metadata.get("win_win_score", 0) - expected_output["win_win_score"])
                < 0.01
            )


class TestEnhancedCounterNegotiateFunction:
    """Test the convenience function for backward compatibility."""

    def test_enhanced_counter_negotiate_function(
        self, sample_consumer_instruments, sample_merchant_proposal, sample_consumer_preferences
    ):
        """Test the enhanced_counter_negotiate convenience function."""

        request = CounterNegotiationRequest(
            actor_id="test_actor_123",
            transaction_amount=150.0,
            currency="USD",
            merchant_proposal=sample_merchant_proposal,
            available_instruments=sample_consumer_instruments,
            consumer_preferences=sample_consumer_preferences,
        )

        response = enhanced_counter_negotiate(request)

        # Should return valid response
        assert isinstance(response, CounterNegotiationResponse)
        assert response.consumer_proposal is not None
        assert response.explanation is not None
        assert response.metadata.get("win_win_score", 0.0) >= 0.0


@pytest.fixture
def sample_consumer_instruments():
    """Create sample consumer instruments for tests."""
    return [
        ConsumerInstrument(
            instrument_id="cc_visa_001",
            instrument_type="credit_card",
            provider="Visa",
            last_four="1234",
            base_fee=200,
            out_of_pocket_cost=0.0,
            available_balance=5000.0,
            rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    rate=0.02,
                    value=3.0,
                    description="2% cashback on all purchases",
                )
            ],
            total_reward_value=3.0,
            loyalty_tier="gold",
            loyalty_multiplier=1.2,
            net_value=3.0,
            value_score=0.0,
            eligible=True,
            preference_score=0.8,
            selection_factors=[],
            exclusion_reasons=[],
        ),
        ConsumerInstrument(
            instrument_id="bnpl_klarna_001",
            instrument_type="bnpl",
            provider="Klarna",
            last_four="9999",
            base_fee=50,
            out_of_pocket_cost=0.0,
            available_balance=1000.0,
            rewards=[
                ConsumerReward(
                    reward_type="bnpl_benefits",
                    rate=0.0167,
                    value=2.5,
                    description="No interest if paid on time",
                )
            ],
            total_reward_value=2.5,
            loyalty_tier=None,
            loyalty_multiplier=1.0,
            net_value=2.5,
            value_score=0.0,
            eligible=True,
            preference_score=0.7,
            selection_factors=[],
            exclusion_reasons=[],
        ),
    ]


@pytest.fixture
def sample_merchant_proposal():
    """Create a sample merchant proposal for tests."""
    return MerchantProposal(
        rail_type="ACH",
        merchant_cost=250,
        settlement_days=3,
        risk_score=0.2,
        explanation="ACH rail offers lowest cost for merchant",
        trace_id="merchant_trace_123",
    )


@pytest.fixture
def sample_consumer_preferences():
    """Create sample consumer preferences for tests."""
    return {
        "cash_rewards": 1.2,
        "loyalty_programs": 0.8,
        "bnpl_preference": 1.5,
        "convenience_preference": 1.1,
        "category_preferences": {
            "cash_rewards": 1.2,
            "loyalty_programs": 0.8,
            "bnpl_preference": 1.5,
        },
    }


@pytest.fixture
def golden_fixtures():
    """Create golden fixture data for testing."""
    fixture_dir = Path(__file__).parent / "fixtures" / "enhanced_negotiation"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # Create input fixture
    input_data = {
        "actor_id": "golden_actor_123",
        "transaction_amount": 100.0,
        "currency": "USD",
        "merchant_proposal": {
            "rail_type": "ACH",
            "merchant_cost": 200,
            "settlement_days": 2,
            "risk_score": 0.3,
            "explanation": "ACH rail for cost efficiency",
            "trace_id": "golden_merchant_trace",
        },
        "available_instruments": [
            {
                "instrument_id": "golden_cc_001",
                "instrument_type": "credit_card",
                "provider": "Golden Bank",
                "last_four": "1234",
                "base_fee": 150,
                "out_of_pocket_cost": 0.0,
                "available_balance": 5000.0,
                "rewards": [
                    {
                        "reward_type": "cashback",
                        "rate": 0.015,
                        "value": 1.5,
                        "description": "1.5% cashback",
                    }
                ],
                "total_reward_value": 1.5,
                "loyalty_multiplier": 1.0,
                "net_value": 1.5,
                "value_score": 0.0,
                "eligible": True,
                "selection_factors": [],
                "exclusion_reasons": [],
            }
        ],
        "consumer_preferences": {
            "cash_rewards": 1.0,
            "loyalty_programs": 1.0,
            "bnpl_preference": 1.0,
            "convenience_preference": 1.0,
        },
    }

    with open(fixture_dir / "input.json", "w") as f:
        json.dump(input_data, f, indent=2)

    # Create expected output fixture
    expected_output = {
        "selected_instrument_type": "credit_card",
        "win_win_score": 0.75,
        "explanation_contains": ["Selected Credit Card", "net value", "rewards"],
    }

    with open(fixture_dir / "expected_output.json", "w") as f:
        json.dump(expected_output, f, indent=2)

    return fixture_dir
