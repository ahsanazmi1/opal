"""
Enhanced Consumer Counter-Negotiation Logic for Opal Phase 4

This module implements advanced consumer-side negotiation logic with multi-instrument
support, sophisticated value scoring, and comprehensive explanation generation.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from uuid import uuid4
import math

from .controls import (
    ConsumerInstrument,
    ConsumerReward,
    ConsumerProposal,
    MerchantProposal,
    CounterNegotiationRequest,
    CounterNegotiationResponse,
    InstrumentType,
)
from .events import emit_consumer_explanation_event
from .llm.explain import explain_consumer_instrument_choice, is_consumer_llm_configured

# Set up logging
logger = logging.getLogger(__name__)


class MultiInstrumentNegotiator:
    """Enhanced negotiator supporting multiple instrument types and advanced value scoring."""

    def __init__(self):
        """Initialize the multi-instrument negotiator."""
        self.instrument_weights = {
            "credit_card": {"rewards": 0.4, "cost": 0.3, "convenience": 0.3},
            "debit_card": {"rewards": 0.2, "cost": 0.5, "convenience": 0.3},
            "prepaid_card": {"rewards": 0.1, "cost": 0.4, "convenience": 0.5},
            "bnpl": {"rewards": 0.3, "cost": 0.2, "convenience": 0.5},
            "stablecoin": {"rewards": 0.5, "cost": 0.3, "convenience": 0.2},
            "digital_wallet": {"rewards": 0.3, "cost": 0.3, "convenience": 0.4},
            "bank_transfer": {"rewards": 0.1, "cost": 0.6, "convenience": 0.3},
        }

        self.reward_multipliers = {
            "cashback": 1.0,
            "points": 0.8,
            "miles": 0.9,
            "loyalty_points": 0.7,
            "discount": 1.2,
            "crypto_rewards": 1.1,
            "bnpl_benefits": 1.3,
            "cash_advance": 0.9,
        }

    def generate_trace_id(self) -> str:
        """Generate a unique trace ID for negotiation."""
        return f"opal_enhanced_{uuid4().hex[:16]}"

    def calculate_instrument_value(
        self,
        instrument: ConsumerInstrument,
        amount: float,
        merchant_proposal: MerchantProposal,
        preferences: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate comprehensive value score for an instrument.

        Args:
            instrument: Consumer instrument to evaluate
            amount: Transaction amount
            merchant_proposal: Merchant's rail proposal
            preferences: Consumer preferences

        Returns:
            Tuple of (value_score, scoring_details)
        """
        instrument_type = instrument.instrument_type
        weights = self.instrument_weights.get(
            instrument_type, {"rewards": 0.33, "cost": 0.33, "convenience": 0.34}
        )

        # Calculate reward value
        reward_value = self._calculate_reward_value(instrument, amount, preferences)

        # Calculate cost efficiency
        cost_score = self._calculate_cost_efficiency(instrument, amount, merchant_proposal)

        # Calculate convenience score
        convenience_score = self._calculate_convenience_score(instrument, preferences)

        # Apply loyalty multiplier
        loyalty_boost = instrument.loyalty_multiplier

        # Calculate weighted value score
        value_score = (
            reward_value * weights["rewards"]
            + cost_score * weights["cost"]
            + convenience_score * weights["convenience"]
        ) * loyalty_boost

        # Normalize to 0-1 range
        value_score = max(0.0, min(1.0, value_score))

        scoring_details = {
            "reward_value": reward_value,
            "cost_score": cost_score,
            "convenience_score": convenience_score,
            "loyalty_boost": loyalty_boost,
            "weights": weights,
            "instrument_type": instrument_type,
        }

        return value_score, scoring_details

    def _calculate_reward_value(
        self, instrument: ConsumerInstrument, amount: float, preferences: Dict[str, Any]
    ) -> float:
        """Calculate reward value for an instrument."""
        if not instrument.rewards:
            return 0.0

        total_reward_value = 0.0
        for reward in instrument.rewards:
            base_value = reward.reward_value
            multiplier = self.reward_multipliers.get(reward.reward_type, 1.0)

            # Apply category bonuses
            category_bonus = self._get_category_bonus(reward, preferences)

            # Apply consumer preference multiplier
            preference_multiplier = preferences.get(f"{reward.reward_type}_preference", 1.0)

            adjusted_value = base_value * multiplier * category_bonus * preference_multiplier
            total_reward_value += adjusted_value

        # Normalize by transaction amount
        return min(1.0, total_reward_value / max(amount * 0.1, 1.0))

    def _get_category_bonus(self, reward: ConsumerReward, preferences: Dict[str, Any]) -> float:
        """Get category-specific bonus multiplier."""
        category_preferences = preferences.get("category_preferences", {})

        # Check for category-specific bonuses
        if reward.reward_type in ["cashback", "crypto_rewards"]:
            return category_preferences.get("cash_rewards", 1.0)
        elif reward.reward_type in ["points", "loyalty_points"]:
            return category_preferences.get("loyalty_programs", 1.0)
        elif reward.reward_type == "bnpl_benefits":
            return category_preferences.get("bnpl_preference", 1.0)

        return 1.0

    def _calculate_cost_efficiency(
        self, instrument: ConsumerInstrument, amount: float, merchant_proposal: MerchantProposal
    ) -> float:
        """Calculate cost efficiency score."""
        # Compare consumer cost vs merchant savings potential
        consumer_cost = instrument.out_of_pocket_cost

        # Calculate potential merchant savings if they accept this instrument
        merchant_savings = self._calculate_merchant_savings(instrument, merchant_proposal)

        # Cost efficiency = merchant_savings / (consumer_cost + small_constant)
        cost_efficiency = merchant_savings / (consumer_cost + 0.01)

        # Normalize to 0-1 range using sigmoid function
        return 1.0 / (1.0 + math.exp(-cost_efficiency))

    def _calculate_merchant_savings(
        self, instrument: ConsumerInstrument, merchant_proposal: MerchantProposal
    ) -> float:
        """Calculate potential merchant savings from accepting this instrument."""
        # Base merchant cost from proposal
        merchant_cost = merchant_proposal.merchant_cost / 10000.0  # Convert basis points

        # Consumer instrument cost (usually lower for consumer instruments)
        instrument_cost = instrument.base_fee / 10000.0

        # Potential savings
        savings = merchant_cost - instrument_cost

        return max(0.0, savings)

    def _calculate_convenience_score(
        self, instrument: ConsumerInstrument, preferences: Dict[str, Any]
    ) -> float:
        """Calculate convenience score for an instrument."""
        convenience_factors = {
            "credit_card": 0.9,
            "debit_card": 0.8,
            "prepaid_card": 0.7,
            "bnpl": 0.6,
            "stablecoin": 0.5,
            "digital_wallet": 0.8,
            "bank_transfer": 0.4,
        }

        base_score = convenience_factors.get(instrument.instrument_type, 0.5)

        # Apply consumer preferences
        convenience_preference = preferences.get("convenience_preference", 1.0)

        return min(1.0, base_score * convenience_preference)

    def counter_negotiate(self, request: CounterNegotiationRequest) -> CounterNegotiationResponse:
        """
        Perform enhanced counter-negotiation with multi-instrument support.

        Args:
            request: Counter-negotiation request

        Returns:
            Enhanced counter-negotiation response
        """
        logger.info(f"Starting enhanced counter-negotiation for actor {request.actor_id}")

        # Score all available instruments
        scored_instruments = []
        for instrument in request.available_instruments:
            if instrument.eligible:
                value_score, scoring_details = self.calculate_instrument_value(
                    instrument,
                    request.transaction_amount,
                    request.merchant_proposal,
                    request.consumer_preferences,
                )

                # Update instrument with calculated score
                instrument.value_score = value_score
                instrument.selection_factors = self._generate_selection_factors(scoring_details)
                scored_instruments.append((instrument, scoring_details))

        # Sort by value score (descending)
        scored_instruments.sort(key=lambda x: x[0].value_score, reverse=True)

        if not scored_instruments:
            raise ValueError("No eligible instruments available for negotiation")

        # Select optimal instrument
        optimal_instrument, optimal_details = scored_instruments[0]

        # Generate counter-proposal
        _ = self._generate_counter_proposal(
            optimal_instrument, request.merchant_proposal, optimal_details
        )

        # Calculate win-win metrics
        merchant_savings = self._calculate_merchant_savings(
            optimal_instrument, request.merchant_proposal
        )
        consumer_value = optimal_instrument.net_value
        win_win_score = self._calculate_win_win_score(merchant_savings, consumer_value)

        # Generate explanation
        explanation = self._generate_explanation(
            optimal_instrument,
            scored_instruments[1:],  # Alternatives
            request.merchant_proposal,
            optimal_details,
        )

        # Create response
        response = CounterNegotiationResponse(
            actor_id=request.actor_id,
            trace_id=self.generate_trace_id(),
            consumer_proposal=ConsumerProposal(
                rail_type=request.merchant_proposal.rail_type,
                instrument_type=optimal_instrument.instrument_type,
                consumer_benefit=optimal_instrument.net_value,
                convenience_score=optimal_instrument.preference_score,
                explanation=f"Selected {optimal_instrument.instrument_type} from {optimal_instrument.provider} for optimal value",
            ),
            consumer_rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    rate=0.02,  # 2% cashback rate
                    value=optimal_instrument.total_reward_value,
                    description=f"Loyalty rewards from {optimal_instrument.provider}",
                )
            ],
            alternatives=[
                {
                    "instrument_type": inst.instrument_type,
                    "provider": inst.provider,
                    "net_value": inst.net_value,
                    "reason": "Lower value score",
                }
                for inst, _ in scored_instruments[1:]
            ],
            explanation=explanation,
            confidence=0.85,  # Default confidence score
            metadata={
                "total_instruments_evaluated": len(scored_instruments),
                "scoring_method": "enhanced_multi_instrument",
                "optimal_details": optimal_details,
                "consumer_preferences": request.consumer_preferences,
                "timestamp": datetime.now().isoformat(),
                "merchant_savings": merchant_savings,
                "consumer_value": consumer_value,
                "win_win_score": win_win_score,
            },
        )

        # Emit explanation event
        emit_consumer_explanation_event(response, request.actor_id)

        logger.info(
            f"Enhanced counter-negotiation completed. Selected {optimal_instrument.instrument_type} with score {optimal_instrument.value_score:.3f}"
        )

        return response

    def _generate_selection_factors(self, scoring_details: Dict[str, Any]) -> List[str]:
        """Generate human-readable selection factors."""
        factors = []

        if scoring_details["reward_value"] > 0.5:
            factors.append(f"High reward value ({scoring_details['reward_value']:.2f})")

        if scoring_details["cost_score"] > 0.5:
            factors.append(f"Cost efficient ({scoring_details['cost_score']:.2f})")

        if scoring_details["convenience_score"] > 0.5:
            factors.append(f"Convenient ({scoring_details['convenience_score']:.2f})")

        if scoring_details["loyalty_boost"] > 1.0:
            factors.append(f"Loyalty tier bonus ({scoring_details['loyalty_boost']:.1f}x)")

        instrument_type = scoring_details["instrument_type"]
        if instrument_type in ["stablecoin", "bnpl"]:
            factors.append(f"Modern payment method ({instrument_type})")

        return factors

    def _generate_counter_proposal(
        self,
        instrument: ConsumerInstrument,
        merchant_proposal: MerchantProposal,
        scoring_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate counter-proposal for merchant."""
        return {
            "proposed_instrument": {
                "type": instrument.instrument_type,
                "provider": instrument.provider,
                "last_four": instrument.last_four,
            },
            "consumer_benefits": {
                "net_value": instrument.net_value,
                "total_rewards": instrument.total_reward_value,
                "convenience_score": scoring_details["convenience_score"],
            },
            "merchant_benefits": {
                "potential_savings": self._calculate_merchant_savings(
                    instrument, merchant_proposal
                ),
                "risk_profile": "consumer_managed",
                "settlement_speed": self._get_settlement_speed(instrument.instrument_type),
            },
            "win_win_metrics": {
                "consumer_value_score": instrument.value_score,
                "merchant_savings_score": scoring_details["cost_score"],
                "overall_efficiency": (instrument.value_score + scoring_details["cost_score"]) / 2,
            },
        }

    def _get_settlement_speed(self, instrument_type: InstrumentType) -> str:
        """Get settlement speed for instrument type."""
        speed_map = {
            "credit_card": "1-3 days",
            "debit_card": "1-2 days",
            "prepaid_card": "immediate",
            "bnpl": "immediate",
            "stablecoin": "immediate",
            "digital_wallet": "immediate",
            "bank_transfer": "1-3 days",
        }
        return speed_map.get(instrument_type, "1-3 days")

    def _calculate_win_win_score(self, merchant_savings: float, consumer_value: float) -> float:
        """Calculate overall win-win score."""
        # Normalize both values to 0-1 range
        norm_merchant = min(1.0, merchant_savings / 10.0)  # Assume max $10 savings
        norm_consumer = min(1.0, max(0.0, consumer_value / 50.0))  # Assume max $50 value

        # Win-win score is harmonic mean of both benefits
        if norm_merchant == 0 and norm_consumer == 0:
            return 0.0

        return (2 * norm_merchant * norm_consumer) / (norm_merchant + norm_consumer)

    def _generate_explanation(
        self,
        selected_instrument: ConsumerInstrument,
        alternatives: List[Tuple[ConsumerInstrument, Dict[str, Any]]],
        merchant_proposal: MerchantProposal,
        scoring_details: Dict[str, Any],
    ) -> str:
        """Generate comprehensive explanation for instrument selection."""

        # Use LLM explanation if available, otherwise generate deterministic explanation
        if is_consumer_llm_configured():
            try:
                llm_explanation = explain_consumer_instrument_choice(
                    selected_instrument, alternatives, merchant_proposal
                )
                if llm_explanation:
                    return llm_explanation
            except Exception as e:
                logger.warning(f"LLM explanation failed, using deterministic: {e}")

        # Generate deterministic explanation
        explanation_parts = []

        # Main selection reason
        instrument_type = selected_instrument.instrument_type
        explanation_parts.append(
            f"Selected {instrument_type.replace('_', ' ').title()} from {selected_instrument.provider}"
        )

        # Value proposition
        if selected_instrument.net_value > 0:
            explanation_parts.append(f"Provides ${selected_instrument.net_value:.2f} net value")

        # Rewards highlight
        if selected_instrument.total_reward_value > 0:
            explanation_parts.append(
                f"Earns ${selected_instrument.total_reward_value:.2f} in rewards"
            )

        # Key factors
        if selected_instrument.selection_factors:
            top_factors = selected_instrument.selection_factors[:2]
            explanation_parts.append(f"Key benefits: {', '.join(top_factors)}")

        # Merchant benefits
        merchant_savings = self._calculate_merchant_savings(selected_instrument, merchant_proposal)
        if merchant_savings > 0:
            explanation_parts.append(
                f"Could save merchant ${merchant_savings:.2f} vs current proposal"
            )

        # Alternative comparison
        if alternatives:
            best_alt = alternatives[0][0]
            explanation_parts.append(
                f"Chosen over {best_alt.instrument_type.replace('_', ' ')} (score: {best_alt.value_score:.2f})"
            )

        return ". ".join(explanation_parts) + "."


# Convenience function for backward compatibility
def enhanced_counter_negotiate(request: CounterNegotiationRequest) -> CounterNegotiationResponse:
    """
    Enhanced counter-negotiation function with multi-instrument support.

    Args:
        request: Counter-negotiation request

    Returns:
        Enhanced counter-negotiation response
    """
    negotiator = MultiInstrumentNegotiator()
    return negotiator.counter_negotiate(request)
