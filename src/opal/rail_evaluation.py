"""
Rail evaluation logic for Opal consumer negotiation.

This module provides rail evaluation capabilities for Opal to assess different
payment rails (Card, ACH, RTP, etc.) from a consumer perspective and make
informed counter-proposals to merchant rail recommendations.
"""

import logging
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .controls import RailType, MerchantProposal
from .ml.value_scoring import score_consumer_instrument_value
from .llm.explain import explain_consumer_rail_choice, is_consumer_llm_configured

logger = logging.getLogger(__name__)


@dataclass
class RailEvaluation:
    """Evaluation result for a payment rail from consumer perspective."""

    rail_type: RailType
    consumer_benefit_score: float  # 0.0-1.0, higher = better for consumer
    cost_efficiency_score: float  # 0.0-1.0, higher = lower cost to consumer
    convenience_score: float  # 0.0-1.0, higher = more convenient
    reward_potential_score: float  # 0.0-1.0, higher = more reward opportunities
    composite_score: float  # 0.0-1.0, weighted combination
    explanation: str  # Human-readable explanation
    key_factors: List[str]  # Key factors in evaluation


class ConsumerRailPreferences(BaseModel):
    """Consumer preferences for rail evaluation."""

    reward_weight: float = Field(
        default=0.4, description="Weight for reward optimization", ge=0.0, le=1.0
    )
    cost_weight: float = Field(
        default=0.3, description="Weight for cost minimization", ge=0.0, le=1.0
    )
    convenience_weight: float = Field(
        default=0.2, description="Weight for convenience", ge=0.0, le=1.0
    )
    speed_weight: float = Field(default=0.1, description="Weight for speed", ge=0.0, le=1.0)

    # Consumer context
    transaction_amount: float = Field(..., description="Transaction amount", gt=0.0)
    available_balance: float = Field(default=10000.0, description="Available consumer balance")
    credit_limit: float = Field(default=5000.0, description="Available credit limit")
    merchant_category: str = Field(default="general", description="Merchant category")
    channel: str = Field(default="online", description="Transaction channel")


class RailEvaluationResult(BaseModel):
    """Result of rail evaluation for consumer negotiation."""

    preferred_rail: RailType = Field(..., description="Consumer's preferred rail")
    rail_evaluations: List[Dict[str, Any]] = Field(..., description="All rail evaluations")
    counter_proposal: Dict[str, Any] = Field(..., description="Consumer's counter-proposal")
    explanation: str = Field(..., description="Explanation for rail choice")
    confidence: float = Field(..., description="Confidence in rail selection", ge=0.0, le=1.0)
    negotiation_strategy: str = Field(..., description="Negotiation strategy")


def evaluate_rail_for_consumer(
    rail_type: RailType,
    merchant_proposal: MerchantProposal,
    preferences: ConsumerRailPreferences,
    available_instruments: List[Dict[str, Any]],
) -> RailEvaluation:
    """
    Evaluate a payment rail from consumer perspective.

    Args:
        rail_type: Payment rail to evaluate
        merchant_proposal: Merchant's proposal for this rail
        preferences: Consumer preferences and context
        available_instruments: Available consumer instruments

    Returns:
        RailEvaluation with scores and explanation
    """

    # Base rail characteristics
    rail_characteristics = get_rail_characteristics(rail_type)

    # Calculate consumer benefit score
    consumer_benefit_score = calculate_consumer_benefit_score(
        rail_type, rail_characteristics, preferences, available_instruments
    )

    # Calculate cost efficiency score (lower merchant cost = better for consumer indirectly)
    cost_efficiency_score = calculate_cost_efficiency_score(merchant_proposal, preferences)

    # Calculate convenience score
    convenience_score = calculate_convenience_score(rail_type, rail_characteristics, preferences)

    # Calculate reward potential score
    reward_potential_score = calculate_reward_potential_score(
        rail_type, rail_characteristics, available_instruments
    )

    # Calculate composite score with consumer preferences
    composite_score = (
        consumer_benefit_score * preferences.reward_weight
        + cost_efficiency_score * preferences.cost_weight
        + convenience_score * preferences.convenience_weight
        + reward_potential_score * preferences.reward_weight
    )

    # Generate explanation and key factors
    explanation, key_factors = generate_rail_explanation(
        rail_type,
        consumer_benefit_score,
        cost_efficiency_score,
        convenience_score,
        reward_potential_score,
        composite_score,
        merchant_proposal,
        preferences,
    )

    return RailEvaluation(
        rail_type=rail_type,
        consumer_benefit_score=consumer_benefit_score,
        cost_efficiency_score=cost_efficiency_score,
        convenience_score=convenience_score,
        reward_potential_score=reward_potential_score,
        composite_score=composite_score,
        explanation=explanation,
        key_factors=key_factors,
    )


def get_rail_characteristics(rail_type: RailType) -> Dict[str, Any]:
    """Get base characteristics for a payment rail."""

    characteristics = {
        "Card": {
            "settlement_speed": 1,  # Instant
            "reward_opportunities": 0.9,  # High rewards
            "consumer_protection": 0.9,  # Strong protection
            "convenience": 0.95,  # Very convenient
            "fees_to_consumer": 0.0,  # No direct fees
            "fraud_protection": 0.9,  # Strong fraud protection
            "dispute_resolution": 0.9,  # Easy disputes
        },
        "ACH": {
            "settlement_speed": 2,  # 1-2 days
            "reward_opportunities": 0.1,  # Very low rewards
            "consumer_protection": 0.3,  # Limited protection
            "convenience": 0.7,  # Moderately convenient
            "fees_to_consumer": 0.0,  # No direct fees
            "fraud_protection": 0.4,  # Limited fraud protection
            "dispute_resolution": 0.2,  # Difficult disputes
        },
        "RTP": {
            "settlement_speed": 0,  # Real-time
            "reward_opportunities": 0.2,  # Low rewards
            "consumer_protection": 0.5,  # Moderate protection
            "convenience": 0.8,  # Convenient
            "fees_to_consumer": 0.0,  # No direct fees
            "fraud_protection": 0.6,  # Moderate fraud protection
            "dispute_resolution": 0.4,  # Moderate disputes
        },
        "FedNow": {
            "settlement_speed": 0,  # Real-time
            "reward_opportunities": 0.1,  # Very low rewards
            "consumer_protection": 0.4,  # Limited protection
            "convenience": 0.6,  # Moderately convenient
            "fees_to_consumer": 0.0,  # No direct fees
            "fraud_protection": 0.5,  # Moderate fraud protection
            "dispute_resolution": 0.3,  # Limited disputes
        },
    }

    return characteristics.get(
        rail_type,
        {
            "settlement_speed": 3,
            "reward_opportunities": 0.0,
            "consumer_protection": 0.2,
            "convenience": 0.5,
            "fees_to_consumer": 0.0,
            "fraud_protection": 0.3,
            "dispute_resolution": 0.1,
        },
    )


def calculate_consumer_benefit_score(
    rail_type: RailType,
    characteristics: Dict[str, Any],
    preferences: ConsumerRailPreferences,
    available_instruments: List[Dict[str, Any]],
) -> float:
    """Calculate consumer benefit score for a rail."""

    base_score = 0.5  # Start with neutral score

    # Reward opportunities (higher = better)
    base_score += characteristics["reward_opportunities"] * 0.3

    # Consumer protection (higher = better)
    base_score += characteristics["consumer_protection"] * 0.25

    # Fraud protection (higher = better)
    base_score += characteristics["fraud_protection"] * 0.2

    # Dispute resolution (higher = better)
    base_score += characteristics["dispute_resolution"] * 0.15

    # Available instruments for this rail
    rail_instruments = [
        inst for inst in available_instruments if instrument_supports_rail(inst, rail_type)
    ]

    if rail_instruments:
        # Calculate ML value scores for available instruments
        best_instrument_score = 0.0
        for instrument in rail_instruments:
            try:
                ml_result = score_consumer_instrument_value(
                    rewards_rate=instrument.get("total_reward_value", 0.0)
                    / preferences.transaction_amount,
                    fee_rate=instrument.get("base_fee", 0.0),
                    loyalty_bonus=instrument.get("loyalty_multiplier", 1.0),
                    card_tier=1 if instrument.get("loyalty_tier") == "standard" else 2,
                    transaction_amount=preferences.transaction_amount,
                    net_reward_value=instrument.get("net_value", 0.0),
                    annual_fee=0.0,
                    merchant_category=preferences.merchant_category,
                    channel=preferences.channel,
                )
                best_instrument_score = max(best_instrument_score, ml_result.value_score)
            except Exception as e:
                logger.warning(f"ML scoring failed for instrument: {e}")
                best_instrument_score = max(
                    best_instrument_score, instrument.get("value_score", 0.5)
                )

        base_score += best_instrument_score * 0.1

    return min(1.0, max(0.0, base_score))


def calculate_cost_efficiency_score(
    merchant_proposal: MerchantProposal, preferences: ConsumerRailPreferences
) -> float:
    """Calculate cost efficiency score (lower merchant cost = better indirectly)."""

    # Lower merchant cost means more room for consumer benefits
    # Normalize based on transaction amount
    cost_percentage = merchant_proposal.merchant_cost / 10000  # Convert bps to percentage
    cost_ratio = cost_percentage / (
        preferences.transaction_amount / 100
    )  # Ratio of cost to transaction

    # Lower cost ratio = higher efficiency score
    efficiency_score = max(0.0, 1.0 - (cost_ratio * 10))  # Scale factor

    return min(1.0, efficiency_score)


def calculate_convenience_score(
    rail_type: RailType, characteristics: Dict[str, Any], preferences: ConsumerRailPreferences
) -> float:
    """Calculate convenience score for a rail."""

    base_convenience = characteristics["convenience"]

    # Adjust for settlement speed preference
    settlement_speed = characteristics["settlement_speed"]
    if preferences.channel == "online" and settlement_speed <= 1:
        speed_bonus = 0.1
    elif settlement_speed <= 2:
        speed_bonus = 0.05
    else:
        speed_bonus = 0.0

    return min(1.0, base_convenience + speed_bonus)


def calculate_reward_potential_score(
    rail_type: RailType,
    characteristics: Dict[str, Any],
    available_instruments: List[Dict[str, Any]],
) -> float:
    """Calculate reward potential score for a rail."""

    base_potential = characteristics["reward_opportunities"]

    # Check if we have high-reward instruments for this rail
    rail_instruments = [
        inst for inst in available_instruments if instrument_supports_rail(inst, rail_type)
    ]

    if rail_instruments:
        max_reward_rate = max(
            (
                inst.get("total_reward_value", 0.0) / 100.0
            )  # Assume $100 transaction for rate calculation
            for inst in rail_instruments
        )
        reward_bonus = min(0.3, max_reward_rate * 2)  # Scale reward rate to bonus
        return min(1.0, base_potential + reward_bonus)

    return base_potential


def instrument_supports_rail(instrument: Dict[str, Any], rail_type: RailType) -> bool:
    """Check if an instrument supports a specific rail."""

    instrument_type = instrument.get("instrument_type", "")

    rail_mapping = {
        "Card": ["credit_card", "debit_card", "prepaid_card"],
        "ACH": ["bank_transfer", "debit_card"],
        "RTP": ["bank_transfer", "debit_card"],
        "FedNow": ["bank_transfer", "debit_card"],
        "Wire": ["bank_transfer"],
        "Crypto": ["stablecoin", "digital_wallet"],
    }

    supported_types = rail_mapping.get(rail_type, [])
    return instrument_type in supported_types


def generate_rail_explanation(
    rail_type: RailType,
    consumer_benefit_score: float,
    cost_efficiency_score: float,
    convenience_score: float,
    reward_potential_score: float,
    composite_score: float,
    merchant_proposal: MerchantProposal,
    preferences: ConsumerRailPreferences,
) -> Tuple[str, List[str]]:
    """Generate explanation and key factors for rail evaluation."""

    key_factors = []

    # Analyze scores and generate factors
    if consumer_benefit_score > 0.7:
        key_factors.append("high consumer protection")
    elif consumer_benefit_score < 0.4:
        key_factors.append("limited consumer protection")

    if reward_potential_score > 0.6:
        key_factors.append("good reward opportunities")
    elif reward_potential_score < 0.3:
        key_factors.append("limited reward potential")

    if convenience_score > 0.8:
        key_factors.append("high convenience")
    elif convenience_score < 0.5:
        key_factors.append("moderate convenience")

    if cost_efficiency_score > 0.7:
        key_factors.append("cost efficient")
    elif cost_efficiency_score < 0.4:
        key_factors.append("higher processing costs")

    # Generate explanation
    explanation_parts = []

    if composite_score > 0.7:
        explanation_parts.append(f"{rail_type} rail shows strong consumer value")
    elif composite_score > 0.5:
        explanation_parts.append(f"{rail_type} rail provides moderate consumer benefits")
    else:
        explanation_parts.append(f"{rail_type} rail has limited consumer advantages")

    if key_factors:
        explanation_parts.append(f"Key factors: {', '.join(key_factors)}")

    explanation = ". ".join(explanation_parts) + "."

    return explanation, key_factors


async def evaluate_all_rails_for_consumer(
    available_rails: List[RailType],
    merchant_proposal: MerchantProposal,
    preferences: ConsumerRailPreferences,
    available_instruments: List[Dict[str, Any]],
) -> List[RailEvaluation]:
    """
    Evaluate all available rails from consumer perspective.

    Args:
        available_rails: List of rails to evaluate
        merchant_proposal: Current merchant proposal
        preferences: Consumer preferences and context
        available_instruments: Available consumer instruments

    Returns:
        List of RailEvaluation results
    """

    evaluations = []

    for rail_type in available_rails:
        try:
            evaluation = evaluate_rail_for_consumer(
                rail_type, merchant_proposal, preferences, available_instruments
            )
            evaluations.append(evaluation)

            logger.debug(
                f"Evaluated {rail_type} rail: score={evaluation.composite_score:.3f}",
                extra={
                    "rail_type": rail_type,
                    "composite_score": evaluation.composite_score,
                    "consumer_benefit": evaluation.consumer_benefit_score,
                    "reward_potential": evaluation.reward_potential_score,
                },
            )

        except Exception as e:
            logger.error(f"Failed to evaluate {rail_type} rail: {e}")
            # Add fallback evaluation
            fallback_evaluation = RailEvaluation(
                rail_type=rail_type,
                consumer_benefit_score=0.3,
                cost_efficiency_score=0.3,
                convenience_score=0.3,
                reward_potential_score=0.3,
                composite_score=0.3,
                explanation=f"Unable to evaluate {rail_type} rail due to error",
                key_factors=["evaluation_error"],
            )
            evaluations.append(fallback_evaluation)

    # Sort by composite score (descending)
    evaluations.sort(key=lambda x: x.composite_score, reverse=True)

    return evaluations


async def determine_consumer_rail_preference(
    available_rails: List[RailType],
    merchant_proposal: MerchantProposal,
    preferences: ConsumerRailPreferences,
    available_instruments: List[Dict[str, Any]],
) -> RailEvaluationResult:
    """
    Determine consumer's preferred rail and create counter-proposal.

    Args:
        available_rails: List of rails to evaluate
        merchant_proposal: Current merchant proposal
        preferences: Consumer preferences and context
        available_instruments: Available consumer instruments

    Returns:
        RailEvaluationResult with preferred rail and counter-proposal
    """

    # Evaluate all rails
    evaluations = await evaluate_all_rails_for_consumer(
        available_rails, merchant_proposal, preferences, available_instruments
    )

    if not evaluations:
        raise ValueError("No rail evaluations available")

    # Select preferred rail (highest composite score)
    preferred_evaluation = evaluations[0]
    preferred_rail = preferred_evaluation.rail_type

    # Generate explanation using LLM if available
    explanation = preferred_evaluation.explanation
    confidence = min(
        1.0, max(0.0, preferred_evaluation.composite_score)
    )  # Ensure confidence is 0-1

    if is_consumer_llm_configured():
        try:
            llm_explanation = await explain_consumer_rail_choice(
                preferred_rail=preferred_rail,
                all_evaluations=evaluations,
                merchant_proposal=merchant_proposal,
                preferences=preferences,
            )
            if llm_explanation:
                explanation = llm_explanation.explanation
                confidence = llm_explanation.confidence
        except Exception as e:
            logger.warning(f"LLM explanation failed, using deterministic: {e}")

    # Determine negotiation strategy
    if preferred_rail == merchant_proposal.rail_type:
        negotiation_strategy = "agree_with_merchant"
    else:
        # Calculate benefit difference to justify counter-proposal
        merchant_eval = next(
            (eval for eval in evaluations if eval.rail_type == merchant_proposal.rail_type), None
        )

        if (
            merchant_eval
            and (preferred_evaluation.composite_score - merchant_eval.composite_score) > 0.2
        ):
            negotiation_strategy = "counter_propose_with_justification"
        else:
            negotiation_strategy = "counter_propose_moderately"

    # Create counter-proposal
    counter_proposal = create_counter_proposal(
        preferred_rail, preferred_evaluation, merchant_proposal, preferences
    )

    return RailEvaluationResult(
        preferred_rail=preferred_rail,
        rail_evaluations=[eval.__dict__ for eval in evaluations],
        counter_proposal=counter_proposal,
        explanation=explanation,
        confidence=confidence,
        negotiation_strategy=negotiation_strategy,
    )


def create_counter_proposal(
    preferred_rail: RailType,
    evaluation: RailEvaluation,
    merchant_proposal: MerchantProposal,
    preferences: ConsumerRailPreferences,
) -> Dict[str, Any]:
    """Create consumer counter-proposal for preferred rail."""

    # Find best instrument for the preferred rail
    # This would be done by the existing instrument selection logic
    # For now, create a basic counter-proposal

    return {
        "rail_type": preferred_rail,
        "consumer_benefit": evaluation.consumer_benefit_score
        * preferences.transaction_amount
        * 0.02,  # Estimate 2% benefit
        "convenience_score": evaluation.convenience_score,
        "reward_potential": evaluation.reward_potential_score,
        "composite_score": evaluation.composite_score,
        "justification": evaluation.explanation,
        "key_factors": evaluation.key_factors,
        "merchant_cost_impact": estimate_merchant_cost_impact(preferred_rail, merchant_proposal),
    }


def estimate_merchant_cost_impact(
    preferred_rail: RailType, merchant_proposal: MerchantProposal
) -> Dict[str, Any]:
    """Estimate the impact on merchant costs for the preferred rail."""

    # Base cost estimates (in basis points)
    rail_costs = {
        "Card": 150.0,
        "ACH": 5.0,
        "RTP": 10.0,
        "FedNow": 8.0,
        "Wire": 25.0,
        "Crypto": 50.0,
    }

    preferred_cost = rail_costs.get(preferred_rail, 150.0)
    current_cost = merchant_proposal.merchant_cost

    cost_difference = preferred_cost - current_cost

    return {
        "preferred_rail_cost_bps": preferred_cost,
        "current_rail_cost_bps": current_cost,
        "cost_difference_bps": cost_difference,
        "cost_difference_percentage": (
            (cost_difference / current_cost) * 100 if current_cost > 0 else 0
        ),
        "impact": (
            "increase" if cost_difference > 0 else "decrease" if cost_difference < 0 else "neutral"
        ),
    }
