"""
Consumer Counter-Negotiation Logic for Opal Phase 3

This module implements consumer-side negotiation logic to counter merchant
rail proposals with optimal consumer instruments based on rewards, loyalty,
and out-of-pocket costs.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from .controls import (
    ConsumerInstrument,
    ConsumerReward,
    ConsumerProposal,
    MerchantProposal,
    CounterNegotiationRequest,
    CounterNegotiationResponse,
)
from .events import emit_consumer_explanation_event
from .ml.value_scoring import score_consumer_instrument_value
from .llm.explain import explain_consumer_instrument_choice, is_consumer_llm_configured
from .rail_evaluation import determine_consumer_rail_preference, ConsumerRailPreferences

# Set up logging
logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    """Generate a unique trace ID for negotiation."""
    return f"opal_neg_{uuid4().hex[:16]}"


def calculate_rewards_for_transaction(
    instrument: ConsumerInstrument, amount: float, mcc: Optional[str] = None
) -> List[ConsumerReward]:
    """
    Calculate rewards for a specific transaction amount and instrument.

    Args:
        instrument: Consumer instrument to calculate rewards for
        amount: Transaction amount
        mcc: Merchant Category Code for category-specific bonuses

    Returns:
        List of calculated rewards for this transaction
    """
    calculated_rewards = []

    for reward in instrument.rewards:
        # Calculate base reward value
        base_value = amount * reward.rate

        # Apply category bonus if applicable
        if reward.category_bonus and mcc:
            category_bonus = reward.category_bonus.get(mcc, 1.0)
            base_value *= category_bonus

        # Apply loyalty multiplier
        base_value *= instrument.loyalty_multiplier

        # Apply cap if specified
        if reward.cap:
            base_value = min(base_value, reward.cap)

        # Create calculated reward
        calculated_reward = ConsumerReward(
            reward_type=reward.reward_type,
            rate=reward.rate,
            value=base_value,
            category_bonus=reward.category_bonus,
            cap=reward.cap,
            description=reward.description,
        )
        calculated_rewards.append(calculated_reward)

    return calculated_rewards


def calculate_instrument_value(
    instrument: ConsumerInstrument,
    amount: float,
    mcc: Optional[str] = None,
    reward_weight: float = 0.5,
    cost_weight: float = 0.3,
    preference_weight: float = 0.2,
) -> ConsumerInstrument:
    """
    Calculate net value and scoring for a consumer instrument.

    Args:
        instrument: Consumer instrument to evaluate
        amount: Transaction amount
        mcc: Merchant Category Code
        reward_weight: Weight for reward optimization
        cost_weight: Weight for cost minimization
        preference_weight: Weight for consumer preferences

    Returns:
        Updated instrument with calculated values
    """
    # Calculate transaction-specific rewards
    calculated_rewards = calculate_rewards_for_transaction(instrument, amount, mcc)
    total_reward_value = sum(reward.value for reward in calculated_rewards)

    # Calculate out-of-pocket cost (usually 0 for credit cards, full amount for debit)
    if instrument.instrument_type in ["credit_card", "bnpl"]:
        out_of_pocket = 0.0  # No immediate out-of-pocket for credit/BNPL
    else:
        out_of_pocket = amount  # Full amount for debit/bank transfer

    # Calculate net value (rewards minus out-of-pocket)
    net_value = total_reward_value - out_of_pocket

    # Calculate normalized value score (0.0-1.0)
    # Higher rewards and lower out-of-pocket = higher score
    max_possible_reward = amount * 0.05  # Assume max 5% reward rate
    reward_score = (
        min(total_reward_value / max_possible_reward, 1.0) if max_possible_reward > 0 else 0.0
    )

    cost_score = max(0.0, 1.0 - (out_of_pocket / amount)) if amount > 0 else 0.0

    # Calculate composite value score
    value_score = (
        reward_score * reward_weight
        + cost_score * cost_weight
        + instrument.preference_score * preference_weight
    )

    # Generate selection factors
    selection_factors = []
    if total_reward_value > amount * 0.02:  # More than 2% rewards
        selection_factors.append("high reward rate")
    if out_of_pocket == 0:
        selection_factors.append("no immediate out-of-pocket cost")
    if instrument.loyalty_multiplier > 1.0:
        selection_factors.append("loyalty tier bonus")
    if instrument.instrument_type == "bnpl":
        selection_factors.append("buy now, pay later flexibility")

    # Generate exclusion reasons if not eligible
    exclusion_reasons = []
    if not instrument.eligible:
        exclusion_reasons.append("instrument not eligible")
    if instrument.available_balance < amount:
        exclusion_reasons.append("insufficient balance/credit")
    if instrument.instrument_type == "wallet" and amount > 1000:
        exclusion_reasons.append("wallet amount limit exceeded")

    # Update instrument with calculated values
    instrument.total_reward_value = total_reward_value
    instrument.out_of_pocket_cost = out_of_pocket
    instrument.net_value = net_value
    instrument.value_score = value_score
    instrument.selection_factors = selection_factors
    instrument.exclusion_reasons = exclusion_reasons

    return instrument


def evaluate_consumer_instruments(request: CounterNegotiationRequest) -> List[ConsumerInstrument]:
    """
    Evaluate all available consumer instruments for counter-negotiation.

    Args:
        request: Counter-negotiation request with available instruments

    Returns:
        List of evaluated instruments sorted by value score
    """
    evaluated_instruments = []

    for instrument in request.available_instruments:
        # Calculate values for this instrument
        evaluated_instrument = calculate_instrument_value(
            instrument=instrument,
            amount=request.transaction_amount,
            mcc=request.mcc,
            reward_weight=request.reward_weight,
            cost_weight=request.cost_weight,
            preference_weight=request.preference_weight,
        )

        # Only include eligible instruments
        if (
            evaluated_instrument.eligible
            and evaluated_instrument.available_balance >= request.transaction_amount
        ):
            evaluated_instruments.append(evaluated_instrument)

    # Sort by value score (highest first)
    evaluated_instruments.sort(key=lambda x: x.value_score, reverse=True)

    return evaluated_instruments


def generate_consumer_explanation(
    selected_instrument: ConsumerInstrument,
    rejected_instruments: List[ConsumerInstrument],
    merchant_proposal: MerchantProposal,
    request: CounterNegotiationRequest,
) -> str:
    """
    Generate human-readable explanation for consumer instrument selection.

    Args:
        selected_instrument: The selected consumer instrument
        rejected_instruments: Other instruments that were considered
        merchant_proposal: Original merchant proposal
        request: Original negotiation request

    Returns:
        Human-readable explanation
    """
    explanation_parts = []

    # Primary selection reason
    if selected_instrument.value_score > 0.8:
        explanation_parts.append(
            f"Selected {selected_instrument.instrument_type} {selected_instrument.provider} ending in {selected_instrument.last_four}"
        )

        # Highlight key benefits
        if selected_instrument.total_reward_value > 0:
            explanation_parts.append(f"for {selected_instrument.total_reward_value:.2f} in rewards")

        if selected_instrument.out_of_pocket_cost == 0:
            explanation_parts.append("with no immediate out-of-pocket cost")

        if selected_instrument.loyalty_multiplier > 1.0:
            explanation_parts.append(f"plus {selected_instrument.loyalty_tier} loyalty bonus")

    # Explain why other instruments were not chosen
    if rejected_instruments:
        primary_rejection = rejected_instruments[0]  # Highest scoring rejected

        if primary_rejection.value_score < selected_instrument.value_score * 0.8:
            if primary_rejection.total_reward_value < selected_instrument.total_reward_value * 0.5:
                explanation_parts.append(
                    f"Declined {primary_rejection.instrument_type} due to lower rewards"
                )
            elif primary_rejection.out_of_pocket_cost > selected_instrument.out_of_pocket_cost:
                explanation_parts.append(
                    f"Declined {primary_rejection.instrument_type} due to higher out-of-pocket cost"
                )

    # Reference merchant proposal
    explanation_parts.append(f"Countering merchant's {merchant_proposal.rail_type} proposal")

    return ". ".join(explanation_parts) + "."


async def negotiateWalletChoice(
    actor_id: str,
    transaction_amount: float,
    available_instruments: List[ConsumerInstrument],
    merchant_proposal: MerchantProposal,
    consumer_preferences: Dict[str, Any] = None,
    currency: str = "USD",
    merchant_id: Optional[str] = None,
    mcc: Optional[str] = None,
    channel: str = "online",
    deterministic_seed: int = 42,
) -> CounterNegotiationResponse:
    """
    Enhanced consumer wallet choice negotiation with ML value scoring.

    This function implements the core negotiateWalletChoice logic:
    - ML-powered value scoring for each instrument
    - Maximize rewards while minimizing out-of-pocket costs
    - Deterministic selection with loyalty boost scenarios
    - LLM-powered explanations for instrument selection

    Args:
        actor_id: Consumer actor identifier
        transaction_amount: Transaction amount
        available_instruments: List of available consumer instruments
        merchant_proposal: Merchant's rail proposal from Orca
        consumer_preferences: Consumer preferences and constraints
        currency: Transaction currency
        merchant_id: Merchant identifier
        mcc: Merchant Category Code
        channel: Transaction channel
        deterministic_seed: Seed for deterministic results

    Returns:
        CounterNegotiationResponse with selected instrument and explanation
    """
    import random
    import numpy as np

    # Set deterministic seed for consistent results
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)

    logger.info(
        f"Starting enhanced wallet choice negotiation for actor {actor_id}",
        extra={
            "actor_id": actor_id,
            "transaction_amount": transaction_amount,
            "available_instruments": len(available_instruments),
            "merchant_rail": merchant_proposal.rail_type,
            "deterministic_seed": deterministic_seed,
        },
    )

    # Create transaction context
    transaction_context = {
        "transaction_amount": transaction_amount,
        "currency": currency,
        "merchant_id": merchant_id,
        "mcc": mcc,
        "channel": channel,
        "deterministic_seed": deterministic_seed,
    }

    # 1. ML-powered value scoring for all instruments
    ml_value_scores = {}
    evaluated_instruments = []

    for instrument in available_instruments:
        try:
            # Calculate ML value score
            ml_result = score_consumer_instrument_value(
                rewards_rate=max([r.rate for r in instrument.rewards], default=0.0),
                fee_rate=instrument.base_fee,
                loyalty_bonus=instrument.loyalty_multiplier,
                card_tier=(
                    1
                    if instrument.loyalty_tier == "standard"
                    else 2 if instrument.loyalty_tier == "premium" else 3
                ),
                transaction_amount=transaction_amount,
                net_reward_value=instrument.net_value,
                annual_fee=0.0,  # Could be added to instrument model
                merchant_category=mcc or "general",
                channel=channel,
            )

            ml_value_scores[instrument.instrument_id] = ml_result.value_score

            # Update instrument with ML score
            instrument.value_score = ml_result.value_score
            instrument.selection_factors.extend(
                [
                    f"ml_value_score_{ml_result.value_score:.3f}",
                    f"model_{ml_result.model_type}",
                ]
            )

            evaluated_instruments.append(instrument)

            logger.debug(
                f"ML scored instrument {instrument.instrument_id}: {ml_result.value_score:.3f}",
                extra={
                    "instrument_id": instrument.instrument_id,
                    "instrument_type": instrument.instrument_type,
                    "ml_value_score": ml_result.value_score,
                    "model_type": ml_result.model_type,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to score instrument {instrument.instrument_id}: {e}")
            # Use fallback deterministic scoring
            ml_value_scores[instrument.instrument_id] = instrument.value_score
            evaluated_instruments.append(instrument)

    # 2. Filter eligible instruments (sufficient balance, etc.)
    eligible_instruments = []
    rejected_instruments = []

    for instrument in evaluated_instruments:
        # Check eligibility criteria
        if instrument.available_balance < transaction_amount:
            instrument.eligible = False
            instrument.exclusion_reasons.append("insufficient_balance")
            rejected_instruments.append(instrument)
        else:
            eligible_instruments.append(instrument)

    if not eligible_instruments:
        logger.warning(f"No eligible instruments for actor {actor_id}")
        if evaluated_instruments:
            # Select the best available (even if over limit)
            selected_instrument = max(evaluated_instruments, key=lambda x: x.value_score)
            selected_instrument.selection_factors.append("no_eligible_alternatives")
        else:
            raise ValueError("No available instruments for negotiation")
    else:
        # 3. Select best-value instrument using ML scores
        # Primary: ML value score, Secondary: net value, Tertiary: reward value
        selected_instrument = max(
            eligible_instruments, key=lambda x: (x.value_score, x.net_value, x.total_reward_value)
        )
        selected_instrument.selection_factors.append("highest_ml_value_score")

    # 4. Generate enhanced explanation using LLM
    explanation = ""
    if is_consumer_llm_configured():
        try:
            # Prepare data for LLM explanation
            selected_data = {
                "instrument_id": selected_instrument.instrument_id,
                "instrument_type": selected_instrument.instrument_type,
                "provider": selected_instrument.provider,
                "total_reward_value": selected_instrument.total_reward_value,
                "out_of_pocket_cost": selected_instrument.out_of_pocket_cost,
                "net_value": selected_instrument.net_value,
                "value_score": selected_instrument.value_score,
                "loyalty_tier": selected_instrument.loyalty_tier,
                "loyalty_multiplier": selected_instrument.loyalty_multiplier,
                "rewards": (
                    [
                        {"rate": r.rate, "value": r.value, "type": r.reward_type}
                        for r in selected_instrument.rewards
                    ]
                    if selected_instrument.rewards
                    else []
                ),
            }

            rejected_data = [
                {
                    "instrument_id": inst.instrument_id,
                    "instrument_type": inst.instrument_type,
                    "provider": inst.provider,
                    "total_reward_value": inst.total_reward_value,
                    "out_of_pocket_cost": inst.out_of_pocket_cost,
                    "net_value": inst.net_value,
                    "value_score": inst.value_score,
                    "exclusion_reasons": inst.exclusion_reasons,
                }
                for inst in rejected_instruments[:3]  # Top 3 rejected
            ]

            merchant_data = {
                "rail_type": merchant_proposal.rail_type,
                "merchant_cost": merchant_proposal.merchant_cost,
                "settlement_days": merchant_proposal.settlement_days,
                "risk_score": merchant_proposal.risk_score,
                "explanation": merchant_proposal.explanation,
                "trace_id": merchant_proposal.trace_id,
            }

            # Generate LLM explanation
            llm_response = explain_consumer_instrument_choice(
                selected_instrument=selected_data,
                rejected_instruments=rejected_data,
                merchant_proposal=merchant_data,
                consumer_preferences=consumer_preferences or {},
                transaction_context=transaction_context,
                ml_value_scores=ml_value_scores,
            )

            if llm_response:
                explanation = llm_response.explanation
                # Add structured analysis to selection factors
                if llm_response.key_factors:
                    selected_instrument.selection_factors.extend(llm_response.key_factors)
            else:
                explanation = generate_consumer_explanation(
                    selected_instrument, rejected_instruments, merchant_proposal, None
                )

        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            explanation = generate_consumer_explanation(
                selected_instrument, rejected_instruments, merchant_proposal, None
            )
    else:
        explanation = generate_consumer_explanation(
            selected_instrument, rejected_instruments, merchant_proposal, None
        )

    # 5. Calculate win-win metrics (counter-proposal details captured in response)
    merchant_savings = max(0.0, merchant_proposal.merchant_cost - selected_instrument.base_fee)
    consumer_value = selected_instrument.net_value

    # Enhanced win-win score with ML confidence
    normalized_consumer_value = min(1.0, (consumer_value / transaction_amount) * 10)
    normalized_merchant_savings = (
        min(1.0, (merchant_savings / merchant_proposal.merchant_cost) * 10)
        if merchant_proposal.merchant_cost > 0
        else 0
    )
    ml_confidence_bonus = max(ml_value_scores.values()) if ml_value_scores else 0.5

    win_win_score = (
        (normalized_consumer_value * 0.5)
        + (normalized_merchant_savings * 0.3)
        + (ml_confidence_bonus * 0.2)
    )
    win_win_score = min(1.0, max(0.0, win_win_score))

    # 7. Create response
    response = CounterNegotiationResponse(
        actor_id=actor_id,
        trace_id=merchant_proposal.trace_id,
        consumer_proposal=ConsumerProposal(
            rail_type=merchant_proposal.rail_type,
            instrument_type=selected_instrument.instrument_type,
            consumer_benefit=selected_instrument.net_value,
            convenience_score=selected_instrument.preference_score,
            explanation=f"Selected {selected_instrument.instrument_type} from {selected_instrument.provider} for optimal value",
        ),
        consumer_rewards=[
            ConsumerReward(
                reward_type="cashback",
                rate=0.02,  # 2% cashback rate
                value=selected_instrument.total_reward_value,
                description=f"Loyalty rewards from {selected_instrument.provider}",
            )
        ],
        alternatives=[
            {
                "instrument_type": inst.instrument_type,
                "provider": inst.provider,
                "net_value": inst.net_value,
                "reason": "Lower value score",
            }
            for inst in rejected_instruments
        ],
        explanation=explanation,
        confidence=0.85,  # Default confidence score
        metadata={
            "negotiation_type": "enhanced_wallet_choice",
            "ml_value_scoring": True,
            "deterministic_seed": deterministic_seed,
            "merchant_proposal_rail": merchant_proposal.rail_type,
            "consumer_instruments_evaluated": len(evaluated_instruments),
            "eligible_instruments_count": len(eligible_instruments),
            "ml_value_scores": ml_value_scores,
            "llm_explanation": is_consumer_llm_configured(),
            "merchant_savings": merchant_savings,
            "consumer_value": consumer_value,
            "win_win_score": win_win_score,
            "timestamp": datetime.now().isoformat(),
        },
    )

    # 8. Emit CloudEvent
    try:
        await emit_consumer_explanation_event(response, actor_id)
    except Exception as e:
        logger.error(f"Failed to emit consumer explanation event: {e}")

    logger.info(
        "Enhanced wallet choice negotiation completed",
        extra={
            "trace_id": merchant_proposal.trace_id,
            "selected_instrument": selected_instrument.instrument_type,
            "ml_value_score": selected_instrument.value_score,
            "consumer_value": consumer_value,
            "win_win_score": win_win_score,
            "llm_explanation": is_consumer_llm_configured(),
        },
    )

    return response


async def enhanced_counter_negotiation_with_rail_evaluation(
    request: CounterNegotiationRequest,
) -> CounterNegotiationResponse:
    """
    Enhanced counter-negotiation with rail evaluation capabilities.

    This function evaluates different payment rails from consumer perspective
    and makes informed counter-proposals instead of blindly accepting
    the merchant's rail recommendation.
    """
    try:
        logger.info(
            "Starting enhanced counter-negotiation with rail evaluation",
            extra={
                "actor_id": request.actor_id,
                "transaction_amount": request.transaction_amount,
                "merchant_rail": request.merchant_proposal.rail_type,
                "available_instruments": len(request.available_instruments),
            },
        )

        # 1. Evaluate rails from consumer perspective
        rail_preferences = ConsumerRailPreferences(
            reward_weight=request.reward_weight,
            cost_weight=request.cost_weight,
            convenience_weight=request.preference_weight,
            speed_weight=0.1,  # Default speed weight
            transaction_amount=request.transaction_amount,
            merchant_category=request.mcc or "general",
            channel=request.channel,
        )

        # Available rails to evaluate (typically Card and ACH)
        available_rails = ["Card", "ACH"]  # Could be expanded based on merchant capabilities

        rail_evaluation_result = await determine_consumer_rail_preference(
            available_rails=available_rails,
            merchant_proposal=request.merchant_proposal,
            preferences=rail_preferences,
            available_instruments=[inst.__dict__ for inst in request.available_instruments],
        )

        consumer_preferred_rail = rail_evaluation_result.preferred_rail
        logger.info(f"Consumer rail evaluation complete: preferred={consumer_preferred_rail}")

        # 2. Select best instrument for the preferred rail
        rail_instruments = [
            inst
            for inst in request.available_instruments
            if instrument_supports_rail_for_consumer(inst, consumer_preferred_rail)
        ]

        if not rail_instruments:
            # Fallback to merchant's rail if no instruments available for preferred rail
            logger.warning(
                f"No instruments available for preferred rail {consumer_preferred_rail}, using merchant rail {request.merchant_proposal.rail_type}"
            )
            consumer_preferred_rail = request.merchant_proposal.rail_type
            rail_instruments = request.available_instruments

        # 3. Use existing instrument selection logic for the chosen rail
        selected_instrument = None
        rejected_instruments = []

        if rail_instruments:
            # Apply ML value scoring to instruments for the chosen rail
            ml_value_scores = {}
            evaluated_instruments = []

            for instrument in rail_instruments:
                try:
                    ml_result = score_consumer_instrument_value(
                        rewards_rate=max([r.rate for r in instrument.rewards], default=0.0),
                        fee_rate=instrument.base_fee,
                        loyalty_bonus=instrument.loyalty_multiplier,
                        card_tier=(
                            1
                            if instrument.loyalty_tier == "standard"
                            else 2 if instrument.loyalty_tier == "premium" else 3
                        ),
                        transaction_amount=request.transaction_amount,
                        net_reward_value=instrument.net_value,
                        annual_fee=0.0,
                        merchant_category=request.mcc or "general",
                        channel=request.channel,
                    )

                    instrument.value_score = ml_result.value_score
                    ml_value_scores[instrument.instrument_id] = ml_result.value_score
                    evaluated_instruments.append(instrument)

                except Exception as e:
                    logger.warning(
                        f"ML scoring failed for instrument {instrument.instrument_id}: {e}"
                    )
                    ml_value_scores[instrument.instrument_id] = instrument.value_score
                    evaluated_instruments.append(instrument)

            # Select best instrument
            if evaluated_instruments:
                selected_instrument = max(evaluated_instruments, key=lambda x: x.value_score)
                rejected_instruments = [
                    inst
                    for inst in evaluated_instruments
                    if inst.instrument_id != selected_instrument.instrument_id
                ]
            else:
                selected_instrument = rail_instruments[0]  # Fallback

        if not selected_instrument:
            raise ValueError("No suitable instrument found for negotiation")

        # 4. Create counter-proposal with consumer's preferred rail
        consumer_benefit = selected_instrument.net_value

        # Adjust benefit based on rail choice
        if consumer_preferred_rail == "Card" and request.merchant_proposal.rail_type == "ACH":
            # Consumer prefers Card for rewards, but merchant wants ACH for cost
            # Calculate the benefit of Card over ACH
            card_benefit = selected_instrument.total_reward_value
            ach_benefit = 0.0  # ACH typically has no rewards
            consumer_benefit = card_benefit - ach_benefit
        elif consumer_preferred_rail == "ACH" and request.merchant_proposal.rail_type == "Card":
            # Consumer agrees with merchant on ACH (unusual but possible)
            consumer_benefit = 0.0  # No additional benefit

        # 5. Generate explanation
        explanation = rail_evaluation_result.explanation

        # Add instrument-specific explanation
        if consumer_preferred_rail != request.merchant_proposal.rail_type:
            explanation += f" Selected {selected_instrument.instrument_type} from {selected_instrument.provider} for optimal value on {consumer_preferred_rail} rail."
        else:
            explanation += f" Selected {selected_instrument.instrument_type} from {selected_instrument.provider} for optimal value."

        # 6. Create response with consumer's rail preference
        response = CounterNegotiationResponse(
            actor_id=request.actor_id,
            trace_id=request.merchant_proposal.trace_id,
            consumer_proposal=ConsumerProposal(
                rail_type=consumer_preferred_rail,  # ← Now using consumer's preferred rail!
                instrument_type=selected_instrument.instrument_type,
                consumer_benefit=consumer_benefit,
                convenience_score=selected_instrument.preference_score,
                explanation=f"Selected {selected_instrument.instrument_type} from {selected_instrument.provider} for optimal value",
            ),
            consumer_rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    rate=0.02,  # 2% cashback rate
                    value=selected_instrument.total_reward_value,
                    description=f"Loyalty rewards from {selected_instrument.provider}",
                )
            ],
            alternatives=[
                {
                    "instrument_type": inst.instrument_type,
                    "provider": inst.provider,
                    "net_value": inst.net_value,
                    "reason": "Lower value score",
                }
                for inst in rejected_instruments
            ],
            explanation=explanation,
            confidence=rail_evaluation_result.confidence,
            metadata={
                "weights": {
                    "reward_weight": request.reward_weight,
                    "cost_weight": request.cost_weight,
                    "preference_weight": request.preference_weight,
                },
                "merchant_proposal": request.merchant_proposal.__dict__,
                "instruments_evaluated": len(rail_instruments),
                "consumer_preferences": {
                    "prefer_instant_settlement": True,
                    "cost_sensitivity": 0.7,
                    "risk_tolerance": 0.5,
                },
                "merchant_savings": request.merchant_proposal.merchant_cost,
                "win_win_score": rail_evaluation_result.confidence,
                "rail_evaluation": {
                    "preferred_rail": consumer_preferred_rail,
                    "merchant_rail": request.merchant_proposal.rail_type,
                    "rail_agreement": consumer_preferred_rail
                    == request.merchant_proposal.rail_type,
                    "negotiation_strategy": rail_evaluation_result.negotiation_strategy,
                    "rail_evaluations": rail_evaluation_result.rail_evaluations,
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info(
            "Enhanced counter-negotiation complete",
            extra={
                "actor_id": request.actor_id,
                "consumer_rail": consumer_preferred_rail,
                "merchant_rail": request.merchant_proposal.rail_type,
                "rail_agreement": consumer_preferred_rail == request.merchant_proposal.rail_type,
                "selected_instrument": selected_instrument.instrument_type,
                "consumer_benefit": consumer_benefit,
            },
        )

        return response

    except Exception as e:
        logger.error(f"Enhanced counter-negotiation failed: {e}")
        # Fallback to original negotiation logic
        return await counter_negotiation(request)


def instrument_supports_rail_for_consumer(instrument: ConsumerInstrument, rail_type: str) -> bool:
    """Check if an instrument supports a specific rail from consumer perspective."""

    instrument_type = instrument.instrument_type

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


async def counter_negotiation(request: CounterNegotiationRequest) -> CounterNegotiationResponse:
    """
    Perform consumer counter-negotiation against merchant proposal.

    Args:
        request: Counter-negotiation request with merchant proposal and available instruments

    Returns:
        Counter-negotiation response with selected instrument and explanation
    """
    trace_id = generate_trace_id()
    timestamp = datetime.now()

    logger.info(
        "Starting consumer counter-negotiation",
        extra={
            "trace_id": trace_id,
            "actor_id": request.actor_id,
            "amount": request.transaction_amount,
            "merchant_proposal": request.merchant_proposal.rail_type,
            "available_instruments": len(request.available_instruments),
        },
    )

    # Evaluate all available instruments
    evaluated_instruments = evaluate_consumer_instruments(request)

    if not evaluated_instruments:
        raise ValueError("No eligible instruments available for counter-negotiation")

    # Select the best instrument
    selected_instrument = evaluated_instruments[0]
    rejected_instruments = evaluated_instruments[1:] if len(evaluated_instruments) > 1 else []

    # Generate explanation
    explanation = generate_consumer_explanation(
        selected_instrument, rejected_instruments, request.merchant_proposal, request
    )

    # Calculate merchant savings potential (counter-proposal details captured in response)
    merchant_savings = max(
        0, request.merchant_proposal.merchant_cost - selected_instrument.base_fee
    )

    # Calculate win-win score (higher is better for both parties)
    consumer_benefit = (
        selected_instrument.net_value / request.transaction_amount
        if request.transaction_amount > 0
        else 0
    )
    merchant_benefit = (
        merchant_savings / request.merchant_proposal.merchant_cost
        if request.merchant_proposal.merchant_cost > 0
        else 0
    )
    win_win_score = (consumer_benefit + merchant_benefit) / 2

    response = CounterNegotiationResponse(
        actor_id=request.actor_id,
        trace_id=trace_id,
        consumer_proposal=ConsumerProposal(
            rail_type=request.merchant_proposal.rail_type,
            instrument_type=selected_instrument.instrument_type,
            consumer_benefit=selected_instrument.net_value,
            convenience_score=selected_instrument.preference_score,
            explanation=f"Selected {selected_instrument.instrument_type} from {selected_instrument.provider} for optimal value",
        ),
        consumer_rewards=[
            ConsumerReward(
                reward_type="cashback",
                rate=0.02,  # 2% cashback rate
                value=selected_instrument.total_reward_value,
                description=f"Loyalty rewards from {selected_instrument.provider}",
            )
        ],
        alternatives=[
            {
                "instrument_type": inst.instrument_type,
                "provider": inst.provider,
                "net_value": inst.net_value,
                "reason": "Lower value score",
            }
            for inst in rejected_instruments
        ],
        explanation=explanation,
        confidence=0.85,  # Default confidence score
        metadata={
            "weights": {
                "reward_weight": getattr(request, "reward_weight", 0.4),
                "cost_weight": getattr(request, "cost_weight", 0.3),
                "preference_weight": getattr(request, "preference_weight", 0.3),
            },
            "merchant_proposal": request.merchant_proposal.dict(),
            "instruments_evaluated": len(evaluated_instruments),
            "consumer_preferences": getattr(request, "consumer_preferences", {}),
            "merchant_savings": merchant_savings,
            "win_win_score": win_win_score,
            "timestamp": timestamp.isoformat(),
        },
    )

    # Emit CloudEvent for consumer explanation
    try:
        await emit_consumer_explanation_event(response, request.actor_id)
    except Exception as e:
        logger.error(f"Failed to emit consumer explanation event: {e}")

    logger.info(
        "Consumer counter-negotiation completed",
        extra={
            "trace_id": trace_id,
            "selected_instrument": selected_instrument.instrument_type,
            "consumer_value": selected_instrument.net_value,
            "win_win_score": win_win_score,
        },
    )

    return response


# Utility functions for creating sample instruments


def create_sample_credit_card(
    instrument_id: str = "cc_chase_sapphire_preferred",
    provider: str = "Chase",
    last_four: str = "4532",
    reward_rate: float = 0.02,
    loyalty_tier: str = "Gold",
    loyalty_multiplier: float = 1.2,
    available_balance: float = 15000.0,
) -> ConsumerInstrument:
    """Create a sample credit card instrument based on Chase Sapphire Preferred."""
    return ConsumerInstrument(
        instrument_id=instrument_id,
        instrument_type="credit_card",
        provider=provider,
        last_four=last_four,
        base_fee=150.0,  # 1.5% merchant fee
        out_of_pocket_cost=0.0,  # Will be calculated
        available_balance=available_balance,
        rewards=[
            ConsumerReward(
                reward_type="cashback",
                rate=reward_rate,
                value=0.0,  # Will be calculated
                description=f"{reward_rate*100:.1f}% cashback on travel & dining",
            )
        ],
        total_reward_value=0.0,  # Will be calculated
        loyalty_tier=loyalty_tier,
        loyalty_multiplier=loyalty_multiplier,
        net_value=0.0,  # Will be calculated
        value_score=0.0,  # Will be calculated
        eligible=True,
        preference_score=0.8,
    )


def create_sample_bnpl(
    instrument_id: str = "bnpl_klarna_5678",
    provider: str = "Klarna",
    last_four: str = "5678",
    available_balance: float = 5000.0,
) -> ConsumerInstrument:
    """Create a sample BNPL instrument."""
    return ConsumerInstrument(
        instrument_id=instrument_id,
        instrument_type="bnpl",
        provider=provider,
        last_four=last_four,
        base_fee=200.0,  # 2% merchant fee (higher for BNPL)
        out_of_pocket_cost=0.0,  # Will be calculated
        available_balance=available_balance,
        rewards=[
            ConsumerReward(
                reward_type="discount",
                rate=0.01,  # 1% discount for using BNPL
                value=0.0,  # Will be calculated
                description="1% BNPL discount",
            )
        ],
        total_reward_value=0.0,  # Will be calculated
        loyalty_tier=None,
        loyalty_multiplier=1.0,
        net_value=0.0,  # Will be calculated
        value_score=0.0,  # Will be calculated
        eligible=True,
        preference_score=0.7,
    )


def create_sample_stablecoin_wallet(
    instrument_id: str = "wallet_usdc_coinbase",
    provider: str = "Coinbase",
    last_four: str = "USDC",
    available_balance: float = 5000.0,
) -> ConsumerInstrument:
    """Create a sample stablecoin wallet instrument."""
    return ConsumerInstrument(
        instrument_id=instrument_id,
        instrument_type="stablecoin",
        provider=provider,
        last_four=last_four,
        base_fee=10.0,  # 0.1% network fee (very low)
        out_of_pocket_cost=0.0,  # Will be calculated
        available_balance=available_balance,
        rewards=[
            ConsumerReward(
                reward_type="crypto_rewards",
                rate=0.005,  # 0.5% crypto rewards
                value=0.0,  # Will be calculated
                description="0.5% USDC rewards",
            )
        ],
        total_reward_value=0.0,  # Will be calculated
        loyalty_tier=None,
        loyalty_multiplier=1.0,
        net_value=0.0,  # Will be calculated
        value_score=0.0,  # Will be calculated
        eligible=True,
        preference_score=0.6,  # Lower preference due to crypto volatility concerns
    )


def create_sample_debit_card(
    instrument_id: str = "dc_checking_9012",
    provider: str = "Bank of America",
    last_four: str = "9012",
    available_balance: float = 2000.0,
) -> ConsumerInstrument:
    """Create a sample debit card instrument."""
    return ConsumerInstrument(
        instrument_id=instrument_id,
        instrument_type="debit_card",
        provider=provider,
        last_four=last_four,
        base_fee=25.0,  # 0.25% merchant fee
        out_of_pocket_cost=0.0,  # Will be calculated
        available_balance=available_balance,
        rewards=[
            ConsumerReward(
                reward_type="cashback",
                rate=0.005,  # 0.5% cashback
                value=0.0,  # Will be calculated
                description="0.5% debit cashback",
            )
        ],
        total_reward_value=0.0,  # Will be calculated
        loyalty_tier=None,
        loyalty_multiplier=1.0,
        net_value=0.0,  # Will be calculated
        value_score=0.0,  # Will be calculated
        eligible=True,
        preference_score=0.6,
    )
