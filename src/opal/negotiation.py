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
    MerchantProposal, 
    CounterNegotiationRequest, 
    CounterNegotiationResponse,
    InstrumentType,
    RewardType
)
from .events import emit_consumer_explanation_event

# Set up logging
logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    """Generate a unique trace ID for negotiation."""
    return f"opal_neg_{uuid4().hex[:16]}"


def calculate_rewards_for_transaction(
    instrument: ConsumerInstrument, 
    amount: float, 
    mcc: Optional[str] = None
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
            description=reward.description
        )
        calculated_rewards.append(calculated_reward)
    
    return calculated_rewards


def calculate_instrument_value(
    instrument: ConsumerInstrument,
    amount: float,
    mcc: Optional[str] = None,
    reward_weight: float = 0.5,
    cost_weight: float = 0.3,
    preference_weight: float = 0.2
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
    reward_score = min(total_reward_value / max_possible_reward, 1.0) if max_possible_reward > 0 else 0.0
    
    cost_score = max(0.0, 1.0 - (out_of_pocket / amount)) if amount > 0 else 0.0
    
    # Calculate composite value score
    value_score = (
        reward_score * reward_weight +
        cost_score * cost_weight +
        instrument.preference_score * preference_weight
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


def evaluate_consumer_instruments(
    request: CounterNegotiationRequest
) -> List[ConsumerInstrument]:
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
            preference_weight=request.preference_weight
        )
        
        # Only include eligible instruments
        if evaluated_instrument.eligible and evaluated_instrument.available_balance >= request.transaction_amount:
            evaluated_instruments.append(evaluated_instrument)
    
    # Sort by value score (highest first)
    evaluated_instruments.sort(key=lambda x: x.value_score, reverse=True)
    
    return evaluated_instruments


def generate_consumer_explanation(
    selected_instrument: ConsumerInstrument,
    rejected_instruments: List[ConsumerInstrument],
    merchant_proposal: MerchantProposal,
    request: CounterNegotiationRequest
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
        explanation_parts.append(f"Selected {selected_instrument.instrument_type} {selected_instrument.provider} ending in {selected_instrument.last_four}")
        
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
                explanation_parts.append(f"Declined {primary_rejection.instrument_type} due to lower rewards")
            elif primary_rejection.out_of_pocket_cost > selected_instrument.out_of_pocket_cost:
                explanation_parts.append(f"Declined {primary_rejection.instrument_type} due to higher out-of-pocket cost")
    
    # Reference merchant proposal
    explanation_parts.append(f"Countering merchant's {merchant_proposal.rail_type} proposal")
    
    return ". ".join(explanation_parts) + "."


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
        f"Starting consumer counter-negotiation",
        extra={
            "trace_id": trace_id,
            "actor_id": request.actor_id,
            "amount": request.transaction_amount,
            "merchant_proposal": request.merchant_proposal.rail_type,
            "available_instruments": len(request.available_instruments)
        }
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
        selected_instrument, 
        rejected_instruments, 
        request.merchant_proposal,
        request
    )
    
    # Create counter-proposal
    counter_proposal = {
        "instrument_type": selected_instrument.instrument_type,
        "instrument_id": selected_instrument.instrument_id,
        "provider": selected_instrument.provider,
        "consumer_value": selected_instrument.net_value,
        "rewards": selected_instrument.total_reward_value,
        "out_of_pocket": selected_instrument.out_of_pocket_cost,
        "settlement": "immediate" if selected_instrument.instrument_type in ["credit_card", "bnpl"] else "instant",
        "merchant_benefit": "lower_processing_cost" if selected_instrument.base_fee < request.merchant_proposal.merchant_cost else "same_processing_cost"
    }
    
    # Calculate merchant savings potential
    merchant_savings = max(0, request.merchant_proposal.merchant_cost - selected_instrument.base_fee)
    
    # Calculate win-win score (higher is better for both parties)
    consumer_benefit = selected_instrument.net_value / request.transaction_amount if request.transaction_amount > 0 else 0
    merchant_benefit = merchant_savings / request.merchant_proposal.merchant_cost if request.merchant_proposal.merchant_cost > 0 else 0
    win_win_score = (consumer_benefit + merchant_benefit) / 2
    
    response = CounterNegotiationResponse(
        selected_instrument=selected_instrument,
        counter_proposal=counter_proposal,
        explanation=explanation,
        trace_id=trace_id,
        timestamp=timestamp,
        negotiation_metadata={
            "weights": {
                "reward_weight": request.reward_weight,
                "cost_weight": request.cost_weight,
                "preference_weight": request.preference_weight,
            },
            "merchant_proposal": request.merchant_proposal.dict(),
            "instruments_evaluated": len(evaluated_instruments),
            "consumer_preferences": request.consumer_preferences,
        },
        rejected_instruments=rejected_instruments,
        merchant_savings=merchant_savings,
        consumer_value=selected_instrument.net_value,
        win_win_score=win_win_score,
    )
    
    # Emit CloudEvent for consumer explanation
    try:
        await emit_consumer_explanation_event(response, request.actor_id)
    except Exception as e:
        logger.error(f"Failed to emit consumer explanation event: {e}")
    
    logger.info(
        f"Consumer counter-negotiation completed",
        extra={
            "trace_id": trace_id,
            "selected_instrument": selected_instrument.instrument_type,
            "consumer_value": selected_instrument.net_value,
            "win_win_score": win_win_score
        }
    )
    
    return response


# Utility functions for creating sample instruments

def create_sample_credit_card(
    instrument_id: str = "cc_visa_1234",
    provider: str = "Visa",
    last_four: str = "1234",
    reward_rate: float = 0.02,
    loyalty_tier: str = "Gold",
    loyalty_multiplier: float = 1.2,
    available_balance: float = 10000.0
) -> ConsumerInstrument:
    """Create a sample credit card instrument."""
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
                description=f"{reward_rate*100:.1f}% cashback"
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
    available_balance: float = 5000.0
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
                description="1% BNPL discount"
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


def create_sample_debit_card(
    instrument_id: str = "dc_checking_9012",
    provider: str = "Bank of America",
    last_four: str = "9012",
    available_balance: float = 2000.0
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
                description="0.5% debit cashback"
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
