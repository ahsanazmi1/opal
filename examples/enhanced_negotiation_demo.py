#!/usr/bin/env python3
"""
Demo script for Opal Phase 4 Enhanced Negotiation functionality.

This script demonstrates the enhanced multi-instrument negotiation capabilities,
including support for credit cards, debit cards, BNPL, stablecoins, and more.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opal.controls import (
    ConsumerInstrument,
    ConsumerReward,
    MerchantProposal,
    CounterNegotiationRequest,
    InstrumentType,
    RewardType
)
from opal.enhanced_negotiation import MultiInstrumentNegotiator


def create_sample_instruments():
    """Create sample consumer instruments for demonstration."""
    
    instruments = [
        # Premium Credit Card with high rewards
        ConsumerInstrument(
            instrument_id="cc_premium_visa",
            instrument_type="credit_card",
            provider="Premium Bank",
            last_four="4321",
            base_fee=180,  # 1.8% in basis points
            out_of_pocket_cost=0.0,
            available_balance=8000.0,
            rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    reward_value=2.5,  # 2.5% cashback
                    description="2.5% cashback on all purchases"
                ),
                ConsumerReward(
                    reward_type="loyalty_points",
                    reward_value=1.0,  # Additional loyalty points
                    description="1x loyalty points bonus"
                )
            ],
            total_reward_value=7.0,  # $7 on $200 transaction
            loyalty_tier="platinum",
            loyalty_multiplier=1.3,
            net_value=7.0,
            value_score=0.0,  # Will be calculated
            eligible=True,
            preference_score=0.9,
            selection_factors=[],
            exclusion_reasons=[]
        ),
        
        # BNPL with benefits
        ConsumerInstrument(
            instrument_id="bnpl_klarna",
            instrument_type="bnpl",
            provider="Klarna",
            last_four="8888",
            base_fee=80,  # 0.8% in basis points
            out_of_pocket_cost=0.0,
            available_balance=1500.0,
            rewards=[
                ConsumerReward(
                    reward_type="bnpl_benefits",
                    reward_value=4.0,  # $4 in BNPL benefits
                    description="No interest if paid within 30 days"
                )
            ],
            total_reward_value=4.0,
            loyalty_multiplier=1.0,
            net_value=4.0,
            value_score=0.0,
            eligible=True,
            preference_score=0.8,
            selection_factors=[],
            exclusion_reasons=[]
        ),
        
        # Stablecoin with crypto rewards
        ConsumerInstrument(
            instrument_id="stable_usdc",
            instrument_type="stablecoin",
            provider="Coinbase",
            last_four="0001",
            base_fee=60,  # 0.6% in basis points
            out_of_pocket_cost=0.0,
            available_balance=1000.0,
            rewards=[
                ConsumerReward(
                    reward_type="crypto_rewards",
                    reward_value=2.4,  # $2.40 in crypto rewards
                    description="1.2% crypto rewards on USDC"
                )
            ],
            total_reward_value=2.4,
            loyalty_multiplier=1.0,
            net_value=2.4,
            value_score=0.0,
            eligible=True,
            preference_score=0.95,
            selection_factors=[],
            exclusion_reasons=[]
        ),
        
        # Debit Card with moderate rewards
        ConsumerInstrument(
            instrument_id="dc_chase",
            instrument_type="debit_card",
            provider="Chase",
            last_four="5678",
            base_fee=100,  # 1.0% in basis points
            out_of_pocket_cost=0.0,
            available_balance=3000.0,
            rewards=[
                ConsumerReward(
                    reward_type="cashback",
                    reward_value=1.0,  # 1% cashback
                    description="1% cashback on debit purchases"
                )
            ],
            total_reward_value=2.0,  # $2 on $200 transaction
            loyalty_multiplier=1.0,
            net_value=2.0,
            value_score=0.0,
            eligible=True,
            preference_score=0.6,
            selection_factors=[],
            exclusion_reasons=[]
        ),
        
        # Prepaid Card (no rewards but convenient)
        ConsumerInstrument(
            instrument_id="prepaid_visa",
            instrument_type="prepaid_card",
            provider="Visa Gift",
            last_four="7777",
            base_fee=120,  # 1.2% in basis points
            out_of_pocket_cost=0.0,
            available_balance=500.0,
            rewards=[],  # No rewards
            total_reward_value=0.0,
            loyalty_multiplier=1.0,
            net_value=0.0,
            value_score=0.0,
            eligible=True,
            preference_score=0.3,
            selection_factors=[],
            exclusion_reasons=[]
        )
    ]
    
    return instruments


def create_sample_merchant_proposal():
    """Create a sample merchant proposal."""
    return MerchantProposal(
        rail_type="ACH",
        merchant_cost=300,  # 3% in basis points
        settlement_days=3,
        risk_score=0.25,
        explanation="ACH rail offers lowest processing cost for merchant",
        trace_id="merchant_proposal_demo_123"
    )


def create_consumer_preferences():
    """Create sample consumer preferences."""
    return {
        "cash_rewards": 1.5,  # 50% preference boost for cash rewards
        "loyalty_programs": 0.8,  # 20% preference reduction for loyalty programs
        "bnpl_preference": 1.2,  # 20% preference boost for BNPL
        "convenience_preference": 1.1,  # 10% preference boost for convenience
        "crypto_rewards": 1.8,  # 80% preference boost for crypto rewards
        "category_preferences": {
            "cash_rewards": 1.5,
            "loyalty_programs": 0.8,
            "bnpl_preference": 1.2,
            "crypto_rewards": 1.8
        }
    }


def main():
    """Main demo function."""
    print("🚀 Opal Phase 4 Enhanced Negotiation Demo")
    print("=" * 60)
    
    # Create sample data
    instruments = create_sample_instruments()
    merchant_proposal = create_sample_merchant_proposal()
    consumer_preferences = create_consumer_preferences()
    
    print(f"💳 Available Instruments ({len(instruments)}):")
    for i, instrument in enumerate(instruments, 1):
        print(f"   {i}. {instrument.instrument_type.replace('_', ' ').title()} from {instrument.provider}")
        print(f"      - Balance: ${instrument.available_balance:,.2f}")
        print(f"      - Rewards: ${instrument.total_reward_value:.2f}")
        print(f"      - Preference Score: {instrument.preference_score:.1f}")
    
    print(f"\n🏪 Merchant Proposal:")
    print(f"   Rail Type: {merchant_proposal.rail_type}")
    print(f"   Cost: {merchant_proposal.merchant_cost} basis points ({merchant_proposal.merchant_cost/100:.1f}%)")
    print(f"   Settlement: {merchant_proposal.settlement_days} days")
    print(f"   Risk Score: {merchant_proposal.risk_score:.2f}")
    print(f"   Explanation: {merchant_proposal.explanation}")
    
    print(f"\n👤 Consumer Preferences:")
    for key, value in consumer_preferences.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     - {sub_key}: {sub_value:.1f}x")
        else:
            print(f"   {key}: {value:.1f}x")
    
    # Initialize enhanced negotiator
    negotiator = MultiInstrumentNegotiator()
    
    print(f"\n🔄 Performing Enhanced Counter-Negotiation...")
    
    # Create negotiation request
    request = CounterNegotiationRequest(
        actor_id="demo_consumer_123",
        transaction_amount=200.0,
        currency="USD",
        merchant_id="demo_merchant",
        merchant_proposal=merchant_proposal,
        available_instruments=instruments,
        consumer_preferences=consumer_preferences
    )
    
    # Perform negotiation
    response = negotiator.counter_negotiate(request)
    
    print(f"\n✅ Counter-Negotiation Complete!")
    print("=" * 60)
    
    # Display results
    selected = response.selected_instrument
    print(f"🎯 Selected Instrument: {selected.instrument_type.replace('_', ' ').title()}")
    print(f"🏦 Provider: {selected.provider}")
    print(f"💳 Last Four: ****{selected.last_four}")
    print(f"💰 Net Value: ${selected.net_value:.2f}")
    print(f"🎁 Total Rewards: ${selected.total_reward_value:.2f}")
    print(f"⭐ Value Score: {selected.value_score:.3f}")
    
    if selected.loyalty_tier:
        print(f"👑 Loyalty Tier: {selected.loyalty_tier} ({selected.loyalty_multiplier:.1f}x multiplier)")
    
    print(f"\n📝 Explanation:")
    print(f"   {response.explanation}")
    
    print(f"\n🤝 Win-Win Metrics:")
    print(f"   Consumer Value: ${response.consumer_value:.2f}")
    print(f"   Merchant Savings: ${response.merchant_savings:.2f}")
    print(f"   Win-Win Score: {response.win_win_score:.3f}")
    
    print(f"\n🔄 Alternatives ({len(response.rejected_instruments)}):")
    for i, alt in enumerate(response.rejected_instruments, 1):
        print(f"   {i}. {alt.instrument_type.replace('_', ' ').title()} from {alt.provider}")
        print(f"      - Value Score: {alt.value_score:.3f}")
        print(f"      - Net Value: ${alt.net_value:.2f}")
        if alt.selection_factors:
            print(f"      - Factors: {', '.join(alt.selection_factors[:2])}")
    
    print(f"\n📊 Counter-Proposal Details:")
    counter_proposal = response.counter_proposal
    print(f"   Proposed Instrument: {counter_proposal['proposed_instrument']['type']}")
    print(f"   Consumer Benefits:")
    print(f"     - Net Value: ${counter_proposal['consumer_benefits']['net_value']:.2f}")
    print(f"     - Total Rewards: ${counter_proposal['consumer_benefits']['total_rewards']:.2f}")
    print(f"     - Convenience Score: {counter_proposal['consumer_benefits']['convenience_score']:.3f}")
    print(f"   Merchant Benefits:")
    print(f"     - Potential Savings: ${counter_proposal['merchant_benefits']['potential_savings']:.2f}")
    print(f"     - Settlement Speed: {counter_proposal['merchant_benefits']['settlement_speed']}")
    print(f"   Win-Win Metrics:")
    print(f"     - Overall Efficiency: {counter_proposal['win_win_metrics']['overall_efficiency']:.3f}")
    
    print(f"\n📈 Negotiation Metadata:")
    metadata = response.negotiation_metadata
    print(f"   Instruments Evaluated: {metadata['total_instruments_evaluated']}")
    print(f"   Scoring Method: {metadata['scoring_method']}")
    print(f"   Trace ID: {response.trace_id}")
    print(f"   Timestamp: {response.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results to JSON file
    output_file = Path(__file__).parent / "enhanced_negotiation_demo_output.json"
    with open(output_file, "w") as f:
        # Convert response to dict for JSON serialization
        response_dict = {
            "selected_instrument": selected.dict(),
            "counter_proposal": counter_proposal,
            "explanation": response.explanation,
            "trace_id": response.trace_id,
            "timestamp": response.timestamp.isoformat(),
            "win_win_score": response.win_win_score,
            "consumer_value": response.consumer_value,
            "merchant_savings": response.merchant_savings,
            "negotiation_metadata": metadata
        }
        json.dump(response_dict, f, indent=2, default=str)
    
    print(f"\n💾 Negotiation results saved to: {output_file}")
    print("\n🎉 Enhanced Negotiation Demo completed successfully!")


if __name__ == "__main__":
    main()








