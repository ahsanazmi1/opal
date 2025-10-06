"""
Multi-rail consumer negotiation engine for Opal Phase 7.

Provides consumer-centric negotiation logic that counters merchant proposals
with optimal consumer value across multiple rails and instruments, including
ACH, RTP, FedNow, SEPA, and various payment instruments.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.opal.controls import (
    RailType, InstrumentType, ConsumerReward, ConsumerProposal,
    CounterNegotiationResponse, MerchantProposal
)

logger = logging.getLogger(__name__)


class MultiRailConsumerCapability:
    """
    Consumer capability information for multi-rail and multi-instrument support.
    
    Tracks consumer's available payment methods, preferences, and constraints
    across different rails and instruments.
    """
    
    def __init__(self):
        """Initialize consumer capability tracking."""
        self.available_instruments = {}
        self.available_rails = {}
        self.consumer_preferences = {}
        self.reward_preferences = {}
        logger.info("Multi-rail consumer capability initialized")
    
    def add_instrument_capability(
        self,
        instrument_type: InstrumentType,
        max_amount: float,
        reward_rate: float = 0.0,
        preferred: bool = False
    ) -> None:
        """Add consumer's instrument capability."""
        self.available_instruments[instrument_type] = {
            "max_amount": max_amount,
            "reward_rate": reward_rate,
            "preferred": preferred,
            "available": True
        }
        logger.debug(f"Added instrument capability: {instrument_type}")
    
    def add_rail_capability(
        self,
        rail_type: RailType,
        supported_instruments: List[InstrumentType],
        consumer_benefit: float = 0.0,
        preferred: bool = False
    ) -> None:
        """Add consumer's rail capability."""
        self.available_rails[rail_type] = {
            "supported_instruments": supported_instruments,
            "consumer_benefit": consumer_benefit,
            "preferred": preferred,
            "available": True
        }
        logger.debug(f"Added rail capability: {rail_type}")
    
    def get_available_combinations(self) -> List[Tuple[RailType, InstrumentType]]:
        """Get all available rail-instrument combinations for the consumer."""
        combinations = []
        for rail_type, rail_info in self.available_rails.items():
            if rail_info["available"]:
                for instrument in rail_info["supported_instruments"]:
                    if instrument in self.available_instruments:
                        if self.available_instruments[instrument]["available"]:
                            combinations.append((rail_type, instrument))
        return combinations
    
    def get_preferred_combinations(self) -> List[Tuple[RailType, InstrumentType]]:
        """Get preferred rail-instrument combinations."""
        combinations = []
        for rail_type, rail_info in self.available_rails.items():
            if rail_info["preferred"] and rail_info["available"]:
                for instrument in rail_info["supported_instruments"]:
                    if (instrument in self.available_instruments and 
                        self.available_instruments[instrument]["preferred"] and
                        self.available_instruments[instrument]["available"]):
                        combinations.append((rail_type, instrument))
        return combinations


class MultiRailConsumerEvaluator:
    """
    Consumer-centric evaluator for multi-rail and multi-instrument combinations.
    
    Evaluates combinations from the consumer's perspective, focusing on
    rewards, benefits, convenience, and value maximization.
    """
    
    def __init__(self):
        """Initialize the consumer evaluator."""
        self.capability = MultiRailConsumerCapability()
        self._initialize_default_capabilities()
        logger.info("Multi-rail consumer evaluator initialized")
    
    def _initialize_default_capabilities(self) -> None:
        """Initialize default consumer capabilities for demo/testing."""
        # Add default instrument capabilities
        self.capability.add_instrument_capability("credit_card", 25000.0, reward_rate=0.015, preferred=True)
        self.capability.add_instrument_capability("debit_card", 10000.0, reward_rate=0.005)
        self.capability.add_instrument_capability("prepaid_card", 5000.0, reward_rate=0.01)
        self.capability.add_instrument_capability("stablecoin", 50000.0, reward_rate=0.02, preferred=True)
        self.capability.add_instrument_capability("bank_transfer", 100000.0, reward_rate=0.0)
        self.capability.add_instrument_capability("digital_wallet", 25000.0, reward_rate=0.008)
        self.capability.add_instrument_capability("bnpl", 15000.0, reward_rate=0.0)
        
        # Add default rail capabilities
        self.capability.add_rail_capability("Card", ["credit_card", "debit_card", "prepaid_card"], consumer_benefit=0.02)
        self.capability.add_rail_capability("ACH", ["debit_card", "bank_transfer"], consumer_benefit=0.0)
        self.capability.add_rail_capability("RTP", ["debit_card", "bank_transfer"], consumer_benefit=0.005, preferred=True)
        self.capability.add_rail_capability("FedNow", ["debit_card", "bank_transfer"], consumer_benefit=0.01, preferred=True)
        self.capability.add_rail_capability("SEPA", ["bank_transfer", "debit_card"], consumer_benefit=0.0)
        self.capability.add_rail_capability("Crypto", ["stablecoin"], consumer_benefit=0.03, preferred=True)
        self.capability.add_rail_capability("Wire", ["bank_transfer"], consumer_benefit=-0.001)
    
    def evaluate_consumer_combination(
        self,
        rail_type: RailType,
        instrument_type: InstrumentType,
        merchant_proposal: MerchantProposal,
        transaction_amount: float,
        consumer_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a rail-instrument combination from consumer perspective.
        
        Args:
            rail_type: Payment rail type
            instrument_type: Payment instrument type
            merchant_proposal: Merchant's proposal to counter
            transaction_amount: Transaction amount
            consumer_context: Consumer context and preferences
            
        Returns:
            Consumer evaluation result or None if combination is not viable
        """
        # Check if combination is available to consumer
        available_combinations = self.capability.get_available_combinations()
        if (rail_type, instrument_type) not in available_combinations:
            return None
        
        # Get consumer capabilities
        instrument_info = self.capability.available_instruments.get(instrument_type)
        rail_info = self.capability.available_rails.get(rail_type)
        
        if not instrument_info or not rail_info:
            return None
        
        # Check amount constraints
        if transaction_amount > instrument_info["max_amount"]:
            return None
        
        # Calculate consumer value metrics
        reward_value = self._calculate_reward_value(instrument_info, transaction_amount)
        convenience_score = self._calculate_convenience_score(rail_type, instrument_type, consumer_context)
        cost_savings = self._calculate_cost_savings(merchant_proposal, rail_type, transaction_amount)
        preference_bonus = self._calculate_preference_bonus(instrument_info, rail_info)
        
        # Calculate total consumer value
        total_value = self._calculate_total_consumer_value(
            reward_value, convenience_score, cost_savings, preference_bonus
        )
        
        # Generate consumer explanation
        explanation = self._generate_consumer_explanation(
            rail_type, instrument_type, reward_value, convenience_score, cost_savings
        )
        
        # Calculate confidence
        confidence = self._calculate_consumer_confidence(instrument_info, rail_info, consumer_context)
        
        return {
            "rail_type": rail_type,
            "instrument_type": instrument_type,
            "total_consumer_value": total_value,
            "reward_value": reward_value,
            "convenience_score": convenience_score,
            "cost_savings": cost_savings,
            "preference_bonus": preference_bonus,
            "explanation": explanation,
            "confidence": confidence,
            "is_preferred": instrument_info["preferred"] and rail_info["preferred"],
            "is_available": True,
            "metadata": {
                "instrument_info": instrument_info,
                "rail_info": rail_info,
                "consumer_context": consumer_context
            }
        }
    
    def _calculate_reward_value(
        self,
        instrument_info: Dict[str, Any],
        transaction_amount: float
    ) -> float:
        """Calculate reward value for the consumer."""
        reward_rate = instrument_info["reward_rate"]
        return transaction_amount * reward_rate
    
    def _calculate_convenience_score(
        self,
        rail_type: RailType,
        instrument_type: InstrumentType,
        consumer_context: Dict[str, Any]
    ) -> float:
        """Calculate convenience score for the consumer."""
        base_score = 0.5
        
        # Real-time rails are more convenient
        real_time_rails = ["RTP", "FedNow", "Crypto", "Card"]
        if rail_type in real_time_rails:
            base_score += 0.3
        
        # Digital instruments are more convenient
        digital_instruments = ["digital_wallet", "stablecoin", "prepaid_card"]
        if instrument_type in digital_instruments:
            base_score += 0.2
        
        # Mobile-first consumers prefer digital options
        if consumer_context.get("prefers_mobile", False):
            if instrument_type in ["digital_wallet", "stablecoin"]:
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_cost_savings(
        self,
        merchant_proposal: MerchantProposal,
        rail_type: RailType,
        transaction_amount: float
    ) -> float:
        """Calculate cost savings compared to merchant proposal."""
        # If consumer is using a cheaper rail, they save on merchant fees
        # (This is a simplified model - in reality, consumers don't directly pay merchant fees)
        
        # Real-time rails might have lower fees for consumers
        if rail_type in ["RTP", "FedNow"] and merchant_proposal.rail_type == "ACH":
            return transaction_amount * 0.001  # 0.1% savings
        
        # Crypto rails might have very low fees
        if rail_type == "Crypto" and merchant_proposal.rail_type in ["Card", "ACH"]:
            return transaction_amount * 0.005  # 0.5% savings
        
        return 0.0
    
    def _calculate_preference_bonus(
        self,
        instrument_info: Dict[str, Any],
        rail_info: Dict[str, Any]
    ) -> float:
        """Calculate preference bonus for preferred options."""
        bonus = 0.0
        
        if instrument_info["preferred"]:
            bonus += 0.1
        
        if rail_info["preferred"]:
            bonus += 0.1
        
        return bonus
    
    def _calculate_total_consumer_value(
        self,
        reward_value: float,
        convenience_score: float,
        cost_savings: float,
        preference_bonus: float
    ) -> float:
        """Calculate total consumer value score."""
        # Normalize reward value to 0-1 scale (assuming max transaction of $10,000)
        normalized_reward = min(1.0, reward_value / 1000.0)  # $1000 reward = 1.0 score
        
        # Normalize cost savings to 0-1 scale
        normalized_savings = min(1.0, cost_savings / 100.0)  # $100 savings = 1.0 score
        
        # Weighted combination
        total_value = (
            normalized_reward * 0.4 +      # 40% reward value
            convenience_score * 0.3 +      # 30% convenience
            normalized_savings * 0.2 +     # 20% cost savings
            preference_bonus * 0.1         # 10% preference bonus
        )
        
        return min(1.0, total_value)
    
    def _generate_consumer_explanation(
        self,
        rail_type: RailType,
        instrument_type: InstrumentType,
        reward_value: float,
        convenience_score: float,
        cost_savings: float
    ) -> str:
        """Generate consumer-friendly explanation."""
        explanation_parts = []
        
        explanation_parts.append(f"{instrument_type.replace('_', ' ').title()} via {rail_type} rail")
        
        # Reward explanation
        if reward_value > 0:
            explanation_parts.append(f"earns ${reward_value:.2f} in rewards")
        
        # Convenience explanation
        if convenience_score > 0.7:
            explanation_parts.append("high convenience")
        elif convenience_score > 0.5:
            explanation_parts.append("good convenience")
        
        # Cost savings explanation
        if cost_savings > 0:
            explanation_parts.append(f"saves ${cost_savings:.2f}")
        
        # Real-time capability
        if rail_type in ["RTP", "FedNow", "Crypto"]:
            explanation_parts.append("instant processing")
        
        return ", ".join(explanation_parts) + "."
    
    def _calculate_consumer_confidence(
        self,
        instrument_info: Dict[str, Any],
        rail_info: Dict[str, Any],
        consumer_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the consumer evaluation."""
        confidence = 0.8  # Base confidence
        
        # Higher confidence for preferred options
        if instrument_info["preferred"]:
            confidence += 0.1
        
        if rail_info["preferred"]:
            confidence += 0.1
        
        # Higher confidence for established options
        established_instruments = ["credit_card", "debit_card", "bank_transfer"]
        if instrument_info["instrument_type"] in established_instruments:
            confidence += 0.05
        
        return min(1.0, confidence)


class MultiRailConsumerNegotiator:
    """
    Main consumer negotiation engine for multi-rail and multi-instrument support.
    
    Orchestrates consumer-centric evaluation and selection of optimal
    rail-instrument combinations to counter merchant proposals.
    """
    
    def __init__(self):
        """Initialize the consumer negotiator."""
        self.evaluator = MultiRailConsumerEvaluator()
        logger.info("Multi-rail consumer negotiator initialized")
    
    def negotiate_consumer_response(
        self,
        merchant_proposal: MerchantProposal,
        transaction_amount: float,
        consumer_context: Dict[str, Any],
        max_alternatives: int = 3
    ) -> CounterNegotiationResponse:
        """
        Negotiate consumer response to merchant proposal.
        
        Args:
            merchant_proposal: Merchant's rail proposal
            transaction_amount: Transaction amount
            consumer_context: Consumer context and preferences
            max_alternatives: Maximum number of alternatives to provide
            
        Returns:
            Consumer counter-negotiation response
        """
        # Get all available consumer combinations
        available_combinations = self.evaluator.capability.get_available_combinations()
        
        # Evaluate all combinations
        evaluations = []
        for rail_type, instrument_type in available_combinations:
            evaluation = self.evaluator.evaluate_consumer_combination(
                rail_type, instrument_type, merchant_proposal, transaction_amount, consumer_context
            )
            if evaluation:
                evaluations.append(evaluation)
        
        if not evaluations:
            # Fallback to merchant proposal if no consumer alternatives
            return self._create_fallback_response(merchant_proposal, transaction_amount)
        
        # Sort by consumer value (descending)
        evaluations.sort(key=lambda e: e["total_consumer_value"], reverse=True)
        
        # Select optimal consumer choice
        optimal_evaluation = evaluations[0]
        
        # Get top alternatives
        alternatives = evaluations[1:max_alternatives + 1]
        
        # Create consumer proposal
        consumer_proposal = self._create_consumer_proposal(optimal_evaluation, transaction_amount)
        
        # Create consumer rewards
        consumer_rewards = self._create_consumer_rewards(optimal_evaluation, transaction_amount)
        
        # Generate overall explanation
        explanation = self._generate_negotiation_explanation(
            optimal_evaluation, alternatives, merchant_proposal
        )
        
        # Calculate overall confidence
        confidence = optimal_evaluation["confidence"]
        
        return CounterNegotiationResponse(
            actor_id=consumer_context.get("actor_id", "consumer"),
            trace_id=merchant_proposal.trace_id,
            consumer_proposal=consumer_proposal,
            consumer_rewards=consumer_rewards,
            alternatives=alternatives,
            explanation=explanation,
            confidence=confidence,
            metadata={
                "total_combinations_evaluated": len(evaluations),
                "consumer_value_score": optimal_evaluation["total_consumer_value"],
                "negotiation_type": "multi_rail_consumer_optimization"
            }
        )
    
    def _create_fallback_response(
        self,
        merchant_proposal: MerchantProposal,
        transaction_amount: float
    ) -> CounterNegotiationResponse:
        """Create fallback response when no consumer alternatives are available."""
        consumer_proposal = ConsumerProposal(
            rail_type=merchant_proposal.rail_type,
            instrument_type="credit_card",  # Default fallback
            consumer_benefit=0.0,
            convenience_score=0.5,
            explanation="No consumer alternatives available, accepting merchant proposal"
        )
        
        consumer_rewards = [
            ConsumerReward(
                reward_type="cashback",
                reward_value=0.0,
                description="No rewards available"
            )
        ]
        
        return CounterNegotiationResponse(
            actor_id="consumer",
            trace_id=merchant_proposal.trace_id,
            consumer_proposal=consumer_proposal,
            consumer_rewards=consumer_rewards,
            alternatives=[],
            explanation="Accepting merchant proposal as no consumer alternatives are available",
            confidence=0.3,
            metadata={
                "total_combinations_evaluated": 0,
                "consumer_value_score": 0.0,
                "negotiation_type": "fallback_to_merchant_proposal"
            }
        )
    
    def _create_consumer_proposal(
        self,
        evaluation: Dict[str, Any],
        transaction_amount: float
    ) -> ConsumerProposal:
        """Create consumer proposal from evaluation."""
        return ConsumerProposal(
            rail_type=evaluation["rail_type"],
            instrument_type=evaluation["instrument_type"],
            consumer_benefit=evaluation["reward_value"],
            convenience_score=evaluation["convenience_score"],
            explanation=evaluation["explanation"]
        )
    
    def _create_consumer_rewards(
        self,
        evaluation: Dict[str, Any],
        transaction_amount: float
    ) -> List[ConsumerReward]:
        """Create consumer rewards from evaluation."""
        rewards = []
        
        # Primary reward from the instrument
        if evaluation["reward_value"] > 0:
            reward_type = self._determine_reward_type(evaluation["instrument_type"])
            rewards.append(ConsumerReward(
                reward_type=reward_type,
                reward_value=evaluation["reward_value"],
                description=f"{reward_type.replace('_', ' ').title()} from {evaluation['instrument_type'].replace('_', ' ').title()}"
            ))
        
        # Convenience bonus
        if evaluation["convenience_score"] > 0.7:
            rewards.append(ConsumerReward(
                reward_type="convenience",
                reward_value=0.0,  # Non-monetary
                description="High convenience and ease of use"
            ))
        
        # Cost savings bonus
        if evaluation["cost_savings"] > 0:
            rewards.append(ConsumerReward(
                reward_type="cost_savings",
                reward_value=evaluation["cost_savings"],
                description="Cost savings compared to merchant proposal"
            ))
        
        return rewards
    
    def _determine_reward_type(self, instrument_type: InstrumentType) -> str:
        """Determine reward type based on instrument."""
        reward_mapping = {
            "credit_card": "cashback",
            "debit_card": "cashback",
            "prepaid_card": "cashback",
            "stablecoin": "crypto_rewards",
            "digital_wallet": "points",
            "bnpl": "bnpl_benefits"
        }
        return reward_mapping.get(instrument_type, "cashback")
    
    def _generate_negotiation_explanation(
        self,
        optimal: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        merchant_proposal: MerchantProposal
    ) -> str:
        """Generate comprehensive negotiation explanation."""
        explanation_parts = []
        
        # Optimal choice
        explanation_parts.append(f"Selected {optimal['instrument_type'].replace('_', ' ').title()} via {optimal['rail_type']} rail")
        explanation_parts.append(f"with consumer value score {optimal['total_consumer_value']:.3f}")
        
        # Value breakdown
        if optimal['reward_value'] > 0:
            explanation_parts.append(f"earning ${optimal['reward_value']:.2f} in rewards")
        
        if optimal['cost_savings'] > 0:
            explanation_parts.append(f"and saving ${optimal['cost_savings']:.2f}")
        
        # Comparison to merchant proposal
        explanation_parts.append(f"compared to merchant's {merchant_proposal.rail_type} proposal")
        
        # Alternative mention
        if alternatives:
            second_best = alternatives[0]
            explanation_parts.append(f"Alternative: {second_best['instrument_type'].replace('_', ' ').title()}/{second_best['rail_type']} "
                                   f"(value: {second_best['total_consumer_value']:.3f})")
        
        return " ".join(explanation_parts) + "."


# Global instances
_consumer_evaluator = None
_consumer_negotiator = None


def get_multi_rail_consumer_evaluator() -> MultiRailConsumerEvaluator:
    """Get global consumer evaluator instance."""
    global _consumer_evaluator
    if _consumer_evaluator is None:
        _consumer_evaluator = MultiRailConsumerEvaluator()
    return _consumer_evaluator


def get_multi_rail_consumer_negotiator() -> MultiRailConsumerNegotiator:
    """Get global consumer negotiator instance."""
    global _consumer_negotiator
    if _consumer_negotiator is None:
        _consumer_negotiator = MultiRailConsumerNegotiator()
    return _consumer_negotiator
