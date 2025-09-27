"""
Deterministic spend controls for Opal wallet operations.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, cast, Literal
from datetime import datetime

from pydantic import BaseModel, Field


class PaymentMethod(BaseModel):
    """Payment method information."""

    method_id: str = Field(..., description="Unique payment method identifier")
    type: str = Field(..., description="Payment method type (card, bank, wallet)")
    provider: str = Field(..., description="Payment provider (visa, mastercard, etc.)")
    last_four: str = Field(..., description="Last four digits of account/card")
    expiry_month: Optional[int] = Field(None, description="Expiry month (for cards)")
    expiry_year: Optional[int] = Field(None, description="Expiry year (for cards)")
    status: str = Field("active", description="Payment method status")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata")


class TransactionRequest(BaseModel):
    """Transaction request for spend control evaluation."""

    amount: Decimal = Field(..., gt=0, description="Transaction amount")
    currency: str = Field("USD", description="Transaction currency")
    mcc: Optional[str] = Field(None, description="Merchant Category Code")
    channel: str = Field(..., description="Transaction channel (web, mobile, pos)")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    actor_id: str = Field(..., description="Actor/user identifier")
    payment_method_id: str = Field(..., description="Selected payment method ID")


class SpendControlResult(BaseModel):
    """Result of spend control evaluation."""

    allowed: bool = Field(..., description="Whether transaction is allowed")
    token_reference: Optional[str] = Field(None, description="Token reference if allowed")
    reasons: List[str] = Field(..., description="Reasons for decision")
    limits_applied: List[str] = Field(default_factory=list, description="Limits that were checked")
    max_amount_allowed: Optional[Decimal] = Field(None, description="Maximum amount allowed")
    control_version: str = Field("v1.0.0", description="Control version used")


class SpendControls:
    """Deterministic spend controls for Opal wallet."""

    # Control limits by MCC and channel
    MCC_LIMITS = {
        # High-risk categories - lower limits
        "5999": {"max_amount": Decimal("100"), "description": "Miscellaneous retail - high risk"},
        "7995": {"max_amount": Decimal("50"), "description": "Gambling - restricted"},
        "7994": {"max_amount": Decimal("50"), "description": "Video game arcades - restricted"},
        # Medium-risk categories
        "5814": {"max_amount": Decimal("500"), "description": "Fast food restaurants"},
        "5812": {"max_amount": Decimal("300"), "description": "Restaurants"},
        "5541": {"max_amount": Decimal("200"), "description": "Gas stations"},
        # Low-risk categories - higher limits
        "5411": {"max_amount": Decimal("2000"), "description": "Grocery stores"},
        "5310": {"max_amount": Decimal("1500"), "description": "Discount stores"},
    }

    # Channel limits
    CHANNEL_LIMITS = {
        "web": {"max_amount": Decimal("5000"), "description": "Web transactions"},
        "mobile": {"max_amount": Decimal("3000"), "description": "Mobile app transactions"},
        "pos": {"max_amount": Decimal("10000"), "description": "Point of sale transactions"},
        "atm": {"max_amount": Decimal("500"), "description": "ATM transactions"},
        "api": {"max_amount": Decimal("10000"), "description": "API transactions"},
    }

    # Daily limits by channel
    DAILY_LIMITS = {
        "web": Decimal("15000"),
        "mobile": Decimal("10000"),
        "pos": Decimal("25000"),
        "atm": Decimal("2000"),
        "api": Decimal("50000"),
    }

    # Payment method type limits
    METHOD_LIMITS = {
        "card": {"max_amount": Decimal("10000"), "description": "Card payments"},
        "bank": {"max_amount": Decimal("25000"), "description": "Bank transfers"},
        "wallet": {"max_amount": Decimal("5000"), "description": "Digital wallet"},
        "crypto": {"max_amount": Decimal("1000"), "description": "Cryptocurrency"},
    }

    @classmethod
    def evaluate_transaction(cls, request: TransactionRequest) -> SpendControlResult:
        """
        Evaluate a transaction request against spend controls.

        Args:
            request: Transaction request to evaluate

        Returns:
            Spend control result with allow/deny decision and token reference
        """
        reasons = []
        limits_applied = []
        max_amount_allowed = Decimal("999999")  # Start with very high limit

        # Check MCC limits
        if request.mcc and request.mcc in cls.MCC_LIMITS:
            mcc_limit: Decimal = cast(Decimal, cls.MCC_LIMITS[request.mcc]["max_amount"])
            max_amount_allowed = min(max_amount_allowed, mcc_limit)
            limits_applied.append(f"MCC {request.mcc} limit: ${mcc_limit}")

            if request.amount > mcc_limit:
                reasons.append(
                    f"Amount ${request.amount} exceeds MCC {request.mcc} limit of ${mcc_limit}"
                )
                return cls._create_denied_result(request, reasons, limits_applied)
            else:
                reasons.append(
                    f"Amount ${request.amount} within MCC {request.mcc} limit of ${mcc_limit}"
                )

        # Check channel limits
        if request.channel in cls.CHANNEL_LIMITS:
            channel_limit: Decimal = cast(
                Decimal, cls.CHANNEL_LIMITS[request.channel]["max_amount"]
            )
            max_amount_allowed = min(max_amount_allowed, channel_limit)
            limits_applied.append(f"Channel {request.channel} limit: ${channel_limit}")

            if request.amount > channel_limit:
                reasons.append(
                    f"Amount ${request.amount} exceeds {request.channel} channel limit of ${channel_limit}"
                )
                return cls._create_denied_result(request, reasons, limits_applied)
            else:
                reasons.append(
                    f"Amount ${request.amount} within {request.channel} channel limit of ${channel_limit}"
                )

        # Check daily limits (simplified - in production would check actual usage)
        if request.channel in cls.DAILY_LIMITS:
            daily_limit = cls.DAILY_LIMITS[request.channel]
            # For demo purposes, assume no previous transactions today
            # In production, this would query actual transaction history
            limits_applied.append(f"Daily {request.channel} limit: ${daily_limit}")
            reasons.append(f"Daily limit check passed for {request.channel} channel")

        # Generate token reference if allowed
        token_reference = cls._generate_token_reference(request)

        reasons.append(f"Transaction approved with token reference: {token_reference}")

        return SpendControlResult(
            allowed=True,
            token_reference=token_reference,
            reasons=reasons,
            limits_applied=limits_applied,
            max_amount_allowed=max_amount_allowed,
            control_version="v1.0.0",
        )

    @classmethod
    def _generate_token_reference(cls, request: TransactionRequest) -> str:
        """Generate a deterministic token reference for the transaction."""
        # Create a deterministic token based on request parameters
        token_data = f"{request.actor_id}:{request.payment_method_id}:{request.amount}:{request.channel}:{request.mcc or 'none'}"
        token_hash = hash(token_data)
        return f"tok_{request.actor_id}_{abs(token_hash)}"

    @classmethod
    def _create_denied_result(
        cls, request: TransactionRequest, reasons: List[str], limits_applied: List[str]
    ) -> SpendControlResult:
        """Create a denied spend control result."""
        return SpendControlResult(
            allowed=False,
            token_reference=None,
            reasons=reasons,
            limits_applied=limits_applied,
            max_amount_allowed=None,
            control_version="v1.0.0",
        )

    @classmethod
    def get_available_payment_methods(cls, actor_id: str) -> List[PaymentMethod]:
        """
        Get available payment methods for an actor (stubbed).

        Args:
            actor_id: Actor identifier

        Returns:
            List of available payment methods
        """
        # Stubbed payment methods - in production would query actual data
        methods = [
            PaymentMethod(
                method_id=f"pm_{actor_id}_visa_001",
                type="card",
                provider="visa",
                last_four="4242",
                expiry_month=12,
                expiry_year=2025,
                status="active",
                metadata={"card_type": "credit", "network": "visa"},
            ),
            PaymentMethod(
                method_id=f"pm_{actor_id}_mc_002",
                type="card",
                provider="mastercard",
                last_four="5555",
                expiry_month=10,
                expiry_year=2026,
                status="active",
                metadata={"card_type": "debit", "network": "mastercard"},
            ),
            PaymentMethod(
                method_id=f"pm_{actor_id}_bank_003",
                type="bank",
                provider="chase",
                last_four="1234",
                expiry_month=None,
                expiry_year=None,
                status="active",
                metadata={"account_type": "checking", "routing": "021000021"},
            ),
            PaymentMethod(
                method_id=f"pm_{actor_id}_wallet_004",
                type="wallet",
                provider="paypal",
                last_four="5678",
                expiry_month=None,
                expiry_year=None,
                status="active",
                metadata={"wallet_type": "paypal", "verified": "true"},
            ),
        ]

        return methods

    @classmethod
    def get_control_limits(cls) -> Dict[str, Any]:
        """Get current control limits and parameters."""
        return {
            "control_version": "v1.0.0",
            "mcc_limits": {
                mcc: {
                    "max_amount": float(cast(Decimal, limits["max_amount"])),
                    "description": limits["description"],
                }
                for mcc, limits in cls.MCC_LIMITS.items()
            },
            "channel_limits": {
                channel: {
                    "max_amount": float(cast(Decimal, limits["max_amount"])),
                    "description": limits["description"],
                }
                for channel, limits in cls.CHANNEL_LIMITS.items()
            },
            "daily_limits": {
                channel: float(cast(Decimal, limit)) for channel, limit in cls.DAILY_LIMITS.items()
            },
            "method_limits": {
                method: {
                    "max_amount": float(cast(Decimal, limits["max_amount"])),
                    "description": limits["description"],
                }
                for method, limits in cls.METHOD_LIMITS.items()
            },
        }


# Phase 3 - Consumer Counter-Negotiation Models

# Define valid instrument types for consumer negotiation
InstrumentType = Literal["credit_card", "debit_card", "bnpl", "wallet", "bank_transfer", "rewards_card"]

# Define valid reward types
RewardType = Literal["cashback", "points", "miles", "loyalty_points", "discount"]

class ConsumerReward(BaseModel):
    """Consumer reward information for an instrument."""
    
    reward_type: RewardType = Field(..., description="Type of reward")
    rate: float = Field(..., description="Reward rate (e.g., 0.02 for 2% cashback)", ge=0.0, le=1.0)
    value: float = Field(..., description="Reward value in transaction currency", ge=0.0)
    category_bonus: Optional[Dict[str, float]] = Field(None, description="Category-specific bonus rates")
    cap: Optional[float] = Field(None, description="Maximum reward per transaction")
    description: str = Field(..., description="Human-readable reward description")


class ConsumerInstrument(BaseModel):
    """Consumer payment instrument for counter-negotiation."""
    
    instrument_id: str = Field(..., description="Unique instrument identifier")
    instrument_type: InstrumentType = Field(..., description="Type of payment instrument")
    provider: str = Field(..., description="Payment provider name")
    last_four: str = Field(..., description="Last four digits")
    
    # Cost and value metrics
    base_fee: float = Field(..., description="Base processing fee in basis points", ge=0.0)
    out_of_pocket_cost: float = Field(..., description="Out-of-pocket cost to consumer", ge=0.0)
    available_balance: float = Field(..., description="Available balance/credit limit", ge=0.0)
    
    # Rewards and loyalty
    rewards: List[ConsumerReward] = Field(default_factory=list, description="Available rewards")
    total_reward_value: float = Field(..., description="Total reward value for this transaction", ge=0.0)
    loyalty_tier: Optional[str] = Field(None, description="Loyalty tier (gold, platinum, etc.)")
    loyalty_multiplier: float = Field(default=1.0, description="Loyalty tier multiplier", ge=0.0, le=5.0)
    
    # Net value calculation
    net_value: float = Field(..., description="Net value to consumer (rewards - out-of-pocket)", ge=-1000.0, le=1000.0)
    value_score: float = Field(..., description="Normalized value score (0.0-1.0)", ge=0.0, le=1.0)
    
    # Eligibility and preferences
    eligible: bool = Field(..., description="Whether instrument is eligible for this transaction")
    preference_score: float = Field(default=0.5, description="Consumer preference score", ge=0.0, le=1.0)
    
    # Explanation data
    selection_factors: List[str] = Field(default_factory=list, description="Factors favoring this instrument")
    exclusion_reasons: List[str] = Field(default_factory=list, description="Reasons this instrument was not selected")


class MerchantProposal(BaseModel):
    """Merchant rail proposal from Orca."""
    
    rail_type: str = Field(..., description="Proposed payment rail (ACH, Debit, Credit)")
    merchant_cost: float = Field(..., description="Merchant processing cost in basis points")
    settlement_days: int = Field(..., description="Days to settlement", ge=0)
    risk_score: float = Field(..., description="Risk score for this rail", ge=0.0, le=1.0)
    explanation: str = Field(..., description="Merchant explanation for rail selection")
    trace_id: str = Field(..., description="Transaction trace ID")


class CounterNegotiationRequest(BaseModel):
    """Request for consumer counter-negotiation."""
    
    actor_id: str = Field(..., description="Consumer actor ID")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Transaction currency")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    mcc: Optional[str] = Field(None, description="Merchant Category Code")
    channel: str = Field(default="online", description="Transaction channel")
    
    # Merchant proposal to counter
    merchant_proposal: MerchantProposal = Field(..., description="Merchant rail proposal")
    
    # Consumer context
    available_instruments: List[ConsumerInstrument] = Field(..., description="Available consumer instruments")
    consumer_preferences: Dict[str, Any] = Field(default_factory=dict, description="Consumer preferences")
    
    # Negotiation parameters
    reward_weight: float = Field(default=0.5, description="Weight for reward optimization", ge=0.0, le=1.0)
    cost_weight: float = Field(default=0.3, description="Weight for cost minimization", ge=0.0, le=1.0)
    preference_weight: float = Field(default=0.2, description="Weight for consumer preferences", ge=0.0, le=1.0)


class CounterNegotiationResponse(BaseModel):
    """Response from consumer counter-negotiation."""
    
    selected_instrument: ConsumerInstrument = Field(..., description="Selected consumer instrument")
    counter_proposal: Dict[str, Any] = Field(..., description="Consumer counter-proposal")
    explanation: str = Field(..., description="Explanation for instrument selection")
    trace_id: str = Field(..., description="Transaction trace ID")
    timestamp: datetime = Field(..., description="Negotiation timestamp")
    
    # Negotiation metadata
    negotiation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Negotiation metadata")
    rejected_instruments: List[ConsumerInstrument] = Field(default_factory=list, description="Rejected instruments")
    
    # Value analysis
    merchant_savings: float = Field(..., description="Potential merchant cost savings")
    consumer_value: float = Field(..., description="Net value to consumer")
    win_win_score: float = Field(..., description="Overall win-win score", ge=0.0, le=1.0)
