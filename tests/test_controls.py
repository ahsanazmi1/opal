"""
Tests for spend controls.
"""

import pytest
from decimal import Decimal

from opal.controls import SpendControls, TransactionRequest, PaymentMethod


class TestPaymentMethod:
    """Test PaymentMethod model."""

    def test_valid_payment_method(self):
        """Test valid payment method."""
        method = PaymentMethod(
            method_id="pm_123_visa_001",
            type="card",
            provider="visa",
            last_four="4242",
            expiry_month=12,
            expiry_year=2025,
            status="active",
            metadata={"card_type": "credit"},
        )

        assert method.method_id == "pm_123_visa_001"
        assert method.type == "card"
        assert method.provider == "visa"
        assert method.last_four == "4242"
        assert method.expiry_month == 12
        assert method.expiry_year == 2025
        assert method.status == "active"


class TestTransactionRequest:
    """Test TransactionRequest model."""

    def test_valid_transaction_request(self):
        """Test valid transaction request."""
        request = TransactionRequest(
            amount=Decimal("100.00"),
            currency="USD",
            mcc="5411",
            channel="web",
            merchant_id="merchant_123",
            actor_id="user_456",
            payment_method_id="pm_123_visa_001",
        )

        assert request.amount == Decimal("100.00")
        assert request.currency == "USD"
        assert request.mcc == "5411"
        assert request.channel == "web"
        assert request.merchant_id == "merchant_123"
        assert request.actor_id == "user_456"
        assert request.payment_method_id == "pm_123_visa_001"

    def test_transaction_request_validation(self):
        """Test transaction request validation."""
        # Invalid amount
        with pytest.raises(ValueError):
            TransactionRequest(
                amount=Decimal("-10.00"),  # Negative
                currency="USD",
                channel="web",
                actor_id="user_123",
                payment_method_id="pm_123",
            )


class TestSpendControls:
    """Test spend control evaluation."""

    def test_low_risk_transaction_approved(self):
        """Test approval for low-risk transaction."""
        request = TransactionRequest(
            amount=Decimal("50.00"),
            currency="USD",
            mcc="5411",  # Grocery stores - low risk
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is True
        assert result.token_reference is not None
        assert result.token_reference.startswith("tok_user_123_")
        assert len(result.reasons) > 0
        assert len(result.limits_applied) > 0
        assert result.control_version == "v1.0.0"

    def test_high_risk_mcc_declined(self):
        """Test decline for high-risk MCC."""
        request = TransactionRequest(
            amount=Decimal("150.00"),  # Above MCC limit
            currency="USD",
            mcc="5999",  # Miscellaneous retail - high risk, $100 limit
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is False
        assert result.token_reference is None
        assert "exceeds MCC 5999 limit" in " ".join(result.reasons)

    def test_channel_limit_exceeded(self):
        """Test decline for channel limit exceeded."""
        request = TransactionRequest(
            amount=Decimal("6000.00"),  # Above web channel limit
            currency="USD",
            channel="web",  # Web limit is $5000
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is False
        assert result.token_reference is None
        assert "exceeds web channel limit" in " ".join(result.reasons)

    def test_pos_channel_higher_limit(self):
        """Test POS channel allows higher amounts."""
        request = TransactionRequest(
            amount=Decimal("8000.00"),  # Within POS limit
            currency="USD",
            mcc=None,  # No MCC restrictions
            channel="pos",  # POS limit is $10000
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is True
        assert result.token_reference is not None

    def test_deterministic_token_reference(self):
        """Test that same inputs produce same token reference."""
        request1 = TransactionRequest(
            amount=Decimal("100.00"),
            currency="USD",
            mcc="5411",
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        request2 = TransactionRequest(
            amount=Decimal("100.00"),
            currency="USD",
            mcc="5411",
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result1 = SpendControls.evaluate_transaction(request1)
        result2 = SpendControls.evaluate_transaction(request2)

        assert result1.token_reference == result2.token_reference

    def test_different_inputs_different_tokens(self):
        """Test that different inputs produce different token references."""
        request1 = TransactionRequest(
            amount=Decimal("100.00"),
            currency="USD",
            mcc="5411",
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        request2 = TransactionRequest(
            amount=Decimal("200.00"),  # Different amount
            currency="USD",
            mcc="5411",
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result1 = SpendControls.evaluate_transaction(request1)
        result2 = SpendControls.evaluate_transaction(request2)

        assert result1.token_reference != result2.token_reference

    def test_multiple_limits_applied(self):
        """Test that multiple limits are checked and applied."""
        request = TransactionRequest(
            amount=Decimal("100.00"),
            currency="USD",
            mcc="5411",  # Grocery stores
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is True
        assert len(result.limits_applied) >= 2  # MCC and channel limits
        assert any("MCC 5411 limit" in limit for limit in result.limits_applied)
        assert any("Channel web limit" in limit for limit in result.limits_applied)

    def test_get_available_payment_methods(self):
        """Test getting available payment methods."""
        actor_id = "test_user_123"
        methods = SpendControls.get_available_payment_methods(actor_id)

        assert len(methods) == 4  # Should return 4 stubbed methods

        # Check that all methods have required fields
        for method in methods:
            assert method.method_id.startswith(f"pm_{actor_id}_")
            assert method.type in ["card", "bank", "wallet"]
            assert method.provider is not None
            assert method.last_four is not None
            assert method.status == "active"

    def test_get_control_limits(self):
        """Test getting control limits."""
        limits = SpendControls.get_control_limits()

        assert "control_version" in limits
        assert "mcc_limits" in limits
        assert "channel_limits" in limits
        assert "daily_limits" in limits
        assert "method_limits" in limits

        # Check MCC limits
        mcc_limits = limits["mcc_limits"]
        assert "5411" in mcc_limits  # Grocery stores
        assert "5999" in mcc_limits  # Miscellaneous retail

        # Check channel limits
        channel_limits = limits["channel_limits"]
        assert "web" in channel_limits
        assert "mobile" in channel_limits
        assert "pos" in channel_limits

        # Check daily limits
        daily_limits = limits["daily_limits"]
        assert "web" in daily_limits
        assert "mobile" in daily_limits
        assert "pos" in daily_limits

    def test_gambling_mcc_restricted(self):
        """Test that gambling MCC is properly restricted."""
        request = TransactionRequest(
            amount=Decimal("25.00"),  # Within gambling limit
            currency="USD",
            mcc="7995",  # Gambling - restricted to $50
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is True  # Should be allowed within limit

        # Test exceeding gambling limit
        request.amount = Decimal("75.00")
        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is False
        assert "exceeds MCC 7995 limit" in " ".join(result.reasons)

    def test_restaurant_mcc_limits(self):
        """Test restaurant MCC limits."""
        request = TransactionRequest(
            amount=Decimal("400.00"),  # Exceeds restaurant limit
            currency="USD",
            mcc="5812",  # Restaurants - $300 limit
            channel="web",
            actor_id="user_123",
            payment_method_id="pm_123_visa_001",
        )

        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is False
        assert "exceeds MCC 5812 limit" in " ".join(result.reasons)

        # Test within restaurant limit
        request.amount = Decimal("250.00")
        result = SpendControls.evaluate_transaction(request)

        assert result.allowed is True
