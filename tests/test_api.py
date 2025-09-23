"""
Integration tests for FastAPI service.
"""

import pytest
from fastapi.testclient import TestClient

from opal.api import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestRootEndpoints:
    """Test root endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "Opal Wallet Agent"
        assert data["version"] == "0.1.0"
        assert data["status"] == "operational"
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "opal-wallet-agent"


class TestControlLimits:
    """Test control limits endpoint."""

    def test_get_control_limits(self, client):
        """Test getting control limits."""
        response = client.get("/controls/limits")
        assert response.status_code == 200

        data = response.json()
        assert "control_version" in data
        assert "mcc_limits" in data
        assert "channel_limits" in data
        assert "daily_limits" in data
        assert "method_limits" in data


class TestPaymentMethods:
    """Test payment methods endpoints."""

    def test_list_payment_methods(self, client):
        """Test listing payment methods."""
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 4  # Should return 4 stubbed methods

        for method in data:
            assert "method_id" in method
            assert "type" in method
            assert "provider" in method
            assert "last_four" in method
            assert "status" in method

    def test_get_payment_method(self, client):
        """Test getting specific payment method."""
        # First get the list to find a method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        # Get specific method
        response = client.get(f"/wallet/methods/{method_id}?actor_id=test_user_123")
        assert response.status_code == 200

        data = response.json()
        assert data["method_id"] == method_id

    def test_get_nonexistent_payment_method(self, client):
        """Test getting nonexistent payment method."""
        response = client.get("/wallet/methods/nonexistent?actor_id=test_user_123")
        assert response.status_code == 404

        data = response.json()
        assert "not found" in data["detail"]


class TestPaymentMethodSelection:
    """Test payment method selection."""

    def test_approved_selection(self, client):
        """Test approved payment method selection."""
        # First get a payment method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        # Select payment method
        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": method_id,
            "amount": 50.0,
            "currency": "USD",
            "mcc": "5411",  # Grocery stores - low risk
            "channel": "web",
            "merchant_id": "merchant_123",
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["allowed"] is True
        assert data["token_reference"] is not None
        assert data["token_reference"].startswith("tok_test_user_123_")
        assert len(data["reasons"]) > 0
        assert data["control_version"] == "v1.0.0"

    def test_declined_selection_high_risk_mcc(self, client):
        """Test declined selection for high-risk MCC."""
        # First get a payment method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        # Select payment method with high-risk MCC
        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": method_id,
            "amount": 150.0,  # Exceeds MCC 5999 limit
            "currency": "USD",
            "mcc": "5999",  # Miscellaneous retail - high risk
            "channel": "web",
            "merchant_id": "merchant_123",
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["allowed"] is False
        assert data["token_reference"] is None
        assert "exceeds MCC 5999 limit" in " ".join(data["reasons"])

    def test_declined_selection_channel_limit(self, client):
        """Test declined selection for channel limit."""
        # First get a payment method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        # Select payment method exceeding channel limit
        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": method_id,
            "amount": 6000.0,  # Exceeds web channel limit
            "currency": "USD",
            "channel": "web",  # Web limit is $5000
            "merchant_id": "merchant_123",
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["allowed"] is False
        assert data["token_reference"] is None
        assert "exceeds web channel limit" in " ".join(data["reasons"])

    def test_invalid_payment_method(self, client):
        """Test selection with invalid payment method."""
        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": "nonexistent_method",
            "amount": 50.0,
            "currency": "USD",
            "channel": "web",
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 400

        data = response.json()
        assert "not available" in data["detail"]

    def test_missing_required_fields(self, client):
        """Test selection with missing required fields."""
        request_data = {
            "actor_id": "test_user_123",
            # Missing payment_method_id, amount, channel
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_amount(self, client):
        """Test selection with invalid amount."""
        # First get a payment method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": method_id,
            "amount": -10.0,  # Invalid negative amount
            "currency": "USD",
            "channel": "web",
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_different_channels(self, client):
        """Test selection with different channels."""
        # First get a payment method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        # Test mobile channel
        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": method_id,
            "amount": 2500.0,  # Within mobile limit
            "currency": "USD",
            "channel": "mobile",
        }

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["allowed"] is True

        # Test POS channel with higher limit
        request_data["channel"] = "pos"
        request_data["amount"] = 8000.0  # Within POS limit

        response = client.post("/wallet/select", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["allowed"] is True

    def test_deterministic_results(self, client):
        """Test that same inputs produce deterministic results."""
        # First get a payment method ID
        response = client.get("/wallet/methods?actor_id=test_user_123")
        assert response.status_code == 200

        methods = response.json()
        method_id = methods[0]["method_id"]

        request_data = {
            "actor_id": "test_user_123",
            "payment_method_id": method_id,
            "amount": 100.0,
            "currency": "USD",
            "mcc": "5411",
            "channel": "web",
        }

        # Make the same request twice
        response1 = client.post("/wallet/select", json=request_data)
        response2 = client.post("/wallet/select", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Token references should be the same (deterministic)
        assert data1["token_reference"] == data2["token_reference"]
        assert data1["allowed"] == data2["allowed"]
