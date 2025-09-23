import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from fastapi import FastAPI
from mcp.server import router as mcp_router


# Create a test FastAPI app and include the MCP router
@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(mcp_router)
    with TestClient(app) as c:
        yield c


def test_mcp_get_status(client):
    """Test MCP getStatus verb returns agent status."""
    response = client.post("/mcp/invoke", json={"verb": "getStatus", "args": {}})
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["data"]["agent"] == "opal"
    assert data["data"]["status"] == "active"


def test_mcp_list_payment_methods(client):
    """Test MCP listPaymentMethods verb returns deterministic stub payment methods."""
    response = client.post("/mcp/invoke", json={"verb": "listPaymentMethods", "args": {}})
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["data"]["agent"] == "opal"
    assert "payment_methods" in data["data"]
    assert len(data["data"]["payment_methods"]) == 3

    # Check specific payment methods
    payment_methods = data["data"]["payment_methods"]
    assert payment_methods[0]["method_id"] == "pm_stub_credit_1234"
    assert payment_methods[0]["type"] == "credit_card"
    assert payment_methods[0]["provider"] == "Visa"
    assert payment_methods[0]["last_four"] == "1234"

    assert payment_methods[1]["method_id"] == "pm_stub_debit_5678"
    assert payment_methods[1]["type"] == "debit_card"
    assert payment_methods[1]["provider"] == "Mastercard"
    assert payment_methods[1]["last_four"] == "5678"

    assert payment_methods[2]["method_id"] == "pm_stub_wallet_9999"
    assert payment_methods[2]["type"] == "digital_wallet"
    assert payment_methods[2]["provider"] == "PayPal"
    assert payment_methods[2]["last_four"] == "9999"


def test_mcp_unsupported_verb(client):
    """Test MCP with unsupported verb returns error."""
    response = client.post("/mcp/invoke", json={"verb": "unsupportedVerb", "args": {}})
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "Unsupported verb" in data["error"]


def test_mcp_missing_verb(client):
    """Test MCP with missing verb returns validation error."""
    response = client.post("/mcp/invoke", json={"args": {}})
    assert response.status_code == 422  # FastAPI validation error
    data = response.json()
    assert "detail" in data
    assert "Field required" in data["detail"][0]["msg"]


def test_mcp_invalid_json(client):
    """Test MCP with invalid JSON returns error."""
    response = client.post("/mcp/invoke", data="invalid json")
    assert response.status_code == 422  # FastAPI validation error
