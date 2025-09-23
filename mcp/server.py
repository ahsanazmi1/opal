from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class MCPRequest(BaseModel):
    """MCP request model."""

    verb: str
    args: Dict[str, Any] = {}


class MCPResponse(BaseModel):
    """MCP response model."""

    ok: bool
    data: Any = None
    error: Any = None


@router.post("/mcp/invoke", response_model=MCPResponse)
async def invoke_mcp_verb(request: MCPRequest) -> MCPResponse:
    """
    Handle MCP protocol requests.

    Supported verbs:
    - getStatus: Returns agent status
    - listPaymentMethods: Returns deterministic stub payment methods
    """
    try:
        if request.verb == "getStatus":
            return MCPResponse(ok=True, data={"agent": "opal", "status": "active"})
        elif request.verb == "listPaymentMethods":
            # Return deterministic stub payment methods
            return MCPResponse(
                ok=True,
                data={
                    "agent": "opal",
                    "payment_methods": [
                        {
                            "method_id": "pm_stub_credit_1234",
                            "type": "credit_card",
                            "provider": "Visa",
                            "last_four": "1234",
                            "expiry_month": 12,
                            "expiry_year": 2025,
                            "status": "active",
                            "description": "Visa Credit Card ending in 1234",
                        },
                        {
                            "method_id": "pm_stub_debit_5678",
                            "type": "debit_card",
                            "provider": "Mastercard",
                            "last_four": "5678",
                            "expiry_month": 6,
                            "expiry_year": 2026,
                            "status": "active",
                            "description": "Mastercard Debit Card ending in 5678",
                        },
                        {
                            "method_id": "pm_stub_wallet_9999",
                            "type": "digital_wallet",
                            "provider": "PayPal",
                            "last_four": "9999",
                            "expiry_month": None,
                            "expiry_year": None,
                            "status": "active",
                            "description": "PayPal Digital Wallet",
                        },
                    ],
                    "description": "Deterministic stub payment methods for testing",
                },
            )
        else:
            return MCPResponse(ok=False, error=f"Unsupported verb: {request.verb}")
    except Exception as e:
        return MCPResponse(ok=False, error=str(e))
