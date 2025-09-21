"""
FastAPI service for Opal wallet operations.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .controls import SpendControls, TransactionRequest, PaymentMethod, SpendControlResult
from .events import emit_method_selected_event


app = FastAPI(
    title="Opal Wallet Agent",
    description="Open Wallet Agent providing payment method selection and spend controls",
    version="0.1.0"
)


# Pydantic models for API
class PaymentMethodResponse(BaseModel):
    """Payment method response."""
    
    method_id: str = Field(..., description="Unique payment method identifier")
    type: str = Field(..., description="Payment method type")
    provider: str = Field(..., description="Payment provider")
    last_four: str = Field(..., description="Last four digits")
    expiry_month: Optional[int] = Field(None, description="Expiry month")
    expiry_year: Optional[int] = Field(None, description="Expiry year")
    status: str = Field(..., description="Payment method status")
    metadata: Dict[str, str] = Field(..., description="Additional metadata")


class SelectPaymentMethodRequest(BaseModel):
    """Request to select a payment method."""
    
    actor_id: str = Field(..., description="Actor/user identifier")
    payment_method_id: str = Field(..., description="Selected payment method ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field("USD", description="Transaction currency")
    mcc: Optional[str] = Field(None, description="Merchant Category Code")
    channel: str = Field(..., description="Transaction channel")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")


class SelectPaymentMethodResponse(BaseModel):
    """Response for payment method selection."""
    
    allowed: bool = Field(..., description="Whether transaction is allowed")
    token_reference: Optional[str] = Field(None, description="Token reference if allowed")
    reasons: List[str] = Field(..., description="Reasons for decision")
    limits_applied: List[str] = Field(..., description="Limits that were checked")
    max_amount_allowed: Optional[float] = Field(None, description="Maximum amount allowed")
    control_version: str = Field(..., description="Control version used")


class ControlLimitsResponse(BaseModel):
    """Response for control limits."""
    
    control_version: str = Field(..., description="Control version")
    mcc_limits: Dict[str, Any] = Field(..., description="MCC limits")
    channel_limits: Dict[str, Any] = Field(..., description="Channel limits")
    daily_limits: Dict[str, float] = Field(..., description="Daily limits")
    method_limits: Dict[str, Any] = Field(..., description="Method limits")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Opal Wallet Agent",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": {
            "wallet_methods": "/wallet/methods",
            "wallet_select": "/wallet/select",
            "controls": "/controls/limits",
            "health": "/health"
        }
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "opal-wallet-agent"}


@app.get("/controls/limits", response_model=ControlLimitsResponse)
async def get_control_limits():
    """Get current spend control limits and parameters."""
    limits = SpendControls.get_control_limits()
    return ControlLimitsResponse(**limits)


@app.get("/wallet/methods", response_model=List[PaymentMethodResponse])
async def list_payment_methods(actor_id: str):
    """
    List available payment methods for an actor.
    
    Args:
        actor_id: Actor identifier
        
    Returns:
        List of available payment methods
    """
    try:
        methods = SpendControls.get_available_payment_methods(actor_id)
        
        return [
            PaymentMethodResponse(
                method_id=method.method_id,
                type=method.type,
                provider=method.provider,
                last_four=method.last_four,
                expiry_month=method.expiry_month,
                expiry_year=method.expiry_year,
                status=method.status,
                metadata=method.metadata
            )
            for method in methods
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving payment methods: {str(e)}"
        )


@app.post("/wallet/select", response_model=SelectPaymentMethodResponse, status_code=status.HTTP_200_OK)
async def select_payment_method(request: SelectPaymentMethodRequest):
    """
    Select a payment method and evaluate against spend controls.
    
    This endpoint:
    1. Validates the payment method selection
    2. Evaluates the transaction against spend controls
    3. Returns a token reference if allowed, or denial reasons
    4. Optionally emits a CloudEvent for the selection
    """
    try:
        # Create transaction request
        transaction_request = TransactionRequest(
            amount=request.amount,
            currency=request.currency,
            mcc=request.mcc,
            channel=request.channel,
            merchant_id=request.merchant_id,
            actor_id=request.actor_id,
            payment_method_id=request.payment_method_id
        )
        
        # Get available payment methods to validate selection
        available_methods = SpendControls.get_available_payment_methods(request.actor_id)
        method_ids = [method.method_id for method in available_methods]
        
        if request.payment_method_id not in method_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment method {request.payment_method_id} not available for actor {request.actor_id}"
            )
        
        # Evaluate against spend controls
        control_result = SpendControls.evaluate_transaction(transaction_request)
        
        # Create response
        response = SelectPaymentMethodResponse(
            allowed=control_result.allowed,
            token_reference=control_result.token_reference,
            reasons=control_result.reasons,
            limits_applied=control_result.limits_applied,
            max_amount_allowed=float(control_result.max_amount_allowed) if control_result.max_amount_allowed else None,
            control_version=control_result.control_version
        )
        
        # Emit CloudEvent for method selection (optional)
        try:
            selected_method = next(method for method in available_methods if method.method_id == request.payment_method_id)
            await emit_method_selected_event(
                actor_id=request.actor_id,
                payment_method=selected_method,
                transaction_request=transaction_request,
                control_result=control_result
            )
        except Exception as e:
            # Don't fail the request if event emission fails
            print(f"Warning: Failed to emit method selected event: {e}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing payment method selection: {str(e)}"
        )


@app.get("/wallet/methods/{method_id}", response_model=PaymentMethodResponse)
async def get_payment_method(actor_id: str, method_id: str):
    """
    Get details of a specific payment method.
    
    Args:
        actor_id: Actor identifier
        method_id: Payment method identifier
        
    Returns:
        Payment method details
    """
    try:
        methods = SpendControls.get_available_payment_methods(actor_id)
        
        method = next((m for m in methods if m.method_id == method_id), None)
        if not method:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Payment method {method_id} not found for actor {actor_id}"
            )
        
        return PaymentMethodResponse(
            method_id=method.method_id,
            type=method.type,
            provider=method.provider,
            last_four=method.last_four,
            expiry_month=method.expiry_month,
            expiry_year=method.expiry_year,
            status=method.status,
            metadata=method.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving payment method: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "opal.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
