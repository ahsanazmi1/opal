"""
CloudEvents emitter for Opal payment method selections.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel

from .controls import PaymentMethod, TransactionRequest, SpendControlResult, CounterNegotiationResponse, ConsumerInstrument


class MethodSelectedEvent(BaseModel):
    """CloudEvent for payment method selections."""

    specversion: str = "1.0"
    id: str
    source: str
    type: str = "ocn.opal.method_selected.v1"
    subject: Optional[str] = None
    time: str
    datacontenttype: str = "application/json"
    dataschema: Optional[str] = None
    data: Dict[str, Any]


class MethodSelectedData(BaseModel):
    """Data payload for method selected events."""

    actor_id: str
    payment_method: Dict[str, Any]
    transaction_request: Dict[str, Any]
    control_result: Dict[str, Any]
    timestamp: str


async def emit_method_selected_event(
    actor_id: str,
    payment_method: PaymentMethod,
    transaction_request: TransactionRequest,
    control_result: SpendControlResult,
    source: str = "https://opal.ocn.ai/v1",
) -> MethodSelectedEvent:
    """
    Emit a CloudEvent for payment method selection.

    Args:
        actor_id: Actor identifier
        payment_method: Selected payment method
        transaction_request: Transaction request details
        control_result: Spend control evaluation result
        source: Event source URI

    Returns:
        CloudEvent object (in production, this would be sent to an event bus)
    """
    # Convert objects to dict for serialization
    payment_method_dict = {
        "method_id": payment_method.method_id,
        "type": payment_method.type,
        "provider": payment_method.provider,
        "last_four": payment_method.last_four,
        "expiry_month": payment_method.expiry_month,
        "expiry_year": payment_method.expiry_year,
        "status": payment_method.status,
        "metadata": payment_method.metadata,
    }

    transaction_request_dict = {
        "amount": float(transaction_request.amount),
        "currency": transaction_request.currency,
        "mcc": transaction_request.mcc,
        "channel": transaction_request.channel,
        "merchant_id": transaction_request.merchant_id,
        "actor_id": transaction_request.actor_id,
        "payment_method_id": transaction_request.payment_method_id,
    }

    control_result_dict = {
        "allowed": control_result.allowed,
        "token_reference": control_result.token_reference,
        "reasons": control_result.reasons,
        "limits_applied": control_result.limits_applied,
        "max_amount_allowed": (
            float(control_result.max_amount_allowed) if control_result.max_amount_allowed else None
        ),
        "control_version": control_result.control_version,
    }

    # Create event data
    event_data = MethodSelectedData(
        actor_id=actor_id,
        payment_method=payment_method_dict,
        transaction_request=transaction_request_dict,
        control_result=control_result_dict,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Create CloudEvent
    event = MethodSelectedEvent(
        id=str(uuid4()),
        source=source,
        subject=actor_id,  # Use actor_id as subject
        time=datetime.now(timezone.utc).isoformat(),
        data=event_data.model_dump(),
    )

    # In production, this would send the event to an event bus
    # For now, we'll just log it
    print(f"Method Selected Event: {event.model_dump_json()}")

    return event


# Phase 3 - Consumer Counter-Negotiation CloudEvents

class ConsumerExplanationEvent(BaseModel):
    """CloudEvent for consumer counter-negotiation explanations."""
    
    specversion: str = "1.0"
    id: str
    source: str
    type: str = "ocn.opal.explanation.v1"
    subject: Optional[str] = None
    time: str
    datacontenttype: str = "application/json"
    dataschema: Optional[str] = None
    data: Dict[str, Any]


class ConsumerExplanationData(BaseModel):
    """Data payload for consumer explanation events."""
    
    actor_id: str
    selected_instrument: Dict[str, Any]
    counter_proposal: Dict[str, Any]
    explanation: str
    negotiation_metadata: Dict[str, Any]
    timestamp: str


async def emit_consumer_explanation_event(
    negotiation_response: CounterNegotiationResponse,
    actor_id: str,
    source: str = "https://opal.ocn.ai/negotiation"
) -> ConsumerExplanationEvent:
    """
    Emit a CloudEvent for consumer counter-negotiation explanation.
    
    Args:
        negotiation_response: Response from counter-negotiation
        actor_id: Consumer actor identifier
        source: Event source URI
        
    Returns:
        ConsumerExplanationEvent
    """
    event_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Create event data payload
    event_data = ConsumerExplanationData(
        actor_id=actor_id,
        selected_instrument={
            "instrument_id": negotiation_response.selected_instrument.instrument_id,
            "instrument_type": negotiation_response.selected_instrument.instrument_type,
            "provider": negotiation_response.selected_instrument.provider,
            "last_four": negotiation_response.selected_instrument.last_four,
            "total_reward_value": negotiation_response.selected_instrument.total_reward_value,
            "out_of_pocket_cost": negotiation_response.selected_instrument.out_of_pocket_cost,
            "net_value": negotiation_response.selected_instrument.net_value,
            "value_score": negotiation_response.selected_instrument.value_score,
            "loyalty_tier": negotiation_response.selected_instrument.loyalty_tier,
            "loyalty_multiplier": negotiation_response.selected_instrument.loyalty_multiplier,
            "selection_factors": negotiation_response.selected_instrument.selection_factors,
        },
        counter_proposal=negotiation_response.counter_proposal,
        explanation=negotiation_response.explanation,
        negotiation_metadata=negotiation_response.negotiation_metadata,
        timestamp=negotiation_response.timestamp.isoformat(),
    )
    
    # Create CloudEvent
    event = ConsumerExplanationEvent(
        id=event_id,
        source=source,
        subject=f"actor_{actor_id}",
        time=timestamp,
        dataschema="https://schemas.ocn.ai/events/v1/opal.explanation.v1.schema.json",
        data=event_data.dict(),
    )
    
    # Log the event
    logging.info(
        f"Emitted consumer explanation CloudEvent",
        extra={
            "event_id": event_id,
            "actor_id": actor_id,
            "selected_instrument": negotiation_response.selected_instrument.instrument_type,
            "consumer_value": negotiation_response.consumer_value,
            "event_type": "ocn.opal.explanation.v1"
        }
    )
    
    return event


async def emit_counter_negotiation_event(
    negotiation_response: CounterNegotiationResponse,
    actor_id: str,
    source: str = "https://opal.ocn.ai/negotiation"
) -> ConsumerExplanationEvent:
    """
    Emit a CloudEvent for counter-negotiation decision.
    
    Args:
        negotiation_response: Response from counter-negotiation
        actor_id: Consumer actor identifier
        source: Event source URI
        
    Returns:
        ConsumerExplanationEvent with counter-negotiation details
    """
    event_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Create detailed event data
    event_data = ConsumerExplanationData(
        actor_id=actor_id,
        selected_instrument={
            "instrument_id": negotiation_response.selected_instrument.instrument_id,
            "instrument_type": negotiation_response.selected_instrument.instrument_type,
            "provider": negotiation_response.selected_instrument.provider,
            "last_four": negotiation_response.selected_instrument.last_four,
            "base_fee": negotiation_response.selected_instrument.base_fee,
            "total_reward_value": negotiation_response.selected_instrument.total_reward_value,
            "out_of_pocket_cost": negotiation_response.selected_instrument.out_of_pocket_cost,
            "net_value": negotiation_response.selected_instrument.net_value,
            "value_score": negotiation_response.selected_instrument.value_score,
            "loyalty_tier": negotiation_response.selected_instrument.loyalty_tier,
            "loyalty_multiplier": negotiation_response.selected_instrument.loyalty_multiplier,
            "selection_factors": negotiation_response.selected_instrument.selection_factors,
            "exclusion_reasons": negotiation_response.selected_instrument.exclusion_reasons,
        },
        counter_proposal=negotiation_response.counter_proposal,
        explanation=negotiation_response.explanation,
        negotiation_metadata={
            **negotiation_response.negotiation_metadata,
            "merchant_savings": negotiation_response.merchant_savings,
            "consumer_value": negotiation_response.consumer_value,
            "win_win_score": negotiation_response.win_win_score,
            "rejected_instruments_count": len(negotiation_response.rejected_instruments),
        },
        timestamp=negotiation_response.timestamp.isoformat(),
    )
    
    # Create CloudEvent with counter-negotiation type
    event = ConsumerExplanationEvent(
        id=event_id,
        source=source,
        subject=f"actor_{actor_id}",
        time=timestamp,
        type="ocn.opal.counter_negotiation.v1",
        dataschema="https://schemas.ocn.ai/events/v1/opal.counter_negotiation.v1.schema.json",
        data=event_data.dict(),
    )
    
    # Log the event
    logging.info(
        f"Emitted counter-negotiation CloudEvent",
        extra={
            "event_id": event_id,
            "actor_id": actor_id,
            "selected_instrument": negotiation_response.selected_instrument.instrument_type,
            "win_win_score": negotiation_response.win_win_score,
            "event_type": "ocn.opal.counter_negotiation.v1"
        }
    )
    
    return event


# Inline schema for method selected events
METHOD_SELECTED_EVENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://schemas.ocn.ai/events/v1/opal.method_selected.v1.schema.json",
    "title": "Opal Method Selected CloudEvent",
    "description": "Schema for Opal payment method selection events, following CloudEvents v1.0 specification.",
    "type": "object",
    "required": ["specversion", "id", "source", "type", "subject", "time", "data"],
    "properties": {
        "specversion": {
            "type": "string",
            "enum": ["1.0"],
            "description": "The version of the CloudEvents specification which the event uses.",
        },
        "id": {"type": "string", "description": "A unique identifier for the event."},
        "source": {
            "type": "string",
            "format": "uri-reference",
            "description": "The context in which the event happened. Often a URI.",
        },
        "type": {
            "type": "string",
            "enum": ["ocn.opal.method_selected.v1"],
            "description": "The type of event, e.g., 'ocn.opal.method_selected.v1'.",
        },
        "subject": {
            "type": "string",
            "description": "The subject of the event in the context of the event producer (e.g., actor_id).",
        },
        "time": {
            "type": "string",
            "format": "date-time",
            "description": "The time when the event occurred as a UTC ISO 8601 timestamp.",
        },
        "datacontenttype": {"type": "string", "description": "Content type of the data attribute."},
        "dataschema": {
            "type": ["string", "null"],
            "description": "A link to the schema that the data attribute adheres to.",
        },
        "data": {
            "type": "object",
            "description": "The method selection payload.",
            "required": [
                "actor_id",
                "payment_method",
                "transaction_request",
                "control_result",
                "timestamp",
            ],
            "properties": {
                "actor_id": {"type": "string", "description": "Actor identifier"},
                "payment_method": {
                    "type": "object",
                    "description": "Selected payment method details",
                    "required": ["method_id", "type", "provider", "last_four", "status"],
                    "properties": {
                        "method_id": {"type": "string"},
                        "type": {"type": "string"},
                        "provider": {"type": "string"},
                        "last_four": {"type": "string"},
                        "expiry_month": {"type": ["integer", "null"]},
                        "expiry_year": {"type": ["integer", "null"]},
                        "status": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                },
                "transaction_request": {
                    "type": "object",
                    "description": "Transaction request details",
                    "required": ["amount", "currency", "channel", "actor_id", "payment_method_id"],
                    "properties": {
                        "amount": {"type": "number"},
                        "currency": {"type": "string"},
                        "mcc": {"type": ["string", "null"]},
                        "channel": {"type": "string"},
                        "merchant_id": {"type": ["string", "null"]},
                        "actor_id": {"type": "string"},
                        "payment_method_id": {"type": "string"},
                    },
                },
                "control_result": {
                    "type": "object",
                    "description": "Spend control evaluation result",
                    "required": ["allowed", "reasons", "limits_applied", "control_version"],
                    "properties": {
                        "allowed": {"type": "boolean"},
                        "token_reference": {"type": ["string", "null"]},
                        "reasons": {"type": "array", "items": {"type": "string"}},
                        "limits_applied": {"type": "array", "items": {"type": "string"}},
                        "max_amount_allowed": {"type": ["number", "null"]},
                        "control_version": {"type": "string"},
                    },
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Event timestamp",
                },
            },
        },
    },
}
