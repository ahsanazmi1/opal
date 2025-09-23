"""
MCP server for Opal wallet agent.
"""

import asyncio
from typing import Any, Dict

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    TextContent,
    Tool,
)

from ..controls import SpendControls, TransactionRequest
from ..events import emit_method_selected_event


# MCP Server instance
server = Server("opal-wallet-agent")


@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available tools."""
    tools = [
        Tool(
            name="listPaymentMethods",
            description="List available payment methods for an actor",
            inputSchema={
                "type": "object",
                "properties": {"actor_id": {"type": "string", "description": "Actor identifier"}},
                "required": ["actor_id"],
            },
        ),
        Tool(
            name="selectPaymentMethod",
            description="Select a payment method and evaluate against spend controls",
            inputSchema={
                "type": "object",
                "properties": {
                    "actor_id": {"type": "string", "description": "Actor identifier"},
                    "payment_method_id": {
                        "type": "string",
                        "description": "Selected payment method ID",
                    },
                    "amount": {"type": "number", "minimum": 0, "description": "Transaction amount"},
                    "currency": {
                        "type": "string",
                        "description": "Transaction currency",
                        "default": "USD",
                    },
                    "mcc": {"type": "string", "description": "Merchant Category Code (optional)"},
                    "channel": {
                        "type": "string",
                        "description": "Transaction channel (web, mobile, pos, atm, api)",
                    },
                    "merchant_id": {
                        "type": "string",
                        "description": "Merchant identifier (optional)",
                    },
                },
                "required": ["actor_id", "payment_method_id", "amount", "channel"],
            },
        ),
        Tool(
            name="getControlLimits",
            description="Get current spend control limits and parameters",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]
    return ListToolsResult(tools=tools)


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "listPaymentMethods":
            return await handle_list_payment_methods(arguments)
        elif name == "selectPaymentMethod":
            return await handle_select_payment_method(arguments)
        elif name == "getControlLimits":
            return await handle_get_control_limits()
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")], isError=True
            )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")], isError=True
        )


async def handle_list_payment_methods(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle listPaymentMethods tool call."""
    try:
        actor_id = arguments["actor_id"]

        methods = SpendControls.get_available_payment_methods(actor_id)

        # Format response
        methods_list = []
        for method in methods:
            methods_list.append(
                {
                    "method_id": method.method_id,
                    "type": method.type,
                    "provider": method.provider,
                    "last_four": method.last_four,
                    "expiry_month": method.expiry_month,
                    "expiry_year": method.expiry_year,
                    "status": method.status,
                    "metadata": method.metadata,
                }
            )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Available Payment Methods for {actor_id}:\n{json.dumps(methods_list, indent=2)}",
                )
            ]
        )

    except KeyError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Missing required argument: {e}")], isError=True
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error listing payment methods: {str(e)}")],
            isError=True,
        )


async def handle_select_payment_method(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle selectPaymentMethod tool call."""
    try:
        # Extract arguments
        actor_id = arguments["actor_id"]
        payment_method_id = arguments["payment_method_id"]
        amount = arguments["amount"]
        currency = arguments.get("currency", "USD")
        mcc = arguments.get("mcc")
        channel = arguments["channel"]
        merchant_id = arguments.get("merchant_id")

        # Create transaction request
        transaction_request = TransactionRequest(
            amount=amount,
            currency=currency,
            mcc=mcc,
            channel=channel,
            merchant_id=merchant_id,
            actor_id=actor_id,
            payment_method_id=payment_method_id,
        )

        # Get available payment methods to validate selection
        available_methods = SpendControls.get_available_payment_methods(actor_id)
        method_ids = [method.method_id for method in available_methods]

        if payment_method_id not in method_ids:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Payment method {payment_method_id} not available for actor {actor_id}",
                    )
                ],
                isError=True,
            )

        # Evaluate against spend controls
        control_result = SpendControls.evaluate_transaction(transaction_request)

        # Emit CloudEvent (optional)
        try:
            selected_method = next(
                method for method in available_methods if method.method_id == payment_method_id
            )
            await emit_method_selected_event(
                actor_id=actor_id,
                payment_method=selected_method,
                transaction_request=transaction_request,
                control_result=control_result,
            )
        except Exception as e:
            print(f"Warning: Failed to emit method selected event: {e}")

        # Format response
        result = {
            "allowed": control_result.allowed,
            "token_reference": control_result.token_reference,
            "reasons": control_result.reasons,
            "limits_applied": control_result.limits_applied,
            "max_amount_allowed": (
                float(control_result.max_amount_allowed)
                if control_result.max_amount_allowed
                else None
            ),
            "control_version": control_result.control_version,
        }

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Payment Method Selection Result:\n{json.dumps(result, indent=2)}",
                )
            ]
        )

    except KeyError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Missing required argument: {e}")], isError=True
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error selecting payment method: {str(e)}")],
            isError=True,
        )


async def handle_get_control_limits() -> CallToolResult:
    """Handle getControlLimits tool call."""
    try:
        limits = SpendControls.get_control_limits()

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Current Spend Control Limits:\n{json.dumps(limits, indent=2)}",
                )
            ]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error getting control limits: {str(e)}")],
            isError=True,
        )


@server.list_resources()
async def handle_list_resources() -> ListResourcesResult:
    """List available resources."""
    resources = [
        Resource(
            uri="opal://controls",
            name="Spend Controls",
            description="Current spend control limits and parameters",
            mimeType="application/json",
        )
    ]
    return ListResourcesResult(resources=resources)


@server.read_resource()
async def handle_read_resource(uri: str) -> ReadResourceResult:
    """Read a resource."""
    if uri == "opal://controls":
        try:
            limits = SpendControls.get_control_limits()
            content = json.dumps(limits, indent=2)

            return ReadResourceResult(contents=[TextContent(type="text", text=content)])
        except Exception as e:
            return ReadResourceResult(
                contents=[TextContent(type="text", text=f"Error reading controls: {str(e)}")]
            )
    else:
        return ReadResourceResult(
            contents=[TextContent(type="text", text=f"Unknown resource: {uri}")]
        )


async def main():
    """Main MCP server loop."""
    # Run server using stdio
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="opal-wallet-agent",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(), experimental_capabilities={}
                ),
            ),
        )


if __name__ == "__main__":
    import json

    asyncio.run(main())
