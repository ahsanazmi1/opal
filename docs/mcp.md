# Opal MCP Integration

## Overview

The Opal Wallet Agent exposes its capabilities through the Multi-Agent Communication Protocol (MCP), allowing other agents to interact with payment method selection and spend controls.

## MCP Manifest

The MCP manifest is located at `src/opal/mcp/manifest.json` and defines the available tools and resources.

## Available Tools

### listPaymentMethods

List available payment methods for an actor.

**Parameters:**
- `actor_id` (string, required): Actor identifier

**Example:**
```json
{
  "name": "listPaymentMethods",
  "arguments": {
    "actor_id": "user_123"
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Available Payment Methods for user_123:\n[\n  {\n    \"method_id\": \"pm_user_123_visa_001\",\n    \"type\": \"card\",\n    \"provider\": \"visa\",\n    \"last_four\": \"4242\",\n    \"status\": \"active\"\n  }\n]"
    }
  ]
}
```

### selectPaymentMethod

Select a payment method and evaluate against spend controls.

**Parameters:**
- `actor_id` (string, required): Actor identifier
- `payment_method_id` (string, required): Selected payment method ID
- `amount` (number, required): Transaction amount
- `currency` (string, optional): Transaction currency (default: "USD")
- `mcc` (string, optional): Merchant Category Code
- `channel` (string, required): Transaction channel
- `merchant_id` (string, optional): Merchant identifier

**Example:**
```json
{
  "name": "selectPaymentMethod",
  "arguments": {
    "actor_id": "user_123",
    "payment_method_id": "pm_user_123_visa_001",
    "amount": 100.0,
    "currency": "USD",
    "mcc": "5411",
    "channel": "web",
    "merchant_id": "merchant_123"
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Payment Method Selection Result:\n{\n  \"allowed\": true,\n  \"token_reference\": \"tok_user_123_1234567890\",\n  \"reasons\": [\"Transaction approved\"],\n  \"control_version\": \"v1.0.0\"\n}"
    }
  ]
}
```

### getControlLimits

Get current spend control limits and parameters.

**Parameters:** None

**Example:**
```json
{
  "name": "getControlLimits",
  "arguments": {}
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Current Spend Control Limits:\n{\n  \"control_version\": \"v1.0.0\",\n  \"mcc_limits\": {\n    \"5411\": {\n      \"max_amount\": 2000.0,\n      \"description\": \"Grocery stores\"\n    }\n  }\n}"
    }
  ]
}
```

## Available Resources

### opal://controls

Current spend control limits and parameters.

**URI:** `opal://controls`
**MIME Type:** `application/json`

**Example:**
```json
{
  "control_version": "v1.0.0",
  "mcc_limits": {
    "5411": {
      "max_amount": 2000.0,
      "description": "Grocery stores"
    }
  },
  "channel_limits": {
    "web": {
      "max_amount": 5000.0,
      "description": "Web transactions"
    }
  },
  "daily_limits": {
    "web": 15000.0
  },
  "method_limits": {
    "card": {
      "max_amount": 10000.0,
      "description": "Card payments"
    }
  }
}
```

## MCP Server Implementation

The MCP server is implemented in `src/opal/mcp/server.py` and provides:

- **stdio transport**: Standard input/output communication
- **Tool handling**: Processes tool calls and returns results
- **Resource management**: Serves control limits resource
- **Error handling**: Graceful error responses

## Usage Examples

### Using with MCP Client

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "opal.mcp.server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Call listPaymentMethods
            result = await session.call_tool(
                "listPaymentMethods",
                {"actor_id": "user_123"}
            )
            print(f"Payment methods: {result.content[0].text}")
            
            # Call selectPaymentMethod
            result = await session.call_tool(
                "selectPaymentMethod",
                {
                    "actor_id": "user_123",
                    "payment_method_id": "pm_user_123_visa_001",
                    "amount": 100.0,
                    "channel": "web"
                }
            )
            print(f"Selection result: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

### Tool Call Errors

- **Missing required arguments**: Returns error with missing parameter details
- **Invalid payment method**: Returns error if payment method not available
- **System errors**: Returns generic error message

### Resource Errors

- **Unknown resource**: Returns error for unknown resource URIs
- **Read errors**: Returns error if resource cannot be read

## Integration Notes

- The MCP server runs as a separate process
- Communication is via stdio (standard input/output)
- All responses are JSON-formatted text content
- Error handling is consistent with FastAPI responses
- CloudEvents are emitted for successful payment method selections

## Development

To run the MCP server directly:

```bash
python -m opal.mcp.server
```

To test MCP integration:

```bash
# Install MCP client
pip install mcp

# Run the example client
python examples/mcp_client.py
```
