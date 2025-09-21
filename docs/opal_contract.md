# Opal Wallet Agent Contract

## Overview

The Opal Wallet Agent provides payment method selection and spend controls for the OCN ecosystem. It implements deterministic spend controls based on Merchant Category Codes (MCC), transaction channels, and payment method types.

## API Endpoints

### GET /wallet/methods

List available payment methods for an actor.

**Parameters:**
- `actor_id` (query): Actor identifier

**Response:**
```json
[
  {
    "method_id": "pm_123_visa_001",
    "type": "card",
    "provider": "visa",
    "last_four": "4242",
    "expiry_month": 12,
    "expiry_year": 2025,
    "status": "active",
    "metadata": {
      "card_type": "credit",
      "network": "visa"
    }
  }
]
```

### POST /wallet/select

Select a payment method and evaluate against spend controls.

**Request Body:**
```json
{
  "actor_id": "user_123",
  "payment_method_id": "pm_123_visa_001",
  "amount": 100.0,
  "currency": "USD",
  "mcc": "5411",
  "channel": "web",
  "merchant_id": "merchant_123"
}
```

**Response:**
```json
{
  "allowed": true,
  "token_reference": "tok_user_123_1234567890",
  "reasons": [
    "Amount $100.0 within MCC 5411 limit of $2000.0",
    "Amount $100.0 within web channel limit of $5000.0"
  ],
  "limits_applied": [
    "MCC 5411 limit: $2000.0",
    "Channel web limit: $5000.0"
  ],
  "max_amount_allowed": 2000.0,
  "control_version": "v1.0.0"
}
```

### GET /controls/limits

Get current spend control limits and parameters.

**Response:**
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

## Spend Controls

### MCC Limits

| MCC | Category | Limit | Description |
|-----|----------|-------|-------------|
| 5411 | Grocery Stores | $2,000 | Low-risk category |
| 5812 | Restaurants | $1,000 | Medium-risk category |
| 5999 | Miscellaneous Retail | $100 | High-risk category |
| 7995 | Gambling | $50 | Restricted category |

### Channel Limits

| Channel | Limit | Daily Limit | Description |
|---------|-------|-------------|-------------|
| web | $5,000 | $15,000 | Web transactions |
| mobile | $3,000 | $10,000 | Mobile app transactions |
| pos | $10,000 | $25,000 | Point of sale transactions |
| atm | $500 | $2,000 | ATM transactions |
| api | $10,000 | $50,000 | API transactions |

### Payment Method Limits

| Type | Limit | Description |
|------|-------|-------------|
| card | $10,000 | Card payments |
| bank | $25,000 | Bank transfers |
| wallet | $5,000 | Digital wallet |
| crypto | $1,000 | Cryptocurrency |

## CloudEvents

### Method Selected Event

**Type:** `ocn.opal.method_selected.v1`

**Schema:** Available at `https://schemas.ocn.ai/events/v1/opal.method_selected.v1.schema.json`

**Data Payload:**
```json
{
  "actor_id": "user_123",
  "payment_method": {
    "method_id": "pm_123_visa_001",
    "type": "card",
    "provider": "visa",
    "last_four": "4242",
    "status": "active"
  },
  "transaction_request": {
    "amount": 100.0,
    "currency": "USD",
    "mcc": "5411",
    "channel": "web"
  },
  "control_result": {
    "allowed": true,
    "token_reference": "tok_user_123_1234567890",
    "reasons": ["Transaction approved"]
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Error Handling

### 400 Bad Request
- Invalid payment method for actor
- Missing required fields

### 404 Not Found
- Payment method not found

### 422 Validation Error
- Invalid amount (negative or zero)
- Missing required parameters

### 500 Internal Server Error
- System errors during processing

## Security

- All transactions are evaluated against spend controls
- Token references are deterministic and traceable
- No sensitive payment data is stored or logged
- CloudEvents provide audit trail for all selections

## Versioning

- API version: `v0.1.0`
- Control version: `v1.0.0`
- Schema version: `v1.0.0`
