# Opal — The Open Payment Agent

[![Contracts Validation](https://github.com/ocn-ai/opal/actions/workflows/contracts.yml/badge.svg)](https://github.com/ocn-ai/opal/actions/workflows/contracts.yml)
[![Security Validation](https://github.com/ocn-ai/opal/actions/workflows/security.yml/badge.svg)](https://github.com/ocn-ai/opal/actions/workflows/security.yml)
[![CI](https://github.com/ocn-ai/opal/actions/workflows/ci.yml/badge.svg)](https://github.com/ocn-ai/opal/actions/workflows/ci.yml)

**Opal** is the **open, transparent payment agent** for the [Open Checkout Network (OCN)](https://github.com/ocn-ai/ocn-common).

## Purpose

Opal provides intelligent payment method selection and spend controls for the OCN ecosystem. Unlike traditional black-box payment systems, Opal offers:

- **Transparent Payment Logic** - Clear, auditable payment method selection
- **Spend Controls** - Configurable spending limits and restrictions
- **Open Architecture** - Integrates seamlessly with OCN protocols
- **MCP Integration** - Model Context Protocol support for agent interactions

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ocn-ai/opal.git
cd opal

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -U pip && pip install -e .[dev]

# Run tests
pytest -q

# Start the service
uvicorn opal.api:app --reload
```

## Related OCN Repositories

- [Orca](https://github.com/ocn-ai/orca): The Open Checkout Agent
- [Okra](https://github.com/ocn-ai/okra): The Open Credit Agent
- [Onyx](https://github.com/ocn-ai/onyx): The Open Trust Registry
- [Oasis](https://github.com/ocn-ai/oasis): The Open Treasury Agent
- [Orion](https://github.com/ocn-ai/orion): The Open Payout Agent
- [Weave](https://github.com/ocn-ai/weave): The Open Receipt Ledger
- [ocn-common](https://github.com/ocn-ai/ocn-common): Common utilities and schemas
