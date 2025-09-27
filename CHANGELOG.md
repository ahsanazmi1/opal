# Changelog

All notable changes to this project will be documented in this file.

## v0.3.0 — Phase 3: Negotiation & Live Fee Bidding
- New branch: phase-3-bidding
- Prep for negotiation, bidding, policy DSL, and processor connectors
- README updated with Phase 3 checklist

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 2 — Explainability scaffolding
- PR template for Phase 2 development

## [0.2.0] - 2025-01-25

### 🚀 Phase 2 Complete: Enhanced Channel Selection & Explainability

This release completes Phase 2 development, delivering AI-powered payment method selection explanations, enhanced spend controls, and production-ready infrastructure for transparent payment channel management.

#### Highlights
- **AI-Powered Channel Selection**: Azure OpenAI integration for human-readable payment method selection reasoning
- **Enhanced Spend Controls**: Comprehensive spend control management with explainable decisions
- **Decision Audit Trails**: Complete decision audit trails with explainable reasoning
- **Production Infrastructure**: Robust CI/CD workflows with security scanning
- **MCP Integration**: Enhanced Model Context Protocol verbs for explainability features

#### Core Features
- **Payment Method Selection**: Intelligent channel selection with risk assessment and cost optimization
- **Spend Controls**: Dynamic spend control management with real-time monitoring
- **Decision Engine**: Comprehensive decision engine with explainable reasoning
- **API Endpoints**: RESTful endpoints for channel selection and spend control operations
- **Event Processing**: Advanced event handling and processing capabilities

#### Quality & Infrastructure
- **Test Coverage**: Comprehensive test suite with channel selection and spend control testing
- **Security Hardening**: Enhanced security validation and risk assessment
- **CI/CD Pipeline**: Complete GitHub Actions workflow with security scanning
- **Documentation**: Comprehensive API and contract documentation

### Added
- AI-powered payment method selection explanations with Azure OpenAI integration
- LLM integration for human-readable decision reasoning
- Explainability API endpoints for spend control decisions
- Decision audit trail with explanations
- Enhanced MCP verbs for explainability features
- Comprehensive spend control management
- Intelligent payment channel selection
- Advanced event processing capabilities
- Production-ready CI/CD infrastructure

### Changed
- Enhanced payment method selection with explainable reasoning
- Improved spend controls with dynamic management
- Streamlined MCP integration for better explainability
- Optimized decision engine performance and accuracy

### Deprecated
- None

### Removed
- None

### Fixed
- Resolved MCP manifest validation indentation errors
- Fixed mypy type checking issues
- Resolved security workflow issues
- Enhanced error handling and validation

### Security
- Enhanced security validation for payment channel selection
- Comprehensive risk assessment and mitigation
- Secure API endpoints with proper authentication
- Robust spend control security measures

## [Unreleased] — Phase 2

### Added
- AI-powered payment method selection explanations
- LLM integration for human-readable decision reasoning
- Explainability API endpoints for spend control decisions
- Decision audit trail with explanations
- Integration with Azure OpenAI for explanations
- Enhanced MCP verbs for explainability features

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-09-22

### Added
- Initial project scaffolding for Opal.
- FastAPI application with wallet operations.
- Basic MCP (Model Context Protocol) stubs.
- Spend controls and payment method selection.
- Initial test suite with fixtures.
- Basic CI workflow.
