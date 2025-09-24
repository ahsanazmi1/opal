# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Phase 2 — Explainability work

## Phase 2 — Explainability Checklist
<!-- Complete this section if this PR is for Phase 2 work -->

### Tests
- [ ] Unit tests added/updated for explainability features
- [ ] Integration tests for LLM integration
- [ ] Edge case testing for payment method explanation generation
- [ ] Performance tests for explanation API endpoints

### Contracts & Validation
- [ ] New API endpoints follow OCN contract standards
- [ ] CloudEvents schemas updated for explanation events
- [ ] Contract validation tests pass
- [ ] Backward compatibility maintained

### CloudEvents (CE) Validation
- [ ] New `ocn.opal.explanation.v1` event schema implemented
- [ ] CE events include proper trace_id propagation
- [ ] Event validation against ocn-common schemas
- [ ] Integration with existing CE infrastructure

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Type hints added for new functions
- [ ] Documentation updated for new features
- [ ] Security review completed for LLM integration

### Explainability Features
- [ ] AI-powered payment method selection explanation generation implemented
- [ ] Human-readable spend control decision reasoning
- [ ] Integration with Azure OpenAI
- [ ] Decision audit trail with explanations
- [ ] Performance optimization for explanation generation
- [ ] Enhanced MCP verbs for explainability

## Testing
<!-- Describe the tests that you ran to verify your changes -->
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Performance testing completed (if applicable)

## Documentation
<!-- List any documentation changes -->
- [ ] README.md updated (if applicable)
- [ ] API documentation updated
- [ ] Code comments added/updated
- [ ] CHANGELOG.md updated

## Security Considerations
<!-- List any security implications or considerations -->
- [ ] No sensitive data exposed in explanations
- [ ] LLM API keys properly secured
- [ ] Input validation for explanation requests
- [ ] Rate limiting implemented for explanation endpoints

## Performance Impact
<!-- Describe any performance implications -->
- [ ] No significant performance degradation
- [ ] Explanation generation optimized
- [ ] Caching implemented where appropriate
- [ ] Resource usage documented

## Breaking Changes
<!-- List any breaking changes and migration steps -->
- [ ] No breaking changes
- [ ] Breaking changes documented with migration guide
- [ ] Version bump appropriate for breaking changes

## Related Issues
<!-- Link to related issues using "Fixes #123" or "Related to #123" -->
- Fixes #
- Related to #

## Additional Notes
<!-- Add any additional information that reviewers should know -->

---

## Reviewer Checklist
- [ ] Code review completed
- [ ] Tests pass in CI
- [ ] Documentation is clear and complete
- [ ] Security considerations addressed
- [ ] Performance impact acceptable
- [ ] Breaking changes properly documented
