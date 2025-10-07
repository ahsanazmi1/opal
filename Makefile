.PHONY: help setup lint fmt test run clean install-hooks

# Default target
help:
	@echo "Available targets:"
	@echo "  setup      - Create venv, install deps + dev extras, install pre-commit hooks"
	@echo "  lint       - Run ruff and black checks"
	@echo "  fmt        - Format code with black"
	@echo "  test       - Run pytest with coverage"
	@echo "  run        - Start FastAPI app with uvicorn (if app exists)"
	@echo "  clean      - Remove virtual environment and cache files"
	@echo "  install-hooks - Install pre-commit hooks"

# Setup development environment
setup:
	@echo "🔧 Setting up Opal development environment..."
	@python -m venv .venv
	@. .venv/bin/activate && pip install -U pip
	@. .venv/bin/activate && pip install -e ".[dev]"
	@. .venv/bin/activate && pre-commit install
	@echo "✅ Setup complete! Activate with: source .venv/bin/activate"

# Install pre-commit hooks
install-hooks:
	@echo "🔗 Installing pre-commit hooks..."
	@. .venv/bin/activate && pre-commit install
	@echo "✅ Pre-commit hooks installed!"

# Lint code
lint:
	@echo "🔍 Running linting checks..."
	@. .venv/bin/activate && ruff check .
	@. .venv/bin/activate && black --check .
	@echo "✅ Linting passed!"

# Format code
fmt:
	@echo "🎨 Formatting code..."
	@. .venv/bin/activate && black .
	@echo "✅ Code formatted!"

# Run tests with coverage
test:
	@echo "🧪 Running tests with coverage..."
	@. .venv/bin/activate && pip install pytest-cov
	@. .venv/bin/activate && pytest --cov=src --cov-report=term-missing --cov-fail-under=60
	@echo "✅ Tests passed!"

# Run FastAPI app (if it exists)
run:
	@echo "🚀 Starting Opal FastAPI app..."
	@if [ -f "src/opal/api.py" ]; then \
		echo "Found FastAPI app at src/opal/api.py"; \
		. .venv/bin/activate && uvicorn opal.api:app --reload --port 8000; \
	else \
		echo "❌ No FastAPI app found at src/opal/api.py"; \
		echo "Available Python files:"; \
		find src/ -name "*.py" -type f | head -10; \
		exit 1; \
	fi

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf .venv
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Quick development workflow
dev: fmt lint test
	@echo "✅ Development checks complete!"

# CI workflow (used by GitHub Actions)
ci: lint test
	@echo "✅ CI checks complete!"
