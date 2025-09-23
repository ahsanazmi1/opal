# Contributing to Opal

We welcome contributions to the Opal project! Please read this guide to understand how you can contribute.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## How to Contribute

### 1. Fork the Repository

Fork the `ocn-ai/opal` repository on GitHub.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/opal.git
cd opal
```

### 3. Create a New Branch

Create a new branch for your feature or bug fix. Use a descriptive name.

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

### 4. Set Up Your Development Environment

We use `pip` and `venv` for dependency management.

```bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

### 5. Make Your Changes

- Write clear, concise code.
- Follow the existing coding style (enforced by `ruff` and `black`).
- Add tests for new features or bug fixes.
- Ensure all tests pass (`pytest -q`).
- Ensure linting and formatting checks pass (`ruff check .`, `black --check .`).
- Run type checks (`mypy src/`).

### 6. Run Tests

Before committing, ensure all tests pass:

```bash
pytest -q
```

### 7. Commit Your Changes

Commit your changes with a clear and descriptive commit message. Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

```bash
git commit -m "feat: Add new wallet feature"
# or
git commit -m "fix: Fix payment method selection bug"
```

### 8. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 9. Create a Pull Request

- Go to your fork on GitHub and open a new pull request to the `ocn-ai/opal` repository's `phase-1-foundations` branch.
- Provide a clear title and description for your pull request, explaining the changes and why they are necessary.
- Link to any relevant issues.

## Style Guides

### Python

- We adhere to `ruff` and `black` for code formatting and linting.
- Type hints are encouraged.
- Docstrings should follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Git Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. Examples:

- `feat: Add new payment method selection`
- `fix: Correct spend control logic`
- `docs: Update README with quickstart`
- `chore: Update dependencies`

## Reporting Bugs

If you find a bug, please open an issue on GitHub with a clear description, steps to reproduce, and expected behavior.

## Feature Requests

For new features, please open an issue on GitHub to discuss the idea before submitting a pull request.

Thank you for contributing to Opal!
