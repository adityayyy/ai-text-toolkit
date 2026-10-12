# Contributing to ai-text-toolkit

Thank you for your interest in contributing! This project is a learning-focused AI text detector + humanizer.

## Development Setup

```bash
# Clone and enter
git clone https://github.com/yourusername/ai-text-toolkit.git
cd ai-text-toolkit

# Install development dependencies
pip install -r requirements-dev.txt
pre-commit install

# Train detector (one-time, ~10-20 min)
make train
```

## Code Style

- **Linting**: `ruff` (line length 100, Python 3.10 target)
- **Formatting**: `black` (line length 100)
- **Type checking**: `mypy` (strict mode)
- **Tests**: `pytest` with asyncio support

Run all checks:
```bash
make lint
make format
make typecheck
make test
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run the full check suite: `make lint format typecheck test`
4. Submit a PR with a clear description

## Code Guidelines

- **Business logic in `core/`** - no FastAPI, no web frameworks
- **API layer in `api/`** - thin FastAPI layer only
- **Frontend in `web/`** - vanilla JS, no build step
- **Configuration in `config/`** - Pydantic Settings + constants
- **Tests in `tests/`** - mirror module structure

## Commit Messages

Use conventional commits:
- `feat: add new feature`
- `fix: bug fix`
- `refactor: code restructuring`
- `docs: documentation updates`
- `test: add tests`
- `chore: maintenance`

## Testing

- **Unit tests**: `tests/test_core.py` for business logic
- **Integration tests**: `tests/test_api.py` for FastAPI endpoints
- Run with: `make test` or `pytest -v`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.