# ai-text-toolkit Makefile
# Cross-platform build commands (works on Windows with Make for Windows or WSL)

.PHONY: help install install-dev train serve test lint format typecheck clean

# Default target
help:
	@echo "ai-text-toolkit - Local AI text detector + humanizer"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  train        Train the detector (downloads HC3 data)"
	@echo "  serve        Start API server (uvicorn)"
	@echo "  frontend     Serve frontend (python http.server)"
	@echo "  test         Run tests"
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with black"
	@echo "  typecheck    Run mypy type checker"
	@echo "  clean        Remove cache and build artifacts"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Train the detector
train:
	python -m core.detector.training

# Train with custom parameters
train-fast:
	python -m core.detector.training --n 100 --test 50

# Start API server
serve:
	python -m api.main

# Serve frontend (in another terminal)
frontend:
	python -m http.server 3000 --directory web

# Run all tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=core --cov=api --cov-report=term-missing

# Run linter
lint:
	ruff check .

# Auto-fix linting issues
lint-fix:
	ruff check . --fix

# Format code
format:
	black .

# Check formatting (without changes)
format-check:
	black --check .

# Type checking
typecheck:
	mypy core api

# Clean cache and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

# Full development setup
setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make train' to train the detector"
	@echo "Run 'make serve' to start the API server"
	@echo "Run 'make frontend' to serve the frontend"