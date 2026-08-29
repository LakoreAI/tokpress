.PHONY: help install dev test lint lint-fix format format-check pre-commit clean check

VENV ?= .venv
BIN := $(shell [ -d $(VENV)/bin ] && echo $(VENV)/bin/ || echo "")

help:
	@echo "TokPress development commands:"
	@echo "  make install       Install package in editable mode"
	@echo "  make dev           Install dev dependencies and pre-commit hooks"
	@echo "  make test          Run pytest suite"
	@echo "  make lint          Run ruff linter"
	@echo "  make lint-fix      Run ruff linter with auto-fix"
	@echo "  make format        Format code using ruff"
	@echo "  make format-check  Verify code formatting"
	@echo "  make pre-commit    Run pre-commit hooks on all files"
	@echo "  make check         Run lint, format check, and test suite"
	@echo "  make clean         Remove cache and build artifacts"

install:
	$(BIN)pip install -e .

dev:
	$(BIN)pip install -e ".[dev]"
	$(BIN)pre-commit install

test:
	$(BIN)pytest tests/

lint:
	$(BIN)ruff check .

lint-fix:
	$(BIN)ruff check --fix .

format:
	$(BIN)ruff format .

format-check:
	$(BIN)ruff format --check .

pre-commit:
	$(BIN)pre-commit run --all-files

check: lint format-check test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
