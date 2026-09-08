.PHONY: help install dev test test-quick lint lint-fix format format-check pre-commit clean check paper paper-clean bench bench-encodings regression

VENV ?= .venv
BIN := $(shell [ -d $(VENV)/bin ] && echo $(VENV)/bin/ || echo "")

help:
	@echo "TokPress development commands:"
	@echo "  make install       Install package in editable mode"
	@echo "  make dev           Install dev dependencies and pre-commit hooks"
	@echo "  make test          Run the full pytest suite"
	@echo "  make test-quick    Run the fast pytest subset (excludes @slow tests)"
	@echo "  make lint          Run ruff linter"
	@echo "  make lint-fix      Run ruff linter with auto-fix"
	@echo "  make format        Format code using ruff"
	@echo "  make format-check  Verify code formatting"
	@echo "  make pre-commit    Run pre-commit hooks on all files"
	@echo "  make check         Run lint, format check, and test-quick suite"
	@echo "  make bench         Run the full benchmark harness (needs data/bench corpora)"
	@echo "  make bench-encodings  Compare tiktoken encodings + domain byte-BPE through the codec (needs data/bench corpora)"
	@echo "  make regression    Run the self-contained ratio regression gate (scripts/bench_regression.py)"
	@echo "  make paper         Rebuild docs/tokpress.pdf from docs/tokpress.tex (needs latexmk)"
	@echo "  make paper-clean   Remove LaTeX build by-products"
	@echo "  make clean         Remove cache and build artifacts"

install:
	$(BIN)pip install -e .

dev:
	$(BIN)pip install -e ".[dev]"
	$(BIN)pre-commit install

test:
	$(BIN)pytest tests/

test-quick:
	$(BIN)pytest tests/ -m "not slow" -q

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

check: lint format-check test-quick

bench:
	$(BIN)python scripts/bench.py

bench-encodings:
	$(BIN)python scripts/bench_encodings.py

regression:
	$(BIN)python scripts/bench_regression.py

paper:
	cd docs && latexmk -pdf -interaction=nonstopmode tokpress.tex

paper-clean:
	cd docs && latexmk -c tokpress.tex

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
