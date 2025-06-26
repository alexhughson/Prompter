.PHONY: format type test test-quick test-all clean help

help:
	@echo "Available commands:"
	@echo "  make format      - Format code with black and isort"
	@echo "  make type        - Run mypy type checker"
	@echo "  make test        - Run all tests with pytest"
	@echo "  make test-quick  - Run tests quickly (no coverage)"
	@echo "  make test-all    - Run tests with all LLM providers"
	@echo "  make check       - Run format, type, and test"
	@echo "  make clean       - Remove cache files"

format:
	@echo "Running black..."
	@black prompter tests
	@echo "Running isort..."
	@isort prompter tests

type:
	@echo "Running mypy..."
	@mypy prompter

test:
	@echo "Running tests..."
	@python -m pytest tests/ -v

test-quick:
	@echo "Running quick tests..."
	@python -m pytest tests/ -q

test-all:
	@echo "Running tests with all providers..."
	@python -m pytest tests/ -v --tb=short

check: format type test
	@echo "All checks passed!"

clean:
	@echo "Cleaning cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*~" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cache cleaned!"