.PHONY: check lint format test typecheck all

all: lint typecheck test  ## Run all checks (lint + types + tests)

lint:  ## Run ruff linter with auto-fix
	uv run ruff check --fix src/ tests/

format:  ## Run ruff formatter
	uv run ruff format src/ tests/

test:  ## Run pytest with coverage
	uv run pytest tests/ -v

typecheck:  ## Run ty type checker
	uv run ty check src/

check: format lint typecheck test  ## Full CI pipeline: format, lint, typecheck, test
