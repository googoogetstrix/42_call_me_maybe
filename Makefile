# Variables
PYTHON := python3
UV := uv
MAIN_SCRIPT := src

.PHONY: install run debug clean lint lint-strict

# Install project dependencies
install:
	$(UV) run sync

# Execute the main script
run:
	$(UV) run $(PYTHON) -m $(MAIN_SCRIPT)

# Run the main script in debug mode using pdb
debug:
	$(PYTHON) -m pdb $(MAIN_SCRIPT)

# Remove temporary files and caches
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Execute standard linting
lint:
	flake8 --exclude .venv,./llm_sdk/ .
	uv run mypy . --exclude '(.venv|llm_sdk)/' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Optional: Execute strict linting
lint-strict:
	flake8 --exclude .venv,./llm_sdk/ .
	uv run mypy . --exclude '(.venv|llm_sdk)/' --strict