# Variables
PYTHON := python3
UV := uv
MODULE := src
MAIN_SCRIPT := src/__main__.py

.PHONY: install run debug clean lint lint-strict

# Install project dependencies
install:
	$(UV) run sync

# Execute the main script
run:
	$(UV) run $(PYTHON) -m $(MODULE)

run-timed:
	@echo "Execution started at: $$(date)"
	@time $(MAKE) run
	@echo "Execution finished at: $$(date)"

# Run the main script in debug mode using pdb
# # $(UV) run $(PYTHON) -m pdb $(MAIN_SCRIPT)
debug:
	$(UV) run $(PYTHON) -m pdb -m src

test:
	$(UV) run pytest --pdb

# Remove temporary files and caches
clean:
	rm -rf __pycache__
	rm -rf $(MODULE)/__pycache__
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