.PHONY: install run headless bootstrap-gazebo lint format lint-fix lint-report

install:
	poetry install
	poetry run pip install -r requirements-ui.txt

run:
	poetry run python -m cais_spade_llm

headless:
	poetry run python -m cais_spade_llm.ui_main --headless

bootstrap-gazebo:
	bash scripts/bootstrap_gazebo_workspace.sh

# --- Code quality (ruff) ---

# Report every lint finding — the full debt dashboard.
lint:
	poetry run ruff check .

# Format the codebase in place.
format:
	poetry run ruff format .

# One-button cleanup: apply safe auto-fixes, then format. Review the diff.
lint-fix:
	poetry run ruff check --fix .
	poetry run ruff format .

# Per-rule counts — the burn-down scoreboard for the cleanup roadmap.
lint-report:
	poetry run ruff check . --statistics
