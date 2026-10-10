# Common commands. On Windows, run these inside WSL2 (Ubuntu) or call the `uv run ...` lines directly.
.PHONY: setup fmt lint typecheck test check cov app refit clean

setup:            ## install all dependencies and git hooks
	uv sync --all-extras
	uv run pre-commit install

fmt:              ## auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

lint:             ## lint without changing files
	uv run ruff format --check .
	uv run ruff check .

typecheck:        ## static type checking
	uv run mypy src

test:             ## run unit tests (no network)
	uv run pytest -m "not network"

check: lint typecheck test   ## everything CI runs. Must pass before every commit.

cov:              ## tests with coverage report
	uv run pytest -m "not network" --cov=radar --cov-report=term-missing

app:              ## start the dashboard
	uv run streamlit run src/radar/report/app.py

# Retrain everything on the newest complete data (a date is complete 12 months after it).
# Rewrites config/scoring_v2.yaml (learned weights), the live models and the walk-forward marts.
# Long: ~2 h on the full data. Run it when new quarter-ends have completed, then review
# docs/results.md before trusting the new numbers.
REFIT_START ?= 2022-09-30
REFIT_END ?= $(shell date +%Y-%m-%d)
refit:            ## retrain on the newest data: backtest → model → weights → walk-forward
	uv run radar backtest --start $(REFIT_START) --end $(REFIT_END)
	uv run radar ablation
	uv run radar model
	uv run radar learn-weights
	uv run radar walkforward
	uv run radar weight-stability

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
