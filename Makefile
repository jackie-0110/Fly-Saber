.PHONY: setup data subgraph maps train-bc train-ppo serve web test lint fmt

setup:
	uv sync
	cd web && npm install

data:
	uv run python scripts/fetch_data.py

subgraph:
	uv run python scripts/build_subgraph.py

maps:
	uv run python scripts/fetch_maps.py

train-bc:
	uv run python -m flyarcade.train.bc

train-ppo:
	uv run python -m flyarcade.train.ppo

serve:
	uv run uvicorn flyarcade.server.app:app --reload --port 8000

web:
	cd web && npm run dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cd web && npm run lint

fmt:
	uv run ruff check --fix .
	uv run ruff format .
