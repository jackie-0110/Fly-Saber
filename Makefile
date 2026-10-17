.PHONY: setup data subgraph export-web maps train-bc train-ppo serve frontend test lint fmt

setup:
	uv sync
	cd frontend && npm install

data:
	uv run python scripts/fetch_data.py

subgraph:
	uv run python scripts/build_subgraph.py

export-web:
	uv run python scripts/export_web.py

maps:
	uv run python scripts/fetch_maps.py

train-bc:
	uv run python -m training.bc

train-ppo:
	uv run python -m training.ppo

serve:
	uv run uvicorn server.app:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cd frontend && npm run lint

fmt:
	uv run ruff check --fix .
	uv run ruff format .
