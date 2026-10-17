.PHONY: setup data subgraph export-web maps training-maps train-bc train-ppo serve frontend test test-python test-frontend lint fmt build check docker-build

setup:
	uv sync --locked --dev
	cd frontend && npm ci

data:
	uv run python scripts/fetch_data.py

subgraph:
	uv run python scripts/build_subgraph.py

export-web:
	uv run python scripts/export_web.py

maps:
	uv run python scripts/fetch_maps.py

training-maps:
	uv run python scripts/fetch_training_maps.py

train-bc:
	uv run python -m training.bc

train-ppo:
	uv run python -m training.ppo

serve:
	uv run uvicorn server.app:serve --factory --reload --port 8000

frontend:
	cd frontend && npm run dev

test: test-python test-frontend

test-python:
	uv run --locked pytest

test-frontend: build
	cd frontend && npm test

build:
	cd frontend && npm run build

check: lint test

docker-build:
	docker build -f docker/Dockerfile -t fly-saber:local .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	cd frontend && npm run lint
	cd frontend && npm run format:check

fmt:
	uv run ruff check --fix .
	uv run ruff format .
	cd frontend && npm run format
