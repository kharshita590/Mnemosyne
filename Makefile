.PHONY: up down migrate test lint

up:
	docker compose up -d

down:
	docker compose down

migrate:
	alembic upgrade head

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	mypy mnemosyne/
