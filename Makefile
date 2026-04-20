.PHONY: up down build logs restart shell migration-up migration-down migration-create

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose up --build

logs:
	docker compose logs -f api

restart:
	docker compose restart api

shell:
	docker compose exec api bash

migration-up:
	docker compose exec api alembic upgrade head

migration-down:
	docker compose exec api alembic downgrade -1

migration-create:
	docker compose exec api alembic revision -m "$(name)"
