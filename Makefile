.PHONY: up down build logs restart shell

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
