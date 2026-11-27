.PHONY: install dev test lint demo bench dashboard docker

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[all,dev]"

test:
	pytest

lint:
	ruff check src tests

demo:
	atsa demo

bench:
	atsa bench

dashboard:
	atsa dashboard

docker:
	docker compose up -d --build
