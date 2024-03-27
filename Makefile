SHELL := /bin/bash
.PHONY: lint typecheck format test docs
.SILENT: lint typecheck format test docs
.NOTPARALLEL: lint typecheck format test docs
.ONESHELL:

all: lint typecheck

lint: fix := false
lint:
ifeq ($(fix), true)
	poetry run ruff check --fix
else
	poetry run ruff check
endif

typecheck:
	poetry run mypy striker

format:
	poetry run ruff format

test:
	@echo 'TESTS ARE CURRENTLY BROKEN'
	exit 1
	poetry run pytest

docs:
	@echo 'NOT YET IMPLEMENTED'
	exit 1
