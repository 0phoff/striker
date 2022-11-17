SHELL := /bin/bash
.PHONY: lint test docs
.SILENT: lint test docs
.NOTPARALLEL: lint test docs
.ONESHELL:

all: lint test docs


lint:
	poetry run flake8
	poetry run mypy striker

test:
	poetry run pytest

docs:
	@echo 'NOT YET IMPLEMENTED'
	exit 1
