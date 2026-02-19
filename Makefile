.PHONY: lint format test precommit

lint:
	python -m ruff check .

format:
	python -m ruff format .
	python -m black .

test:
	python -m pytest -q

precommit:
	python -m pre_commit install
	python -m pre_commit run --all-files
