PYTHON ?= python3

.PHONY: contracts format lint typecheck test test-data release-safety release-parity quality pre-commit

contracts:
	$(PYTHON) scripts/check_docs_sync.py
	sam validate --lint --template-file template.yaml

format:
	$(PYTHON) -m black app tests scripts
	$(PYTHON) -m isort app tests scripts

lint:
	$(PYTHON) -m ruff check app tests scripts
	$(PYTHON) -m black --check app tests scripts
	$(PYTHON) -m isort --check-only app tests scripts

typecheck:
	$(PYTHON) -m mypy app

test:
	$(PYTHON) -m pytest

test-data:
	$(PYTHON) -m pytest tests/test_prepare_dataset.py -rs

release-safety:
	$(PYTHON) -m app.ml.release_gates safety --output-json release-safety.json

release-parity:
	$(PYTHON) -m app.ml.release_gates parity --output-json release-parity.json

quality: lint typecheck contracts test

pre-commit:
	$(PYTHON) -m pre_commit run --all-files
