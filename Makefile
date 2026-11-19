PY ?= python3
VENV = .venv
BIN = $(VENV)/bin

.PHONY: setup test data-dry data check clean

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest -q

data-dry:
	$(BIN)/python -m grouppulse.data.ami_download --dry-run

data:
	$(BIN)/python -m grouppulse.data.ami_download --annotations --meetings demo --views Closeup1 Closeup2 Closeup3 Closeup4 room

check:
	$(BIN)/python -m grouppulse.data.check --data data/raw

clean:
	rm -rf $(VENV) .pytest_cache build *.egg-info
