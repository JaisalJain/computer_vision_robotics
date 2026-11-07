# Computer vision for robotics - developer tasks.
# Run inside the activated environment (conda `cv` or a venv); override with `make PYTHON=...`.

SHELL       := /bin/bash
PYTHON      ?= python
PIP         := $(PYTHON) -m pip
CONSTRAINTS := requirements/constraints.txt
TORCH_INDEX := https://download.pytorch.org/whl/cu128
GPU_LOCK    ?= /tmp/cv-gpu.lock
KERNEL      ?= cv
NB_TIMEOUT  ?= 1800

.DEFAULT_GOAL := help
.PHONY: help setup check-env lint format test verify verify-static clean

help: ## list targets
	@grep -hE '^[a-zA-Z0-9_%-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## install the core stack + the cvlab package (editable)
	$(PIP) install --extra-index-url $(TORCH_INDEX) -c $(CONSTRAINTS) -r requirements/base.txt
	$(PIP) install -c $(CONSTRAINTS) -e .
	$(PYTHON) scripts/check_env.py

setup-%: ## install one project's extras, e.g. make setup-03_object_detection
	$(PIP) install --extra-index-url $(TORCH_INDEX) -c $(CONSTRAINTS) -r requirements/$*.txt
	$(PYTHON) scripts/check_env.py

check-env: ## verify torch/CUDA/sm_120, pip check and key imports
	$(PYTHON) scripts/check_env.py

lint: ## ruff (python files + notebooks)
	$(PYTHON) -m ruff check .

format: ## auto-format and auto-fix with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

test: ## CPU unit tests
	$(PYTHON) -m pytest -q

verify: ## full repository verification (lint, tests, notebooks, READMEs, sizes, links)
	$(PYTHON) scripts/verify_repo.py

verify-static: ## verification without tests (CI)
	$(PYTHON) scripts/verify_repo.py --static

data-%: ## download a project's data, e.g. make data-08_monocular_depth
	$(PYTHON) scripts/download_$*.py

nb-%: ## execute a project's notebooks headless on the GPU lock, e.g. make nb-01_classical_cv
	@set -e; for nb in projects/$*/notebooks/*.ipynb; do \
	  echo ">> executing $$nb"; \
	  flock $(GPU_LOCK) $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.kernel_name=$(KERNEL) --ExecutePreprocessor.timeout=$(NB_TIMEOUT) "$$nb"; \
	  $(PYTHON) scripts/nb_clean.py "$$nb"; \
	done

clean: ## remove caches (never touches data/ or weights/)
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
