# XAI-SDN Makefile
# Usage: make help

.PHONY: help install install-dev test test-cov lint format type-check \
        train train-synthetic evaluate baselines ablation smoke-test \
        api dashboard docker-build docker-up docker-down clean

PYTHON   := python
PYTEST   := pytest
UVICORN  := uvicorn
BLACK    := black
ISORT    := isort
FLAKE8   := flake8
MYPY     := mypy

SRC_DIRS := features model explainability api sdn
TEST_DIR := tests

## ─── Help ─────────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

## ─── Installation ─────────────────────────────────────────────────────────────

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install all dependencies including dev/test tools
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

## ─── Testing ──────────────────────────────────────────────────────────────────

test:  ## Run full test suite
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-cov:  ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) -v --cov=$(shell echo $(SRC_DIRS) | tr ' ' ',') \
		--cov-report=term-missing --cov-report=html --tb=short

test-entropy:  ## Run entropy-specific tests
	$(PYTEST) $(TEST_DIR)/test_entropy.py -v

test-model:  ## Run model-specific tests
	$(PYTEST) $(TEST_DIR)/test_model.py -v

test-api:  ## Run API tests
	$(PYTEST) $(TEST_DIR)/test_api.py -v --asyncio-mode=auto

smoke-test:  ## Run smoke test suite (no API server needed)
	$(PYTHON) scripts/smoke_test.py --skip-api

smoke-test-full:  ## Run full smoke test suite (starts API)
	$(PYTHON) scripts/smoke_test.py

## ─── Code Quality ─────────────────────────────────────────────────────────────

lint:  ## Lint with flake8
	$(FLAKE8) $(SRC_DIRS) --max-line-length=100 --ignore=E501,W503

format:  ## Format with black + isort
	$(BLACK) $(SRC_DIRS) $(TEST_DIR) scripts --line-length=100
	$(ISORT) $(SRC_DIRS) $(TEST_DIR) scripts --profile=black

format-check:  ## Check formatting without applying
	$(BLACK) --check $(SRC_DIRS) --line-length=100
	$(ISORT) --check $(SRC_DIRS) --profile=black

type-check:  ## Type check with mypy
	$(MYPY) $(SRC_DIRS) --ignore-missing-imports

## ─── ML Pipeline ──────────────────────────────────────────────────────────────

generate-data:  ## Generate synthetic dataset
	$(PYTHON) scripts/generate_synthetic_data.py

train:  ## Train RF model (requires real data in data/raw/)
	$(PYTHON) model/train.py --config configs/model_config.yaml

train-synthetic:  ## Train RF model on synthetic data (no dataset needed)
	$(PYTHON) model/train.py --config configs/model_config.yaml --use-synthetic

evaluate:  ## Evaluate trained model
	$(PYTHON) model/evaluate.py --artifacts-dir model/artifacts --use-synthetic

evaluate-real:  ## Evaluate on real CIC-DDoS2019 data
	$(PYTHON) model/evaluate.py --artifacts-dir model/artifacts --data-dir data/raw --run-shap

baselines:  ## Run baseline comparison
	$(PYTHON) model/baselines.py --use-synthetic

ablation:  ## Run ablation study
	$(PYTHON) model/ablation.py --use-synthetic

shap-global:  ## Compute global SHAP importance
	$(PYTHON) model/evaluate.py --artifacts-dir model/artifacts --use-synthetic --run-shap

## ─── Services ─────────────────────────────────────────────────────────────────

api:  ## Start FastAPI development server
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:  ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

## ─── Docker ───────────────────────────────────────────────────────────────────

docker-build:  ## Build Docker images
	docker compose build

docker-up:  ## Start all services
	docker compose up -d
	@echo "Services started:"
	@echo "  API:       http://localhost:8000/docs"
	@echo "  Dashboard: http://localhost:8501"
	@echo "  Prometheus:http://localhost:9090"
	@echo "  Grafana:   http://localhost:3000"

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## Tail all service logs
	docker compose logs -f

docker-train:  ## Run training job in Docker
	docker compose --profile train run --rm trainer

## ─── Setup ────────────────────────────────────────────────────────────────────

setup:  ## Full development setup
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/synthetic data/splits model/artifacts logs
	$(PYTHON) scripts/generate_synthetic_data.py
	$(PYTHON) model/train.py --use-synthetic
	@echo "\nSetup complete. Run 'make api' to start the API server."

## ─── Cleanup ──────────────────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage .mypy_cache dist build *.egg-info

clean-data:  ## Remove generated data (keeps raw data)
	rm -rf data/synthetic data/processed data/splits

clean-model:  ## Remove trained model artifacts
	rm -rf model/artifacts/*.pkl model/artifacts/*.json model/artifacts/*.csv
