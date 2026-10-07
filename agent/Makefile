# Makefile for AWS GPU Agent

.PHONY: help install test test-coverage test-watch clean lint format

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	python -m pytest tests/ -v --watch

test-unit: ## Run only unit tests
	python -m pytest tests/ -v -m unit

test-integration: ## Run only integration tests
	python -m pytest tests/ -v -m integration

test-aws: ## Run AWS-related tests
	python -m pytest tests/ -v -m aws

test-slow: ## Run slow tests
	python -m pytest tests/ -v -m slow

lint: ## Run linting
	flake8 src/ tests/
	pylint src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

setup-dev: ## Setup development environment
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pre-commit install

run: ## Run the agent
	python main.py

run-dev: ## Run the agent in development mode
	LOG_LEVEL=DEBUG python main.py

docker-build: ## Build Docker image
	docker build -t aws-gpu-agent .

docker-run: ## Run Docker container
	docker run --env-file .env aws-gpu-agent

docker-test: ## Run tests in Docker
	docker run --rm -v $(PWD):/app -w /app python:3.9 bash -c "pip install -r requirements.txt && python -m pytest tests/"

ci-test: ## Run tests for CI
	python -m pytest tests/ --cov=src --cov-report=xml --cov-fail-under=80 --junitxml=test-results.xml
