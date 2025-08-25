.PHONY: help install clean run-notebook activate setup check-env lint format test
.DEFAULT_GOAL := help

# Variables
VENV_PATH := .venv
PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip
JUPYTER := $(VENV_PATH)/bin/jupyter
STREAMLIT := $(VENV_PATH)/bin/streamlit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Complete project setup (create venv, install dependencies, setup kernel)
	@echo "🚀 Setting up BYU Pathway Questions Topic Modeling project..."
	@$(MAKE) install
	@$(MAKE) setup-kernel
	@$(MAKE) check-env
	@echo "✅ Setup complete! Run 'make activate' to activate the virtual environment."

install: ## Create virtual environment and install dependencies
	@echo "📦 Creating virtual environment..."
	python3 -m venv $(VENV_PATH)
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

setup-kernel: ## Setup Jupyter kernel for the virtual environment
	@echo "🔧 Setting up Jupyter kernel..."
	$(PYTHON) -m ipykernel install --user --name=pathway-questions --display-name="Python (Pathway Questions)"
	@echo "✅ Jupyter kernel 'pathway-questions' created!"

activate: ## Show command to activate virtual environment
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_PATH)/bin/activate"

run-notebook: ## Start Jupyter notebook server
	@echo "🚀 Starting Jupyter notebook server..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(JUPYTER) notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

run-lab: ## Start JupyterLab server
	@echo "🚀 Starting JupyterLab server..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(JUPYTER) lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

streamlit: ## Run Streamlit app (when ready)
	@echo "🚀 Starting Streamlit app..."
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	@if [ ! -f "streamlit_app.py" ]; then echo "❌ streamlit_app.py not found. Create it first."; exit 1; fi
	$(STREAMLIT) run streamlit_app.py

check-env: ## Check if environment variables are set
	@echo "🔍 Checking environment setup..."
	@if [ -f ".env" ]; then \
		echo "✅ .env file found"; \
		if grep -q "OPENAI_API_KEY" .env; then \
			echo "✅ OPENAI_API_KEY found in .env"; \
		else \
			echo "⚠️  OPENAI_API_KEY not found in .env file"; \
		fi \
	else \
		echo "⚠️  .env file not found. Create one with your OPENAI_API_KEY"; \
	fi

create-env: ## Create .env template file
	@echo "📝 Creating .env template..."
	@if [ ! -f ".env" ]; then \
		echo "# OpenAI API Configuration" > .env; \
		echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env; \
		echo "✅ .env template created. Please edit it with your actual API key."; \
	else \
		echo "⚠️  .env file already exists. Not overwriting."; \
	fi

clean: ## Remove virtual environment and cache files
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV_PATH)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"

lint: ## Run linting checks
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

format: ## Format code with black
	@if [ ! -d "$(VENV_PATH)" ]; then echo "❌ Virtual environment not found. Run 'make install' first."; exit 1; fi
	$(PYTHON) -m black .

test: ## Run tests (when available)
	@echo "🧪 Running tests..."
	@echo "No tests configured yet."

info: ## Show project information
	@echo "📊 BYU Pathway Questions Topic Modeling Project"
	@echo "=============================================="
	@echo "Python Version: $$(python3 --version)"
	@echo "Virtual Environment: $(VENV_PATH)"
	@echo "Main Notebook: Text_Clustering_&_Topic_Modeling_of_Pathway_Questions_with_OpenAI_and_BERTopic.ipynb"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup          # Complete project setup"
	@echo "  2. make create-env      # Create .env template"
	@echo "  3. Edit .env with your OpenAI API key"
	@echo "  4. make activate       # Get activation command"
	@echo "  5. make run-notebook   # Start Jupyter notebook"

update: ## Update dependencies
	@echo "📦 Updating dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated!"

freeze: ## Generate current dependencies list
	@echo "📋 Current dependencies:"
	$(PIP) freeze > requirements-freeze.txt
	@echo "✅ Dependencies saved to requirements-freeze.txt"
