# Suggested Commands for OmniMemory Development

## Development Commands
```bash
# Setup and install dependencies
poetry install
poetry run pre-commit install

# Code quality and formatting
poetry run black .               # Format code
poetry run isort .               # Sort imports  
poetry run mypy src/            # Type checking
poetry run flake8 src/          # Linting

# Testing
poetry run pytest              # Run all tests
poetry run pytest -m unit     # Unit tests only
poetry run pytest -m integration  # Integration tests
poetry run pytest --cov       # Coverage report

# Task completion validation
poetry run black . && poetry run isort . && poetry run mypy src/ && poetry run pytest

# Migration and validation
python validate_foundation.py      # ONEX compliance validation
python scripts/migrate_intelligence.py  # Legacy tool migration
```

## System Commands (Darwin)
```bash
# File operations
ls -la                     # List files with details
find . -name "*.py" -type f  # Find Python files
grep -r "pattern" src/     # Search in source code
git status                 # Check git status
git log --oneline -10      # Recent commits
```

## Environment Setup
```bash
# Database setup
export DATABASE_URL="postgresql://..."
export REDIS_URL="redis://localhost:6379"
export PINECONE_API_KEY="..."
```