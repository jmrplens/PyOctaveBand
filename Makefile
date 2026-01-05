install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check .
	mypy src

format:
	ruff check --fix .
	ruff format .

security:
	bandit -r src

snyk:
	@echo "Running Snyk..."
	@if [ -f .env ]; then export $$(cat .env | xargs) && pnpm exec snyk test --all-projects; else pnpm exec snyk test --all-projects; fi

sonar:
	@echo "Running SonarQube Scanner..."
	@if [ -f .env ]; then export $$(cat .env | xargs) && pnpm exec sonar-scanner; else pnpm exec sonar-scanner; fi

test:
	python tests/test_basic.py

	python tests/test_multichannel.py
	python tests/test_audio_processing.py

check: lint security test
