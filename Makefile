.PHONY: test pre-commit

test:
	poetry run pytest --cov --cov-report=term --cov-report=html

pre-commit:
	poetry run pre-commit run --all-files
