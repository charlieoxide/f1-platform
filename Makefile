.PHONY: install lint test ingest dashboard api clean-disk

install:
	pip install -r requirements.txt

lint:
	ruff check src tests
	black --check src tests

test: lint
	pytest -q

ingest:
	python -m src.ingest --session "$(RACE)"

dashboard:
	streamlit run src/dashboard.py

api:
	uvicorn src.api:app --reload

clean-disk:
	bash scripts/cleanup_disk.sh
