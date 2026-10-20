.PHONY: install lint test ingest train coach dashboard api clean-disk

install:
	pip install -r requirements.txt

lint:
	ruff check src tests
	black --check src tests

test: lint
	pytest -q

ingest:
	python -m src.ingest --session "$(RACE)"

coach:
	python -m src.telemetry --session "$(SESSION)" --driver $(DRIVER) --reference $(REF)

train:
	python -m src.predictor

dashboard:
	PYTHONPATH=. python -m streamlit run src/dashboard.py

api:
	uvicorn src.api:app --reload

clean-disk:
	bash scripts/cleanup_disk.sh
