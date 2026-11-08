# Optional shortcuts for Linux/macOS. Everything delegates to tasks.py, which
# is the single source of truth and also works on Windows (no make needed):
#   python tasks.py test
.PHONY: install lint test ingest train coach dashboard api backup clean-disk

install:
	python tasks.py install

lint:
	python tasks.py lint

test:
	python tasks.py test

ingest:
	python tasks.py ingest --session "$(RACE)"

train:
	python tasks.py train

coach:
	python tasks.py coach --session "$(SESSION)" --driver $(DRIVER) --reference $(REF)

dashboard:
	python tasks.py dashboard

api:
	python tasks.py api

backup:
	python tasks.py backup

# Linux only (apt/snap)
clean-disk:
	bash scripts/cleanup_disk.sh
