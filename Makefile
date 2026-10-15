.PHONY: all data run examples test clean

all: run examples test

data:
	python scripts/download_data.py

run:
	python scripts/run_all.py

examples:
	python scripts/error_examples.py

test:
	python -m pytest -q

clean:
	rm -rf results/tables results/figures results/error_examples results/participant_metrics_itt.csv
