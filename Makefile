.PHONY: install data test lint bench serve clean

install:
	pip install -e ".[dev]"

data:
	python -m ctr.cli gen-data --data-dir data

test:
	pytest -q

lint:
	ruff check .

bench: data
	python -m ctr.cli bench --data-dir data --out bench

serve:
	uvicorn ctr.service:app --reload

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info
