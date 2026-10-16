# UCL — developer shortcuts. Stdlib-only; needs Python 3.10+ (and pytest for `test`).
PYTHON ?= python3
FILE   ?= examples/hello.ucl
# object/AI examples are check-only until the v0.3+ roadmap; skip them when running.
RUNNABLE := $(filter-out examples/neuron.ucl,$(wildcard examples/*.ucl))
export PYTHONPATH := src

.PHONY: help install test run lint examples clean

help:
	@echo "make install          install uclc in editable mode with test extras"
	@echo "make test             run the test suite"
	@echo "make run FILE=...     run a .ucl program (default: $(FILE))"
	@echo "make lint             byte-compile + parse-check every source and example"
	@echo "make examples         run every example program"
	@echo "make clean            remove caches and build artifacts"

install:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m uclc.cli run $(FILE)

lint:
	$(PYTHON) -m compileall -q src tests
	@for f in examples/*.ucl; do $(PYTHON) -m uclc.cli check "$$f" >/dev/null || exit 1; done
	@echo "lint OK"

examples:
	@for f in $(RUNNABLE); do echo "== $$f"; $(PYTHON) -m uclc.cli run "$$f" || exit 1; done

clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
