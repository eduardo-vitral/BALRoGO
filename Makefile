
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
list:
	@sh -c "$(MAKE) -p no_targets__ | \
		awk -F':' '/^[a-zA-Z0-9][^\$$#\/\\t=]*:([^=]|$$)/ {\
			split(\$$1,A,/ /);for(i in A)print A[i]\
		}' | grep -v '__\$$' | grep -v 'make\[1\]' | grep -v 'Makefile' | sort"
no_targets__:

setup:
	@./scripts/poetry-wrapper.sh install

test: setup
	@./scripts/poetry-wrapper.sh run pytest --cov=balrogo tests/ -s

format:
	@./scripts/poetry-wrapper.sh run black balrogo/ tests/ samples/ docs/

lint:
	@./scripts/poetry-wrapper.sh run flake8 --count --statistics balrogo/ tests/ samples/ docs/

build: clean
	@./scripts/poetry-wrapper.sh build

publish: build
	@./scripts/poetry-wrapper.sh publish

sample:
	@./scripts/poetry-wrapper.sh run python samples/sample.py

docs:
	@./scripts/poetry-wrapper.sh run $(MAKE) -C docs html

# https://raw.githubusercontent.com/python-poetry/poetry/master/Makefile
clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

.PHONY: setup test format lint clean sample docs