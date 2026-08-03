# Justfile for retinoto_py

# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    uv run --python=3.13 --extra test ruff format .
    uv run --python=3.13 --extra test ruff check . --fix
    uv run --python=3.13 --extra test ruff check --select I --fix .
    uv run --python=3.13 --extra test ty check .
    uv run --python=3.13 --extra test pytest .

# Run all the tests for all the supported Python versions
testall:
    uv run --python=3.10 --extra test pytest
    uv run --python=3.11 --extra test pytest
    uv run --python=3.12 --extra test pytest
    uv run --python=3.13 --extra test pytest

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.13 --extra test pytest {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    uv run --python=3.13  --extra test pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb {{ARGS}}

# Run coverage, and build to HTML
coverage:
    uv run --python=3.13 --extra test coverage run -m pytest .
    uv run --python=3.13 --extra test coverage report -m
    uv run --python=3.13 --extra test coverage html

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    uv build

# Install dependencies needed locally to reproduce RTD's exact doc build (sphinx + nbsphinx)
doc-install-deps:
    @echo "Installing Sphinx + nbsphinx deps..."
    uv pip install sphinx myst-nb nbsphinx ruff[python]

# Build HTML docs locally (same command RTD would run)
doc-build-html:
    @echo "Building documentation to _build/html/ ..."
    sphinx-build -b html . _build/html

# Clean the doc build outputs
doc-clean:
    rm -rf _build

# Run all document commands in sequence - clean, then install deps, then build (same as RTD does)
doc:
    @echo "Reproducing RTD's docs build locally..."
    $(MAKE) doc-clean
    $(MAKE) doc-install-deps
    $(MAKE) doc-build-html

# Run HTML linkcheck without rebuilding - useful for verifying internal links
doc-linkcheck:
    sphinx-build -b linkcheck . _build/linkcheck

VERSION := `grep -m1 '^version' pyproject.toml | sed -E 's/version = "(.*)"/\1/'`

# Print the current version of the project
version:
    @echo "Current version is {{VERSION}}"

# Tag the current version in git and put to github
tag:
    echo "Tagging version v{{VERSION}}"
    git tag -a v{{VERSION}} -m "Creating version v{{VERSION}}"
    git push origin v{{VERSION}}

# remove all build, test, coverage and Python artifacts
clean: 
	clean-build
	clean-pyc
	clean-test

# remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# remove test and coverage artifacts
clean-test:
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache