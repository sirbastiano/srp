.PHONY: clean_venv install_pdm pdm_install_toml

clean_venv:
	@echo 'Removing virtual environment...'
	@if [ -d '.venv' ]; then rm -rf .venv; fi

install_pdm:
	@echo 'Installing pdm...'
	python3 -m pip install --upgrade pip
	python3 -m pip install --user pdm

pdm_install_deps:
	@echo 'Installing toml package using pdm...'
	pdm install 

setup: clean_venv install_pdm pdm_install_deps
	@echo 'Setup complete.'