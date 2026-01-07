.PHONY: clean_venv install_pdm pdm_install_toml install_snap

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

install_phidown:
	@echo 'Installing phi-down...'
	pdm add phidown
 

install_snap:
	@echo 'Installing packages for S1 data processing...'
	sudo apt update
	sudo apt-get install -y libfftw3-dev libtiff5-dev gdal-bin gfortran libgfortran5 jblas git --fix-missing
	@echo 'Downloading SNAP installer...'
	wget https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-12.0.0.sh
	@echo 'Configuring SNAP installation...'
	chmod +x esa-snap_all_linux-12.0.0.sh
	@echo -e "deleteAllSnapEngineDir\$$Boolean=false\ndeleteOnlySnapDesktopDir\$$Boolean=false\nexecuteLauncherWithPythonAction\$$Boolean=false\nforcePython\$$Boolean=false\npythonExecutable=/usr/bin/python\nsys.adminRights\$$Boolean=true\nsys.component.RSTB\$$Boolean=true\nsys.component.S1TBX\$$Boolean=true\nsys.component.S2TBX\$$Boolean=false\nsys.component.S3TBX\$$Boolean=false\nsys.component.SNAP\$$Boolean=true\nsys.installationDir=$$(PWD)/snap\nsys.languageId=en\nsys.programGroupDisabled\$$Boolean=false\nsys.symlinkDir=/usr/local/bin" > snap.varfile
	@echo 'Installing SNAP...'
	./esa-snap_all_linux-12.0.0.sh -q -varfile $(CURDIR)/snap.varfile -dir $(CURDIR)/..
	@echo 'Configuring SNAP memory settings...'
	echo "-Xmx8G" > $(CURDIR)/snap/bin/gpt.vmoptions
	@echo 'SNAP installation complete.'

setup: clean_venv install_pdm pdm_install_deps
	@echo 'Setup complete.'

prune_docker:
	@echo 'Pruning Docker system...'
	docker system prune -a

up_recreate:
	docker compose up --build --force-recreate

up:
	docker compose up