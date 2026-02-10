# ──────────────────────────────────────────────────────────────
# sarpyx – Makefile
# ──────────────────────────────────────────────────────────────

DOCKER_IMAGE   ?= sirbastiano94/sarpyx
DOCKER_TAG     ?= latest
DOCKER_FULL    := $(DOCKER_IMAGE):$(DOCKER_TAG)
PLATFORM       ?= linux/amd64

.PHONY: clean_venv install_pdm pdm_install_deps install_phidown install_snap \
        setup prune_docker up_recreate up \
        docker-build docker-test docker-push docker-all

# ──────────────────────────  Python / PDM  ──────────────────────────

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

setup: clean_venv install_pdm pdm_install_deps
	@echo 'Setup complete.'

# ──────────────────────────  SNAP  ──────────────────────────────────

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

# ──────────────────────────  Docker  ───────────────────────────────

docker-build:
	@echo "Building Docker image $(DOCKER_FULL) ..."
	docker build --platform $(PLATFORM) -t $(DOCKER_FULL) .

docker-test: docker-build
	@echo "Running Docker build tests …"
	docker run --rm $(DOCKER_FULL) sh -c "\
		python3.11 -m pip install pytest && \
		python3.11 -m pytest /workspace/tests/test_docker.py -v --tb=short"

docker-push: docker-build
	@echo "Pushing $(DOCKER_FULL) …"
	docker push $(DOCKER_FULL)

docker-all: docker-build docker-test docker-push

prune_docker:
	@echo 'Pruning Docker system...'
	docker system prune -a

up_recreate:
	docker compose up --build --force-recreate

up:
	docker compose up