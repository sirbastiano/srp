# sarpyx Makefile

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------- Config ----------
SUDO ?= sudo
DOCKER ?= $(SUDO) docker
COMPOSE_BAKE ?= false
COMPOSE_FILE ?= docker-compose.yml
COMPOSE ?= COMPOSE_BAKE=$(COMPOSE_BAKE) $(DOCKER) compose -f $(COMPOSE_FILE)

DOCKER_IMAGE ?= sirbastiano94/sarpyx
DOCKER_TAG ?= latest
DOCKER_FULL := $(DOCKER_IMAGE):$(DOCKER_TAG)
PLATFORM ?= linux/amd64

SIF ?= sarpyx.sif
HF_REPO ?= WORLDSAR/support
HF_REPO_TYPE ?= dataset

SNAP_VERSION ?= 12.0.0
SNAP_INSTALLER ?= esa-snap_all_linux-$(SNAP_VERSION).sh
SNAP_INSTALL_URL ?= https://download.esa.int/step/snap/$(SNAP_VERSION)/installers/$(SNAP_INSTALLER)
SNAP_VARFILE ?= $(CURDIR)/snap.varfile
SNAP_VMOPTIONS ?= $(CURDIR)/snap/bin/gpt.vmoptions
SNAP_INSTALL_PARENT ?= $(CURDIR)/..

# ---------- Meta ----------
.PHONY: help \
	check-docker check-compose check-uv check-wget check-singularity check-hf \
	clean-venv venv install-deps install-phidown setup \
	install-snap \
	docker-build docker-test docker-push docker-all prune-docker \
	recreate up-recreate up down logs ps pull push \
	sif-build sif-push sif-all sifbuid sifpush sifup \
	clean_venv install_deps install_phidown prune_docker up_recreate

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------- Guard checks ----------
check-docker: ## Verify docker CLI is available
	@command -v docker >/dev/null 2>&1 || { echo "Error: docker not found in PATH."; exit 1; }

check-compose: check-docker ## Verify docker compose plugin is available
	@$(DOCKER) compose version >/dev/null 2>&1 || { echo "Error: docker compose plugin not available."; exit 1; }

check-uv: ## Verify uv is available
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv not found in PATH."; exit 1; }

check-wget: ## Verify wget is available
	@command -v wget >/dev/null 2>&1 || { echo "Error: wget not found in PATH."; exit 1; }

check-singularity: ## Verify singularity/apptainer is available
	@command -v singularity >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1 || { \
		echo "Error: singularity or apptainer not found in PATH."; exit 1; }

check-hf: ## Verify Hugging Face CLI is available
	@command -v hf >/dev/null 2>&1 || { echo "Error: hf (Hugging Face CLI) not found in PATH."; exit 1; }

# ---------- Python / uv ----------
clean-venv: ## Remove local virtual environment (.venv)
	@echo "Removing virtual environment..."
	@rm -rf .venv

venv: check-uv ## Create virtual environment with uv
	@echo "Creating virtual environment with uv..."
	uv venv .venv

install-deps: check-uv ## Sync project dependencies with uv
	@echo "Installing dependencies with uv..."
	uv sync

install-phidown: check-uv ## Add phidown dependency with uv
	@echo "Installing phi-down..."
	uv add phidown

setup: clean-venv venv install-deps ## Recreate venv and install dependencies
	@echo "Setup complete."

# ---------- SNAP ----------
install-snap: check-wget ## Install SNAP and configure default memory
	@echo "Installing packages for S1 data processing..."
	$(SUDO) apt-get update
	$(SUDO) apt-get install -y --fix-missing libfftw3-dev libtiff5-dev gfortran libgfortran5 jblas git wget
	@echo "Downloading SNAP installer $(SNAP_VERSION)..."
	wget -O "$(SNAP_INSTALLER)" "$(SNAP_INSTALL_URL)"
	chmod +x "$(SNAP_INSTALLER)"
	@echo "Writing SNAP varfile..."
	@printf '%s\n' \
		'deleteAllSnapEngineDir$$Boolean=false' \
		'deleteOnlySnapDesktopDir$$Boolean=false' \
		'executeLauncherWithPythonAction$$Boolean=false' \
		'forcePython$$Boolean=false' \
		'pythonExecutable=/usr/bin/python' \
		'sys.adminRights$$Boolean=true' \
		'sys.component.RSTB$$Boolean=true' \
		'sys.component.S1TBX$$Boolean=true' \
		'sys.component.S2TBX$$Boolean=false' \
		'sys.component.S3TBX$$Boolean=false' \
		'sys.component.SNAP$$Boolean=true' \
		'sys.installationDir=$(CURDIR)/snap' \
		'sys.languageId=en' \
		'sys.programGroupDisabled$$Boolean=false' \
		'sys.symlinkDir=/usr/local/bin' \
		> "$(SNAP_VARFILE)"
	@echo "Installing SNAP..."
	./"$(SNAP_INSTALLER)" -q -varfile "$(SNAP_VARFILE)" -dir "$(SNAP_INSTALL_PARENT)"
	@echo "Configuring SNAP memory settings..."
	echo "-Xmx8G" > "$(SNAP_VMOPTIONS)"
	@echo "Cleaning installer artifacts..."
	rm -f "$(SNAP_INSTALLER)" "$(SNAP_VARFILE)"
	@echo "SNAP installation complete."

# ---------- SIF / Singularity ----------
sif-build: check-singularity ## Build SIF from Docker image (uses DOCKER_FULL)
	@echo "Building $(SIF) from docker://$(DOCKER_FULL)..."
	@if command -v apptainer >/dev/null 2>&1; then \
		apptainer build --disable-cache "$(SIF)" "docker://$(DOCKER_FULL)"; \
	else \
		singularity build --disable-cache "$(SIF)" "docker://$(DOCKER_FULL)"; \
	fi

sif-push: check-hf ## Upload SIF to Hugging Face (uses HF_REPO)
	@test -f "$(SIF)" || { echo "Error: $(SIF) not found. Run 'make sif-build' first."; exit 1; }
	@echo "Uploading $(SIF) to $(HF_REPO) ($(HF_REPO_TYPE))..."
	hf upload "$(HF_REPO)" "$(SIF)" --repo-type "$(HF_REPO_TYPE)"

sif-run:
	apptainer run --writable-tmpfs sarpyx.sif /bin/bash


sif-all: recreate push sif-build sif-push ## Build + upload SIF

sifbuid: sif-build
sifpush: sif-push
sifup: sif-all

# ---------- Docker ----------
docker-build: check-docker ## Build Docker image
	@echo "Building Docker image $(DOCKER_FULL)..."
	$(DOCKER) build --platform "$(PLATFORM)" -t "$(DOCKER_FULL)" .

docker-test: docker-build ## Run containerized Docker tests
	@echo "Running Docker build tests..."
	$(DOCKER) run --rm "$(DOCKER_FULL)" sh -c "\
		python3 -m pip install pytest && \
		python3 -m pytest /workspace/tests/test_docker.py -v --tb=short"

docker-push: docker-build ## Push Docker image
	@echo "Pushing $(DOCKER_FULL)..."
	$(DOCKER) push "$(DOCKER_FULL)"

docker-all: docker-build docker-test docker-push ## Build, test, and push image

prune-docker: check-docker ## Prune local Docker data
	@echo "Pruning Docker system..."
	$(DOCKER) system prune -a

# ---------- Compose ----------
recreate: check-compose ## Compose up with build, force recreate, and remove orphans
	$(COMPOSE) up --build --force-recreate --remove-orphans

up-recreate: check-compose ## Compose up with build and force recreate
	$(COMPOSE) up --build --force-recreate

up: check-compose ## Compose up
	$(COMPOSE) up

down: check-compose ## Compose down
	$(COMPOSE) down

logs: check-compose ## Follow compose logs
	$(COMPOSE) logs -f

ps: check-compose ## List compose services
	$(COMPOSE) ps

pull: check-compose ## Pull compose service images
	$(COMPOSE) pull

push: check-docker ## Push image configured by DOCKER_IMAGE/DOCKER_TAG
	$(DOCKER) push "$(DOCKER_FULL)"


# ---------- Backward-compatible aliases ----------
clean_venv: clean-venv
install_deps: install-deps
install_phidown: install-phidown
prune_docker: prune-docker
up_recreate: up-recreate
