# sarpyx Makefile

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------- Config ----------
SUDO ?= sudo
DOCKER ?= docker
COMPOSE_BAKE ?= false
COMPOSE_FILE ?= docker-compose.yml
COMPOSE ?= COMPOSE_BAKE=$(COMPOSE_BAKE) $(DOCKER) compose -f $(COMPOSE_FILE)

DOCKER_IMAGE ?= sirbastiano94/sarpyx
DOCKER_TAG ?= latest
DOCKER_FULL := $(DOCKER_IMAGE):$(DOCKER_TAG)
PLATFORM ?= linux/amd64
SMOKE_TEST_GRID ?= $(CURDIR)/tests/fixtures/grid_smoke.geojson
SMOKE_TEST_GRID_CONTAINER ?= /workspace/grid/grid_smoke.geojson
SMOKE_TESTS_DIR ?= /opt/smoke-tests
VALIDATE_GRID ?= tests/fixtures/sentinel_smoke_grid.geojson
SENTINEL_VALIDATE_SAFE ?= data/S1C_IW_SLC__1SDV_20260130T152608_20260130T152634_006135_00C4FA_664F.SAFE
SENTINEL_VALIDATE_GRID ?= $(VALIDATE_GRID)
SENTINEL_VALIDATE_OUT ?= outputs/validate-sentinel/processed
SENTINEL_VALIDATE_CUTS ?= outputs/validate-sentinel/cuts
SENTINEL_VALIDATE_DB ?= outputs/validate-sentinel/db
SENTINEL_VALIDATE_SNAP_USERDIR ?= outputs/validate-sentinel/snap-userdir
TSX_VALIDATE_PRODUCT ?= data/TSX_OPER_SAR_HS_EEC_20071130T165208_N51-485_E011-982_0000_v0104.SIP.ZIP
TSX_VALIDATE_GRID ?= $(VALIDATE_GRID)
TSX_VALIDATE_OUT ?= outputs/validate-tsx/subset
TSX_VALIDATE_PREPROCESS_OUT ?= outputs/validate-tsx/processed
TSX_VALIDATE_CUTS ?= outputs/validate-tsx/cuts
TSX_VALIDATE_DB ?= outputs/validate-tsx/db
TSX_VALIDATE_SNAP_USERDIR ?= outputs/validate-tsx/snap-userdir
TSX_VALIDATE_SUBSET_NAME ?= TSX_VALIDATE_SUBSET
TSX_VALIDATE_SUBSET_REGION ?= 0,0,2048,2048
TSX_VALIDATE_SUBSET_WKT_FILE ?= $(TSX_VALIDATE_OUT)/$(TSX_VALIDATE_SUBSET_NAME).wkt
NISAR_VALIDATE_PRODUCT ?= data/NISAR_GSLC_SAMPLE.h5
NISAR_VALIDATE_GRID ?= $(VALIDATE_GRID)
NISAR_VALIDATE_OUT ?= outputs/validate-nisar/processed
NISAR_VALIDATE_CUTS ?= outputs/validate-nisar/cuts
NISAR_VALIDATE_DB ?= outputs/validate-nisar/db
SENTINEL_GPT_PATH ?= /Applications/esa-snap/bin/gpt
SENTINEL_VALIDATE_IW ?= IW1
SENTINEL_VALIDATE_BURST ?= 1
SENTINEL_VALIDATE_TC_SOURCE_BAND ?= Alpha
# Optional manual override for the Sentinel validation footprint.
# Leave blank to derive expected coverage from the processed *_TC.dim raster.
SENTINEL_VALIDATE_WKT ?=
SENTINEL_GPT_MEMORY ?= 8G
SENTINEL_GPT_PARALLELISM ?= 1
SENTINEL_GPT_TIMEOUT ?= 14400

SIF ?= sarpyx.sif
SIF_TMPDIR ?= $(CURDIR)/.singularity/tmp
SIF_CACHEDIR ?= $(CURDIR)/.singularity/cache
HF_REPO ?= WORLDSAR/support
HF_REPO_TYPE ?= dataset

SNAP_VERSION ?= 12.0.0
SNAP_INSTALLER ?= esa-snap_all_linux-$(SNAP_VERSION).sh
SNAP_INSTALL_URL ?= https://download.esa.int/step/snap/$(SNAP_VERSION)/installers/$(SNAP_INSTALLER)
SNAP_VARFILE ?= $(CURDIR)/snap.varfile
SNAP_VMOPTIONS ?= $(CURDIR)/snap/bin/gpt.vmoptions
SNAP_INSTALL_PARENT ?= $(CURDIR)

# ---------- Meta ----------
.PHONY: help \
	check-docker check-compose check-uv check-wget check-singularity check-hf \
	check-grid check-sentinel-product check-sentinel-grid check-sentinel-gpt check-tsx-product check-tsx-grid check-nisar-product check-nisar-grid \
	clean-venv venv install-deps install-phidown setup \
	install-snap \
	validate-sentinel validate-tsx validate-nisar validate validate-all \
	docker-build docker-test docker-push docker-all prune-docker \
	recreate up-recreate up down logs ps pull push \
	push-to-hpc \
	sif-build sif-push sif-all sifbuid sifpush sifup \
	clean_venv install_deps install_phidown prune_docker up_recreate

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------- Guard checks ----------
generate-grid: ## Generate grid_10km.geojson file in grid folder
	@echo "Generating grid file..."
	@mkdir -p grid
	@if [ -f .venv/bin/python ]; then \
		.venv/bin/python -c "from sarpyx.utils.grid import Grid; grid = Grid(10); grid.points.to_file('grid/grid_10km.geojson', driver='GeoJSON'); print('Grid file created at grid/grid_10km.geojson')"; \
	else \
		python3 -c "from sarpyx.utils.grid import Grid; grid = Grid(10); grid.points.to_file('grid/grid_10km.geojson', driver='GeoJSON'); print('Grid file created at grid/grid_10km.geojson')"; \
	fi

check-docker: ## Verify docker CLI is available
	@command -v docker >/dev/null 2>&1 || { echo "Error: docker not found in PATH."; exit 1; }

check-compose: check-docker ## Verify docker compose plugin is available
	@$(DOCKER) compose version >/dev/null 2>&1 || { echo "Error: docker compose plugin not available."; exit 1; }

check-grid: ## Verify an external GRID_PATH or a host grid file exists
	@if [ -n "${GRID_PATH}" ]; then \
		if [ -f "${GRID_PATH}" ] && [[ "${GRID_PATH}" == *.geojson ]]; then \
			echo "Using GRID_PATH=${GRID_PATH}"; \
		else \
			echo "Error: GRID_PATH must point to an existing .geojson file: ${GRID_PATH}"; \
			exit 1; \
		fi; \
	elif compgen -G "./grid/*.geojson" > /dev/null; then \
		echo "Using host grid directory ./grid"; \
	else \
		echo "Error: no host grid file found in ./grid and GRID_PATH is unset."; \
		echo "Hint: run 'make generate-grid' to create ./grid/grid_10km.geojson manually."; \
		exit 1; \
	fi

check-sentinel-product: ## Verify the Sentinel-1 smoke SAFE product exists
	@test -d "$(SENTINEL_VALIDATE_SAFE)" || { echo "Error: Sentinel SAFE product not found: $(SENTINEL_VALIDATE_SAFE)"; exit 1; }
	@test -f "$(SENTINEL_VALIDATE_SAFE)/manifest.safe" || { echo "Error: Sentinel manifest not found: $(SENTINEL_VALIDATE_SAFE)/manifest.safe"; exit 1; }

check-sentinel-grid: ## Verify the Sentinel-1 smoke grid exists
	@test -f "$(SENTINEL_VALIDATE_GRID)" || { echo "Error: Sentinel smoke grid not found: $(SENTINEL_VALIDATE_GRID)"; exit 1; }

check-sentinel-gpt: ## Verify SNAP GPT for Sentinel validation is available
	@test -x "$(SENTINEL_GPT_PATH)" || { \
		echo "Error: SNAP GPT executable not found: $(SENTINEL_GPT_PATH)"; \
		echo "Set SENTINEL_GPT_PATH to the SNAP gpt executable or run 'make install-snap'."; \
		exit 1; \
	}

check-tsx-product: ## Verify the TerraSAR-X validation product exists
	@test -e "$(TSX_VALIDATE_PRODUCT)" || { echo "Error: TerraSAR-X product not found: $(TSX_VALIDATE_PRODUCT)"; exit 1; }

check-tsx-grid: ## Verify the TerraSAR-X validation grid exists
	@test -f "$(TSX_VALIDATE_GRID)" || { echo "Error: TerraSAR-X validation grid not found: $(TSX_VALIDATE_GRID)"; exit 1; }

check-nisar-product: ## Verify the NISAR validation product exists
	@test -f "$(NISAR_VALIDATE_PRODUCT)" || { echo "Error: NISAR product not found: $(NISAR_VALIDATE_PRODUCT)"; exit 1; }

check-nisar-grid: ## Verify the NISAR validation grid exists
	@test -f "$(NISAR_VALIDATE_GRID)" || { echo "Error: NISAR validation grid not found: $(NISAR_VALIDATE_GRID)"; exit 1; }

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
	@mkdir -p "$(SIF_TMPDIR)" "$(SIF_CACHEDIR)"
	@if command -v apptainer >/dev/null 2>&1; then \
		APPTAINER_TMPDIR="$(SIF_TMPDIR)" APPTAINER_CACHEDIR="$(SIF_CACHEDIR)" \
		apptainer build --force --disable-cache "$(SIF)" "docker://$(DOCKER_FULL)"; \
	else \
		SINGULARITY_TMPDIR="$(SIF_TMPDIR)" SINGULARITY_CACHEDIR="$(SIF_CACHEDIR)" \
		singularity build --force --disable-cache "$(SIF)" "docker://$(DOCKER_FULL)"; \
	fi

sif-push: check-hf ## Upload SIF to Hugging Face (uses HF_REPO)
	@test -f "$(SIF)" || { echo "Error: $(SIF) not found. Run 'make sif-build' first."; exit 1; }
	@echo "Uploading $(SIF) to $(HF_REPO) ($(HF_REPO_TYPE))..."
	hf upload "$(HF_REPO)" "$(SIF)" --repo-type "$(HF_REPO_TYPE)"

sif-run:
	apptainer run --writable-tmpfs sarpyx.sif /bin/bash


# sif-all: sif-build sif-push ## Build + upload SIF
sif-all: sif-build push-to-hpc ## Build + upload SIF

sifbuid: sif-build
sifpush: sif-push
sifup: sif-all

# ---------- Docker ----------
docker-build: check-docker ## Build Docker image
	@echo "Building Docker image $(DOCKER_FULL)..."
	$(DOCKER) build --platform "$(PLATFORM)" -t "$(DOCKER_FULL)" .

docker-test: docker-build ## Run containerized Docker tests
	@echo "Running Docker build tests..."
	@test -f "$(SMOKE_TEST_GRID)" || { echo "Error: smoke test grid missing at $(SMOKE_TEST_GRID)"; exit 1; }
	$(DOCKER) run --rm \
		-e GRID_PATH="$(SMOKE_TEST_GRID_CONTAINER)" \
		-v "$(CURDIR)/tests:$(SMOKE_TESTS_DIR):ro" \
		-v "$(CURDIR)/tests/fixtures:/workspace/grid:ro" \
		"$(DOCKER_FULL)" sh -lc "\
			python3.11 -m pip install --no-cache-dir pytest==8.4.0 && \
			python3.11 -m pytest $(SMOKE_TESTS_DIR)/test_docker.py -v --tb=short"

docker-push: docker-build ## Push Docker image
	@echo "Pushing $(DOCKER_FULL)..."
	$(DOCKER) push "$(DOCKER_FULL)"

docker-all: docker-build docker-test docker-push ## Build, test, and push image

prune-docker: check-docker ## Prune local Docker data
	@echo "Pruning Docker system..."
	$(DOCKER) system prune -a

validate-sentinel: check-uv check-sentinel-product check-sentinel-grid check-sentinel-gpt ## Run a smoke pipeline test on the Sentinel-1 SAFE product
	@mkdir -p "$(SENTINEL_VALIDATE_OUT)" "$(SENTINEL_VALIDATE_CUTS)" "$(SENTINEL_VALIDATE_DB)" "$(SENTINEL_VALIDATE_SNAP_USERDIR)"
	@sentinel_skip_preprocessing=""; \
	if find "$(SENTINEL_VALIDATE_OUT)/$(SENTINEL_VALIDATE_IW)" -maxdepth 1 -type f -name '*_TC.dim' | grep -q .; then \
		echo "Reusing existing Sentinel TerrainCorrection product from $(SENTINEL_VALIDATE_OUT)/$(SENTINEL_VALIDATE_IW)"; \
		sentinel_skip_preprocessing="--skip-preprocessing"; \
	fi; \
		PRODUCT_WKT="$(SENTINEL_VALIDATE_WKT)" uv run sarpyx \
		--input "$(SENTINEL_VALIDATE_SAFE)" \
		--output "$(SENTINEL_VALIDATE_OUT)" \
		--cuts-outdir "$(SENTINEL_VALIDATE_CUTS)" \
		--grid-path "$(SENTINEL_VALIDATE_GRID)" \
		--db-dir "$(SENTINEL_VALIDATE_DB)" \
		--snap-userdir "$(SENTINEL_VALIDATE_SNAP_USERDIR)" \
		--gpt-path "$(SENTINEL_GPT_PATH)" \
		--gpt-memory "$(SENTINEL_GPT_MEMORY)" \
		--gpt-parallelism "$(SENTINEL_GPT_PARALLELISM)" \
		--gpt-timeout "$(SENTINEL_GPT_TIMEOUT)" \
			--sentinel-swath "$(SENTINEL_VALIDATE_IW)" \
			--sentinel-first-burst "$(SENTINEL_VALIDATE_BURST)" \
			--sentinel-last-burst "$(SENTINEL_VALIDATE_BURST)" \
			--sentinel-tc-source-band "$(SENTINEL_VALIDATE_TC_SOURCE_BAND)" \
			--orbit-continue-on-fail \
			$$sentinel_skip_preprocessing
	@cut_report=$$(find "$(SENTINEL_VALIDATE_CUTS)" -type f -name '*_cuts_report_*.txt' | sort | head -n 1); \
	pdf_report=$$(find "$(SENTINEL_VALIDATE_CUTS)" -maxdepth 1 -type f -name '*_h5_validation_report.pdf' | sort | head -n 1); \
	tile_count=$$(find "$(SENTINEL_VALIDATE_CUTS)" -type f -name '*.h5' | wc -l | tr -d ' '); \
	test -n "$$cut_report" || { echo "Error: Sentinel tile cut report not found under $(SENTINEL_VALIDATE_CUTS)"; exit 1; }; \
	test -n "$$pdf_report" || { echo "Error: Sentinel H5 validation report not found under $(SENTINEL_VALIDATE_CUTS)"; exit 1; }; \
	test "$$tile_count" -gt 0 || { echo "Error: Sentinel validation produced zero .h5 tiles under $(SENTINEL_VALIDATE_CUTS)"; exit 1; }; \
	echo "Sentinel tiles generated: $$tile_count"; \
	echo "Sentinel tile cut report: $$cut_report"; \
	sed -n '1,14p' "$$cut_report"; \
	echo "Sentinel H5 validation report: $$pdf_report"

# TSX_VALIDATE_PRODUCT may point to either the TerraSAR-X XML metadata file or the product directory/archive.
# TSX validation first writes a deterministic Subset region ($(TSX_VALIDATE_SUBSET_REGION)) to $(TSX_VALIDATE_OUT)
# and then reuses that subset DIM product via --skip-preprocessing for the normal tile validation workflow.
# If a TerrainCorrection DIM already exists in $(TSX_VALIDATE_PREPROCESS_OUT), the target reuses it and skips TSX preprocessing.
validate-tsx: check-uv check-tsx-product check-tsx-grid check-sentinel-gpt ## Run TerraSAR-X validation using a deterministic subset before tiling
	@mkdir -p "$(TSX_VALIDATE_PREPROCESS_OUT)" "$(TSX_VALIDATE_OUT)" "$(TSX_VALIDATE_CUTS)" "$(TSX_VALIDATE_DB)" "$(TSX_VALIDATE_SNAP_USERDIR)"
	uv run python -c 'from pathlib import Path; import xml.etree.ElementTree as ET; from pyproj import Transformer; from sarpyx.cli import worldsar; product_path = Path(r"$(TSX_VALIDATE_PRODUCT)"); preprocess_out = Path(r"$(TSX_VALIDATE_PREPROCESS_OUT)"); subset_out = Path(r"$(TSX_VALIDATE_OUT)"); subset_name = "$(TSX_VALIDATE_SUBSET_NAME)"; subset_region = "$(TSX_VALIDATE_SUBSET_REGION)"; wkt_file = Path(r"$(TSX_VALIDATE_SUBSET_WKT_FILE)"); gpt_memory = "$(SENTINEL_GPT_MEMORY)"; gpt_parallelism = $(SENTINEL_GPT_PARALLELISM); gpt_timeout = $(SENTINEL_GPT_TIMEOUT); preprocess_out.mkdir(parents=True, exist_ok=True); subset_out.mkdir(parents=True, exist_ok=True); existing_dims = sorted(preprocess_out.glob("*.dim"), key=lambda path: path.stat().st_mtime, reverse=True); intermediate = existing_dims[0] if existing_dims else Path(worldsar.pipeline_tsx_csg(product_path, preprocess_out, gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)); print(f"Reusing existing TSX TerrainCorrection product: {intermediate}") if existing_dims else None; subset_dim = Path(worldsar._run_gpt_op(intermediate, subset_out, "BEAM-DIMAP", "Subset", region=subset_region, copy_metadata=True, output_name=subset_name, gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)); tree = ET.parse(subset_dim); root = tree.getroot(); gt = worldsar._read_geotransform(subset_dim); ncols = int(root.findtext(".//Raster_Dimensions/NCOLS")); nrows = int(root.findtext(".//Raster_Dimensions/NROWS")); crs_name = root.findtext(".//Coordinate_Reference_System/NAME", default="EPSG:4326"); epsg = 4326; epsg = int(crs_name.upper().split("EPSG:")[1].split()[0]) if "EPSG:" in crs_name.upper() else epsg; origin_x, px_w, rot_x, origin_y, rot_y, px_h = gt; corners = [(origin_x, origin_y), (origin_x + ncols * px_w, origin_y + ncols * rot_y), (origin_x + ncols * px_w + nrows * rot_x, origin_y + ncols * rot_y + nrows * px_h), (origin_x + nrows * rot_x, origin_y + nrows * px_h)]; corners = [Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y) for x, y in corners] if epsg != 4326 else corners; corners.append(corners[0]); wkt_file.write_text("POLYGON((" + ", ".join(f"{lon} {lat}" for lon, lat in corners) + "))\n", encoding="utf-8"); print(subset_dim); print(wkt_file)'
	uv run sarpyx \
		--input "$(TSX_VALIDATE_PRODUCT)" \
		--output "$(TSX_VALIDATE_OUT)" \
		--cuts-outdir "$(TSX_VALIDATE_CUTS)" \
		--grid-path "$(TSX_VALIDATE_GRID)" \
		--db-dir "$(TSX_VALIDATE_DB)" \
		--snap-userdir "$(TSX_VALIDATE_SNAP_USERDIR)" \
		--gpt-path "$(SENTINEL_GPT_PATH)" \
		--gpt-memory "$(SENTINEL_GPT_MEMORY)" \
		--gpt-parallelism "$(SENTINEL_GPT_PARALLELISM)" \
		--gpt-timeout "$(SENTINEL_GPT_TIMEOUT)" \
		--product-wkt "$$(tr -d '\n' < "$(TSX_VALIDATE_SUBSET_WKT_FILE)")" \
		--skip-preprocessing

validate-nisar: check-uv check-nisar-product check-nisar-grid ## Run NISAR validation using the shared worldsar tiling and H5 checks
	@mkdir -p "$(NISAR_VALIDATE_OUT)" "$(NISAR_VALIDATE_CUTS)" "$(NISAR_VALIDATE_DB)"
	uv run sarpyx \
		--input "$(NISAR_VALIDATE_PRODUCT)" \
		--output "$(NISAR_VALIDATE_OUT)" \
		--cuts-outdir "$(NISAR_VALIDATE_CUTS)" \
		--grid-path "$(NISAR_VALIDATE_GRID)" \
		--db-dir "$(NISAR_VALIDATE_DB)"

validate-all: validate-sentinel validate-tsx validate-nisar ## Run all mission validation targets

validate: validate-all ## Alias for aggregate validation

# ---------- Compose ----------
compose-precheck: check-compose check-grid ## Validate docker compose and grid prerequisites

recreate: compose-precheck ## Compose up with build, force recreate, and remove orphans
	$(COMPOSE) up --build --force-recreate --remove-orphans

up-recreate: compose-precheck ## Compose up with build and force recreate
	$(COMPOSE) up --build --force-recreate

up: compose-precheck ## Compose up
	$(COMPOSE) up

down: check-compose ## Compose down
	$(COMPOSE) down

logs: check-compose ## Follow compose logs
	$(COMPOSE) logs -f

ps: check-compose ## List compose services
	$(COMPOSE) ps

pull: check-compose ## Pull compose service images
	$(COMPOSE) pull

push: check-docker recreate ## Push image configured by DOCKER_IMAGE/DOCKER_TAG
	$(DOCKER) push "$(DOCKER_FULL)"

push-to-hpc: ## Run HPC upload script
	bash /shared/home/rdelprete/PythonProjects/srp/scripts/upload_sif.sh
# ---------- Backward-compatible aliases ----------
clean_venv: clean-venv
install_deps: install-deps
install_phidown: install-phidown
prune_docker: prune-docker
up_recreate: up-recreate
