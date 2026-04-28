"""Docker build verification tests.

These tests are executed inside the Docker container to validate that the image
was built correctly and that all critical components are functional.

Run locally:
    make docker-test

Run in CI:
    Pytest is installed at runtime and this test file is mounted into the
    container from the host workspace.
"""

import importlib
import os
import shutil
import subprocess
import sys

import pytest

IN_DOCKER_IMAGE = os.path.exists("/.dockerenv") or os.path.isdir("/workspace")

# ── 1. Python runtime ───────────────────────────────────────────────

class TestPythonRuntime:
    """Validate the Python interpreter and standard library."""

    def test_python_version(self):
        assert sys.version_info[:2] >= (3, 11), (
            f"Expected Python >=3.11, got {sys.version}"
        )

    def test_pip_available(self):
        assert shutil.which("pip") or shutil.which("pip3"), "pip not found on PATH"


# ── 2. Package import ───────────────────────────────────────────────

class TestSarpyxImport:
    """Validate that the sarpyx package is importable."""

    def test_import_sarpyx(self):
        import sarpyx
        assert hasattr(sarpyx, "__version__"), "sarpyx missing __version__"

    def test_version_not_empty(self):
        import sarpyx
        assert sarpyx.__version__, "sarpyx.__version__ is empty"

    @pytest.mark.parametrize("submodule", [
        "sarpyx.utils",
        "sarpyx.processor",
        "sarpyx.science",
        "sarpyx.sla",
        "sarpyx.cli",
    ])
    def test_import_submodules(self, submodule: str):
        mod = importlib.import_module(submodule)
        assert mod is not None


# ── 3. CLI entry-points ─────────────────────────────────────────────

class TestCLIEntryPoints:
    """Verify CLI scripts are installed and respond to --help."""

    @pytest.mark.parametrize("cmd", [
        "sarpyx",
        "sarpyx-decode",
        "sarpyx-focus",
        "sarpyx-shipdet",
        "sarpyx-unzip",
        "sarpyx-upload",
    ])
    def test_cli_help(self, cmd: str):
        result = subprocess.run(
            [cmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # --help should exit 0
        assert result.returncode == 0, (
            f"{cmd} --help failed (rc={result.returncode}): {result.stderr}"
        )


# ── 4. Key dependencies ─────────────────────────────────────────────

class TestDependencies:
    """Spot-check that heavy dependencies resolve correctly."""

    @pytest.mark.parametrize("pkg", [
        "numpy",
        "scipy",
        "matplotlib",
        "rasterio",
        "geopandas",
        "zarr",
        "h5py",
        "numba",
        "shapely",
        "dask",
    ])
    def test_dependency_importable(self, pkg: str):
        importlib.import_module(pkg)


# ── 5. SNAP / Java environment ──────────────────────────────────────

@pytest.mark.skipif(not IN_DOCKER_IMAGE, reason="Docker image checks only")
class TestSnapEnvironment:
    """Validate SNAP toolbox and Java are present."""

    def test_java_home_set(self):
        java_home = os.environ.get("JAVA_HOME", "")
        assert java_home, "JAVA_HOME is not set"
        assert os.path.isdir(java_home), f"JAVA_HOME dir missing: {java_home}"

    def test_snap_home_set(self):
        snap_home = os.environ.get("SNAP_HOME", "")
        assert snap_home, "SNAP_HOME is not set"
        assert os.path.isdir(snap_home), f"SNAP_HOME dir missing: {snap_home}"

    def test_gpt_on_path(self):
        assert shutil.which("gpt"), "SNAP gpt not found on PATH"


# ── 6. Grid configuration ───────────────────────────────────────────

@pytest.mark.skipif(not IN_DOCKER_IMAGE, reason="Docker image checks only")
class TestGridConfiguration:
    """Check that the runtime grid contract is satisfied."""

    def test_grid_path_set(self):
        assert os.environ.get("GRID_PATH"), "GRID_PATH is not set"

    def test_grid_path_exists(self):
        grid_path = os.environ.get("GRID_PATH", "")
        assert grid_path.endswith(".geojson"), f"GRID_PATH is not a .geojson: {grid_path}"
        assert os.path.isfile(grid_path), f"GRID_PATH missing: {grid_path}"
