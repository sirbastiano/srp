# Reviewer smoke test for sarpyx

This file provides a compact, objective validation path for reviewers. It complements the
project's broader documentation and test suite.

## Path A: local Python install

Use this path to verify importability, CLI exposure, and basic local setup.

```bash
git clone https://github.com/ESA-PhiLab/sarpyx.git
cd sarpyx
uv venv .venv
source .venv/bin/activate
uv sync
python -m pip install -e .
sarpyx --help
sarpyx decode --help
sarpyx focus --help
pytest -q tests/test_docker.py
```

Optional import check:

```bash
python - <<'PY'
import sarpyx
from sarpyx.utils.zarr_utils import ZarrManager
print(sarpyx.__version__)
print(ZarrManager.__name__)
PY
```

Expected result: commands exit with status 0, the package imports cleanly, and the test
command passes.

## Path B: container validation

Use this path when you want the closest match to the documented reproducible environment.

```bash
docker compose version
make recreate
make docker-test
```

Expected result: the image builds, the container starts successfully, and the Docker smoke
tests pass.

## Notes

- SNAP + Java are required for SNAP-dependent commands such as `worldsar` and `shipdet`.
- The repository also documents a broader `pytest -q` path for maintainers and power users.
- For the JOSS review, the key objective is that a reviewer can install the software and
  verify the core functionality with an objective procedure.
