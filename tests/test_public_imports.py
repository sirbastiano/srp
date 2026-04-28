import importlib
from pathlib import Path


def test_package_exports_snapflow_and_snap_alias() -> None:
    import sarpyx

    assert 'snapflow' in sarpyx.__all__
    assert sarpyx.snap is sarpyx.snapflow

    snap_module = importlib.import_module('sarpyx.snap')
    assert snap_module is sarpyx.snapflow
    assert sarpyx.snap is sarpyx.snapflow


def test_reported_public_modules_import() -> None:
    for module_name in (
        'sarpyx.processor.algorithms.mbautofocus',
        'sarpyx.processor.core.subaperture',
        'sarpyx.snapflow.dim_updater',
        'sarpyx.utils.complex_losses',
    ):
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_dim_updater_lives_in_snapflow() -> None:
    from sarpyx import snapflow

    package_root = Path(__file__).resolve().parents[1] / 'sarpyx'

    assert not (package_root / 'processor' / 'core' / 'dim_updater.py').exists()
    assert (package_root / 'snapflow' / 'dim_updater.py').exists()
    assert snapflow.update_dim_add_bands_from_data_dir.__name__ == 'update_dim_add_bands_from_data_dir'


def test_data_and_processor_utils_exports_are_explicit() -> None:
    from sarpyx.processor import data, utils

    assert 'read_tif' in data.__all__
    assert data.read_tif.__name__ == 'read_tif'
    assert 'write_geotiff' in data.__all__
    assert data.write_geotiff.__name__ == 'write_geotiff'
    assert 'format_converter' in data.__all__
    assert data.format_converter.__name__ == 'format_converter'

    assert 'summarize_2d_array' in utils.__all__
    assert utils.summarize_2d_array.__name__ == 'summarize_2d_array'
    assert 'cleanup_memory' in utils.__all__
    assert utils.cleanup_memory.__name__ == 'cleanup_memory'
    assert 'ssim' in utils.__all__
    assert utils.ssim.__name__ == 'ssim'


def test_package_tree_has_no_backup_named_python_modules() -> None:
    package_root = Path(__file__).resolve().parents[1] / 'sarpyx'
    bad_modules = [
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob('*.py')
        if any(part in path.stem.lower().split('_') for part in ('old', 'backup', 'tmp'))
    ]

    assert bad_modules == []
