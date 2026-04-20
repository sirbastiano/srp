from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from pyscripts import worldsar as pyscripts_worldsar
from sarpyx.utils.worldsar_h5 import (
    CORE_METADATA_KEYS,
    convert_tile_h5_to_zarr,
    enrich_validation_results_with_h5_structure,
    validate_h5_tile,
)


def _write_tile(
    path: Path,
    *,
    include_extra_array: bool = True,
    include_quality_group: bool = True,
    include_quality_flag: bool = True,
) -> None:
    with h5py.File(path, 'w') as h5_file:
        h5_file.attrs['root_attr'] = np.bytes_(b'root')

        bands_group = h5_file.create_group('bands')
        band = bands_group.create_dataset('Band_A', data=np.arange(4, dtype=np.float32).reshape(2, 2))
        band.attrs['CLASS'] = 'RasterDataNode'
        band.attrs['IMAGE_VERSION'] = '1.0'
        band.attrs['log10_scaled'] = False
        band.attrs['raster_height'] = 2
        band.attrs['raster_width'] = 2
        band.attrs['scaling_factor'] = 1.0
        band.attrs['scaling_offset'] = 0.0
        band.attrs['unit'] = 'linear'

        if include_extra_array:
            geolocation = h5_file.create_group('geolocation')
            geolocation.create_dataset('latitude', data=np.linspace(0.0, 1.0, 4).reshape(2, 2))

        metadata = h5_file.create_group('metadata')
        abstracted = metadata.create_group('Abstracted_Metadata')
        for key in CORE_METADATA_KEYS:
            abstracted.attrs[key] = f'{key}-value'
        if include_quality_flag:
            abstracted.attrs['quality_flag'] = 'ok'

        if include_quality_group:
            quality = metadata.create_group('Quality')
            quality.attrs['state'] = 'good'


def test_convert_tile_h5_to_zarr_preserves_nested_structure_and_attributes(tmp_path: Path) -> None:
    input_tile = tmp_path / 'tile.h5'
    output_store = tmp_path / 'tile.zarr'
    _write_tile(input_tile)

    converted = convert_tile_h5_to_zarr(input_tile, output_store, overwrite=True)

    assert converted == output_store
    root = zarr.open(output_store.as_posix(), mode='r')
    assert isinstance(root, zarr.Group)
    assert root.attrs['root_attr'] == 'root'
    np.testing.assert_array_equal(root['bands']['Band_A'][:], np.arange(4, dtype=np.float32).reshape(2, 2))
    assert root['bands']['Band_A'].attrs['unit'] == 'linear'
    np.testing.assert_array_equal(root['geolocation']['latitude'][:], np.linspace(0.0, 1.0, 4).reshape(2, 2))
    assert root['metadata']['Abstracted_Metadata'].attrs['MISSION'] == 'MISSION-value'


def test_validate_h5_tile_reports_missing_arrays_and_metadata_structure(tmp_path: Path) -> None:
    complete_tile = tmp_path / 'complete.h5'
    missing_tile = tmp_path / 'missing.h5'
    _write_tile(complete_tile)
    _write_tile(
        missing_tile,
        include_extra_array=False,
        include_quality_group=False,
        include_quality_flag=False,
    )

    results = [
        validate_h5_tile(complete_tile, expected_bands=['Band_A']),
        validate_h5_tile(missing_tile, expected_bands=['Band_A']),
    ]
    summary = enrich_validation_results_with_h5_structure(results)

    assert summary['expected_array_paths'] == ['geolocation/latitude']
    assert 'metadata/Quality' in summary['expected_metadata_paths']
    assert 'metadata/Abstracted_Metadata@quality_flag' in summary['expected_metadata_attr_paths']

    complete_result, missing_result = results
    assert complete_result['structure_ok'] is True
    assert missing_result['structure_ok'] is False
    assert missing_result['missing_array_paths'] == ['geolocation/latitude']
    assert missing_result['missing_metadata_paths'] == ['metadata/Quality']
    assert missing_result['missing_metadata_attrs'] == ['metadata/Abstracted_Metadata@quality_flag']
    assert missing_result['status'] == 'failed'


def test_pyscripts_worldsar_h5_to_zarr_only_mode_creates_sibling_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    input_tile = tmp_path / 'tile_only.h5'
    _write_tile(input_tile)

    monkeypatch.setattr(sys, 'argv', ['worldsar.py', '--input', str(input_tile), '--h5-to-zarr-only'])

    with pytest.raises(SystemExit) as excinfo:
        pyscripts_worldsar.main()

    assert excinfo.value.code == 0

    output = capsys.readouterr().out
    summary = json.loads(output)
    expected_store = input_tile.with_suffix('.zarr')
    assert summary['output'] == str(expected_store)
    assert expected_store.is_dir()
