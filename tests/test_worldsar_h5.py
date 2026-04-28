from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from sarpyx.utils.worldsar_h5 import (
    CORE_METADATA_KEYS,
    build_validation_dashboard_rows,
    build_validation_group_summary_rows,
    build_validation_map_layers,
    convert_tile_h5_to_zarr,
    enrich_validation_results_with_h5_structure,
    extract_tile_geometry_from_abstract_metadata,
    normalize_expected_tile_geometries,
    resolve_expected_band_names_from_dim_product,
    validate_h5_tile,
    write_h5_validation_report_pdf,
)


def _load_legacy_worldsar_module():
    module_path = Path(__file__).resolve().parents[1] / 'pyscripts' / 'worldsar.py'
    spec = importlib.util.spec_from_file_location('tests_pyscripts_worldsar', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load legacy worldsar module from {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pyscripts_worldsar = _load_legacy_worldsar_module()


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


def test_resolve_expected_band_names_prefers_materialized_data_dir(tmp_path: Path) -> None:
    dim_path = tmp_path / 'product.dim'
    data_dir = tmp_path / 'product.data'
    data_dir.mkdir()

    dim_path.write_text(
        """<Dimap_Document>
  <Image_Interpretation>
    <Spectral_Band_Info><BAND_NAME>Alpha</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Anisotropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Entropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>i_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>q_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Intensity_IW3_VH</BAND_NAME></Spectral_Band_Info>
  </Image_Interpretation>
  <Data_Access>
    <Data_File>
      <BAND_INDEX>1</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/Alpha.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>2</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/Anisotropy.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>3</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/Entropy.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>4</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/i_IW3_VH.hdr" />
    </Data_File>
    <Data_File>
      <BAND_INDEX>5</BAND_INDEX>
      <DATA_FILE_PATH href="./product.data/q_IW3_VH.hdr" />
    </Data_File>
  </Data_Access>
</Dimap_Document>
""",
        encoding='utf-8',
    )

    for name in ('Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH'):
        (data_dir / f'{name}.hdr').write_text('ENVI\n', encoding='utf-8')

    expected_bands = resolve_expected_band_names_from_dim_product(dim_path)

    assert expected_bands == ['Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH']


def test_validate_h5_tile_ignores_virtual_intensity_bands_when_data_dir_is_materialized(tmp_path: Path) -> None:
    dim_path = tmp_path / 'product.dim'
    data_dir = tmp_path / 'product.data'
    data_dir.mkdir()

    dim_path.write_text(
        """<Dimap_Document>
  <Image_Interpretation>
    <Spectral_Band_Info><BAND_NAME>Alpha</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Anisotropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Entropy</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>i_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>q_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Intensity_IW3_VH</BAND_NAME></Spectral_Band_Info>
    <Spectral_Band_Info><BAND_NAME>Intensity_IW3_VV</BAND_NAME></Spectral_Band_Info>
  </Image_Interpretation>
</Dimap_Document>
""",
        encoding='utf-8',
    )

    for name in ('Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH'):
        (data_dir / f'{name}.hdr').write_text('ENVI\n', encoding='utf-8')

    tile_path = tmp_path / 'tile.h5'
    with h5py.File(tile_path, 'w') as h5_file:
        bands_group = h5_file.create_group('bands')
        for band_name in ('Alpha', 'Anisotropy', 'Entropy', 'i_IW3_VH', 'q_IW3_VH'):
            band = bands_group.create_dataset(band_name, data=np.arange(4, dtype=np.float32).reshape(2, 2))
            band.attrs['CLASS'] = 'RasterDataNode'
            band.attrs['IMAGE_VERSION'] = '1.0'
            band.attrs['log10_scaled'] = False
            band.attrs['raster_height'] = 2
            band.attrs['raster_width'] = 2
            band.attrs['scaling_factor'] = 1.0
            band.attrs['scaling_offset'] = 0.0
            band.attrs['unit'] = 'linear'

        metadata = h5_file.create_group('metadata')
        abstracted = metadata.create_group('Abstracted_Metadata')
        for key in CORE_METADATA_KEYS:
            abstracted.attrs[key] = f'{key}-value'

    expected_bands = resolve_expected_band_names_from_dim_product(dim_path)
    result = validate_h5_tile(tile_path, expected_bands=expected_bands)

    assert result['missing_bands'] == []
    assert result['extra_bands'] == []
    assert result['bands_ok'] is True


def test_extract_tile_geometry_from_abstract_metadata_builds_polygon_and_center() -> None:
    geometry = extract_tile_geometry_from_abstract_metadata(
        {
            'first_near_lat': 45.0,
            'first_near_long': 10.0,
            'first_far_lat': 45.0,
            'first_far_long': 10.1,
            'last_far_lat': 44.9,
            'last_far_long': 10.1,
            'last_near_lat': 44.9,
            'last_near_long': 10.0,
            'centre_lat': 44.95,
            'centre_lon': 10.05,
        }
    )

    assert geometry['tile_polygon_coords'] == [
        (10.0, 45.0),
        (10.1, 45.0),
        (10.1, 44.9),
        (10.0, 44.9),
        (10.0, 45.0),
    ]
    assert geometry['tile_center_coords'] == (10.05, 44.95)


def test_extract_tile_geometry_from_abstract_metadata_falls_back_to_center_only() -> None:
    geometry = extract_tile_geometry_from_abstract_metadata(
        {
            'centre_lat': 12.5,
            'centre_lon': 41.9,
        }
    )

    assert geometry['tile_polygon_coords'] is None
    assert geometry['tile_center_coords'] == (41.9, 12.5)


def test_normalize_expected_tile_geometries_preserves_tile_names() -> None:
    rectangles = [
        {
            'TL': {'geometry': {'coordinates': [10.0, 45.0]}},
            'TR': {'geometry': {'coordinates': [10.1, 45.0]}},
            'BR': {'geometry': {'coordinates': [10.1, 44.9]}},
            'BL': {'geometry': {'coordinates': [10.0, 44.9]}, 'properties': {'name': 'tile-a'}},
        },
        {
            'TL': {'geometry': {'coordinates': [10.1, 45.0]}},
            'TR': {'geometry': {'coordinates': [10.2, 45.0]}},
            'BR': {'geometry': {'coordinates': [10.2, 44.9]}},
            'BL': {'geometry': {'coordinates': [10.1, 44.9]}, 'properties': {'name': 'tile-b'}},
        },
    ]

    normalized = normalize_expected_tile_geometries(rectangles)

    assert sorted(normalized) == ['tile-a', 'tile-b']
    assert normalized['tile-a'][0] == (10.0, 45.0)
    assert normalized['tile-a'][-1] == (10.0, 45.0)


def _synthetic_validation_group() -> dict[str, object]:
    expected_tile_geometries = {
        'tile-a': [(10.0, 45.0), (10.1, 45.0), (10.1, 44.9), (10.0, 44.9), (10.0, 45.0)],
        'tile-b': [(10.1, 45.0), (10.2, 45.0), (10.2, 44.9), (10.1, 44.9), (10.1, 45.0)],
        'tile-c': [(10.2, 45.0), (10.3, 45.0), (10.3, 44.9), (10.2, 44.9), (10.2, 45.0)],
    }
    source_wkt = 'POLYGON ((9.95 44.85, 10.35 44.85, 10.35 45.05, 9.95 45.05, 9.95 44.85))'
    return {
        'name': 'IW1',
        'swath': 'IW1',
        'cuts_dir': 'cuts/IW1',
        'intermediate_product': 'intermediate/IW1.dim',
        'expected_bands': ['Band_A', 'Band_B'],
        'expected_array_paths': ['geolocation/latitude'],
        'expected_metadata_paths': ['metadata', 'metadata/Abstracted_Metadata'],
        'expected_metadata_attr_paths': ['metadata/Abstracted_Metadata@quality_flag'],
        'expected_tiles': ['tile-a', 'tile-b', 'tile-c'],
        'actual_tiles': ['tile-a', 'tile-b', 'tile-extra'],
        'missing_tiles': ['tile-c'],
        'extra_tiles': ['tile-extra'],
        'skipped_tiles': ['tile-skip'],
        'failed_tiles': ['tile-b'],
        'expected_tile_count': 3,
        'actual_tile_count': 3,
        'source_wkt': source_wkt,
        'report_source_wkt': source_wkt,
        'expected_tile_geometries': expected_tile_geometries,
        'results': [
            {
                'tile': 'tile-a',
                'swath': 'IW1',
                'status': 'success',
                'bands_ok': True,
                'metadata_ok': True,
                'band_attrs_ok': True,
                'structure_ok': True,
                'missing_bands': [],
                'extra_bands': [],
                'missing_metadata_section': False,
                'empty_metadata_fields': [],
                'missing_core_metadata_fields': [],
                'empty_core_metadata_fields': [],
                'band_attr_issues': {},
                'shape_summary': [],
                'missing_array_paths': [],
                'missing_metadata_paths': [],
                'missing_metadata_attrs': [],
                'tile_polygon_coords': expected_tile_geometries['tile-a'],
                'tile_center_coords': (10.05, 44.95),
            },
            {
                'tile': 'tile-b',
                'swath': 'IW1',
                'status': 'failed',
                'bands_ok': False,
                'metadata_ok': False,
                'band_attrs_ok': False,
                'structure_ok': False,
                'missing_bands': ['Band_B'],
                'extra_bands': [],
                'missing_metadata_section': False,
                'empty_metadata_fields': ['quality_flag'],
                'missing_core_metadata_fields': [],
                'empty_core_metadata_fields': ['MISSION'],
                'band_attr_issues': {'Band_A': {'missing_attrs': ['unit'], 'empty_attrs': [], 'invalid_shape': False, 'shape': (2, 2)}},
                'shape_summary': ['(2, 2)', '(3, 3)'],
                'missing_array_paths': ['geolocation/latitude'],
                'missing_metadata_paths': ['metadata/Abstracted_Metadata'],
                'missing_metadata_attrs': ['metadata/Abstracted_Metadata@quality_flag'],
                'tile_polygon_coords': None,
                'tile_center_coords': (10.15, 44.95),
            },
            {
                'tile': 'tile-extra',
                'swath': 'IW1',
                'status': 'success',
                'bands_ok': True,
                'metadata_ok': True,
                'band_attrs_ok': True,
                'structure_ok': True,
                'missing_bands': [],
                'extra_bands': [],
                'missing_metadata_section': False,
                'empty_metadata_fields': [],
                'missing_core_metadata_fields': [],
                'empty_core_metadata_fields': [],
                'band_attr_issues': {},
                'shape_summary': [],
                'missing_array_paths': [],
                'missing_metadata_paths': [],
                'missing_metadata_attrs': [],
                'tile_polygon_coords': [(10.32, 45.0), (10.34, 45.0), (10.34, 44.98), (10.32, 44.98), (10.32, 45.0)],
                'tile_center_coords': (10.33, 44.99),
            },
        ],
        'rows': [{'ID': 'tile-a'}, {'ID': 'tile-b'}, {'ID': 'tile-extra'}],
    }


def test_validation_summary_dashboard_and_map_helpers_cover_status_buckets() -> None:
    group = _synthetic_validation_group()

    summary_rows = build_validation_group_summary_rows([group])
    dashboard_rows = build_validation_dashboard_rows(group)
    map_layers = build_validation_map_layers([group])

    assert summary_rows == [{
        'group': 'IW1',
        'expected': 3,
        'actual': 3,
        'passed': 2,
        'failed': 1,
        'skipped': 1,
        'missing': 1,
        'extra': 1,
        'overall_status': 'FAIL',
    }]
    assert dashboard_rows[0]['check'] == 'band inventory'
    assert dashboard_rows[0]['passed'] == 2
    assert dashboard_rows[-1] == {'check': 'overall', 'passed': 2, 'failed': 1, 'pass_pct': pytest.approx(66.7, abs=0.1)}
    assert [item['tile'] for item in map_layers['passed_polygons']] == ['tile-a']
    assert [item['tile'] for item in map_layers['failed_points']] == ['tile-b']
    assert [item['tile'] for item in map_layers['extra_polygons']] == ['tile-extra']
    assert [item['tile'] for item in map_layers['missing_polygons']] == ['tile-c']
    assert map_layers['counts'] == {
        'expected': 3,
        'passed': 1,
        'failed': 1,
        'skipped': 1,
        'missing': 1,
        'extra': 1,
    }
    assert map_layers['tiles_with_center_only_count'] == 1
    assert map_layers['tiles_without_geometry_count'] == 0


def test_write_h5_validation_report_pdf_smoke(tmp_path: Path) -> None:
    report_path = tmp_path / 'validation_report.pdf'
    group = _synthetic_validation_group()

    written = write_h5_validation_report_pdf(report_path, 'synthetic-product', [group])

    assert written == report_path
    assert report_path.exists()
    assert report_path.stat().st_size > 0
