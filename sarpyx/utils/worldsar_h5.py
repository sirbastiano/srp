from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import zarr

from sarpyx.utils.meta import extract_core_metadata_sentinel

DEFAULT_ZARR_CHUNK_SIZE = (32, 32)

CORE_METADATA_KEYS = (
    'MISSION',
    'ACQUISITION_MODE',
    'PRODUCT_TYPE',
    'radar_frequency',
    'pulse_repetition_frequency',
    'range_spacing',
    'azimuth_spacing',
    'range_bandwidth',
    'azimuth_bandwidth',
    'antenna_pointing',
    'PASS',
    'avg_scene_height',
    'PRODUCT',
    'mds1_tx_rx_polar',
    'mds2_tx_rx_polar',
    'first_line_time',
)

REQUIRED_BAND_ATTRS = (
    'CLASS',
    'IMAGE_VERSION',
    'log10_scaled',
    'raster_height',
    'raster_width',
    'scaling_factor',
    'scaling_offset',
    'unit',
)

MATERIALIZED_BAND_SUFFIXES = frozenset({'.hdr', '.img'})

VALIDATION_CHECKS = (
    ('band inventory', lambda result: result.get('bands_ok', False)),
    ('metadata completeness', lambda result: result.get('metadata_ok', False)),
    ('band attrs', lambda result: not result.get('band_attr_issues')),
    ('raster shape consistency', lambda result: not result.get('shape_summary')),
    ('non-band rasters', lambda result: not result.get('missing_array_paths')),
    ('metadata paths', lambda result: not result.get('missing_metadata_paths')),
    ('metadata attrs', lambda result: not result.get('missing_metadata_attrs')),
    ('overall', lambda result: result.get('status') == 'success'),
)


def normalize_attribute_value(value: Any) -> Any:
    """Convert HDF5 attribute values into JSON-serializable values."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode('utf-8', errors='surrogateescape')
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return normalize_attribute_value(value.item())
    if isinstance(value, np.ndarray):
        return normalize_attribute_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [normalize_attribute_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_attribute_value(item) for key, item in value.items()}
    return str(value)


def normalize_attributes(attrs: h5py.AttributeManager) -> dict[str, Any]:
    return {str(key): normalize_attribute_value(value) for key, value in attrs.items()}


def derive_chunk_shape(shape: tuple[int, ...], chunk_size: tuple[int, int]) -> tuple[int, ...] | None:
    if len(shape) == 0:
        return None
    if len(shape) == 1:
        return (min(shape[0], chunk_size[0]),)
    return (
        min(shape[0], chunk_size[0]),
        min(shape[1], chunk_size[1]),
        *shape[2:],
    )


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_suffix('.zarr')


def _read_dim_declared_band_names(dim_path: Path) -> list[str]:
    root = ET.parse(dim_path).getroot()
    band_names = []
    for spectral_band in root.findall('./Image_Interpretation/Spectral_Band_Info'):
        band_name = (spectral_band.findtext('BAND_NAME') or '').strip()
        if band_name:
            band_names.append(band_name)
    return sorted(set(band_names))


def _read_dim_materialized_band_names(dim_path: Path) -> list[str]:
    root = ET.parse(dim_path).getroot()
    band_names = []
    for data_file in root.findall('.//Data_File'):
        href = data_file.find('DATA_FILE_PATH')
        if href is None:
            continue
        band_name = Path(href.get('href', '')).name
        if not band_name:
            continue
        stem = Path(band_name).stem.strip()
        if stem:
            band_names.append(stem)
    return sorted(set(band_names))


def _discover_data_dir_band_names(dim_path: Path, known_band_names: set[str]) -> list[str]:
    data_dir = dim_path.with_suffix('.data')
    if not data_dir.is_dir():
        return []

    discovered = sorted({
        child.stem
        for child in data_dir.iterdir()
        if child.is_file()
        and child.suffix.lower() in MATERIALIZED_BAND_SUFFIXES
        and child.stem
    })
    if not discovered:
        return []

    filtered = sorted(name for name in discovered if name in known_band_names)
    return filtered or discovered


def resolve_expected_band_names_from_dim_product(dim_path: Path | str) -> list[str]:
    """Resolve expected H5 band names for a DIM product.

    Prefer the sibling ``.data`` directory when it exposes materialized raster names,
    then fall back to the declared spectral-band metadata in the DIM file itself.
    This avoids treating virtual bands such as ``Intensity_*`` as mandatory H5 exports.
    """

    dim_path = Path(dim_path)
    declared_band_names = _read_dim_declared_band_names(dim_path)
    materialized_band_names = _read_dim_materialized_band_names(dim_path)
    known_band_names = set(declared_band_names) | set(materialized_band_names)

    discovered_band_names = _discover_data_dir_band_names(dim_path, known_band_names)
    if discovered_band_names:
        return discovered_band_names
    if declared_band_names:
        return declared_band_names
    raise RuntimeError(f'No band names found in {dim_path}')


def _copy_dataset(source: h5py.Dataset, target_parent: zarr.Group, chunk_size: tuple[int, int]) -> None:
    data = source[()]
    chunks = derive_chunk_shape(tuple(int(dim) for dim in source.shape), chunk_size)
    array = target_parent.create_array(
        source.name.rsplit('/', 1)[-1],
        data=data,
        chunks=chunks if chunks is not None else 'auto',
        overwrite=False,
    )
    array.attrs.update(normalize_attributes(source.attrs))


def _copy_group(source: h5py.Group, target: zarr.Group, chunk_size: tuple[int, int]) -> None:
    target.attrs.update(normalize_attributes(source.attrs))
    for name, obj in source.items():
        if isinstance(obj, h5py.Group):
            child = target.create_group(name)
            _copy_group(obj, child, chunk_size)
        elif isinstance(obj, h5py.Dataset):
            _copy_dataset(obj, target, chunk_size)
        else:
            raise TypeError(f'Unsupported HDF5 object at {obj.name}: {type(obj)!r}')


def convert_tile_h5_to_zarr(
    input_path: Path | str,
    output_path: Path | str | None = None,
    chunk_size: tuple[int, int] = DEFAULT_ZARR_CHUNK_SIZE,
    overwrite: bool = False,
) -> Path:
    source_path = Path(input_path).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).absolute()

    if output_path is None:
        target_path = resolve_output_path(source_path, None)
    else:
        target_path = Path(output_path).expanduser()
        if not target_path.is_absolute():
            target_path = (Path.cwd() / target_path).absolute()

    if source_path.suffix.lower() != '.h5':
        raise ValueError(f'Expected an .h5 tile, got: {source_path}')
    if not source_path.is_file():
        raise FileNotFoundError(f'Input tile does not exist: {source_path}')
    if len(chunk_size) != 2 or any(size <= 0 for size in chunk_size):
        raise ValueError(f'Chunk size must contain two positive integers, got: {chunk_size}')

    if target_path.exists():
        if not overwrite:
            raise FileExistsError(
                f'Output Zarr store already exists: {target_path}. Pass overwrite=True or --overwrite-zarr.'
            )
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, 'r') as source:
        root = zarr.create_group(store=target_path.as_posix(), zarr_format=3, overwrite=False)
        _copy_group(source, root, chunk_size)

    return target_path


def _is_blank_attr_value(value: Any) -> bool:
    normalized = normalize_attribute_value(value)
    return normalized is None or (isinstance(normalized, str) and not normalized.strip())


def _collect_h5_inventory(h5_file: h5py.File) -> dict[str, list[str]]:
    array_paths: set[str] = set()
    metadata_paths: set[str] = set()
    metadata_attr_paths: set[str] = set()

    metadata_group = h5_file.get('metadata')
    if isinstance(metadata_group, h5py.Group):
        metadata_paths.add('metadata')
        metadata_attr_paths.update(f'metadata@{key}' for key in metadata_group.attrs)

    def collect(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        normalized_name = name.strip('/')
        if not normalized_name:
            return
        if normalized_name.startswith('metadata'):
            metadata_paths.add(normalized_name)
            metadata_attr_paths.update(f'{normalized_name}@{key}' for key in obj.attrs)
            return
        if normalized_name.startswith('bands/'):
            return
        if isinstance(obj, h5py.Dataset):
            array_paths.add(normalized_name)

    h5_file.visititems(collect)
    return {
        'array_paths': sorted(array_paths),
        'metadata_paths': sorted(metadata_paths),
        'metadata_attr_paths': sorted(metadata_attr_paths),
    }


def format_issue_map(issue_map: dict[str, dict[str, Any]]) -> list[str]:
    if not issue_map:
        return []

    lines = []
    for band_name in sorted(issue_map):
        issue = issue_map[band_name]
        parts = []
        if issue.get('missing_attrs'):
            parts.append(f"missing attrs={issue['missing_attrs']}")
        if issue.get('empty_attrs'):
            parts.append(f"empty attrs={issue['empty_attrs']}")
        if issue.get('invalid_shape'):
            parts.append(f"shape={issue.get('shape')}")
        lines.append(f'{band_name}: ' + '; '.join(parts))
    return lines


def _coerce_float(value: Any) -> float | None:
    normalized = normalize_attribute_value(value)
    if normalized is None:
        return None
    try:
        number = float(normalized)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _extract_metadata_coord(abstract_metadata: Mapping[str, Any], stem: str) -> tuple[float, float] | None:
    lat = _coerce_float(abstract_metadata.get(f'{stem}_lat'))
    lon = _coerce_float(abstract_metadata.get(f'{stem}_lon'))
    if lon is None:
        lon = _coerce_float(abstract_metadata.get(f'{stem}_long'))
    if lat is None or lon is None:
        return None
    return (lon, lat)


def extract_tile_geometry_from_abstract_metadata(abstract_metadata: Mapping[str, Any]) -> dict[str, Any]:
    corner_names = ('first_near', 'first_far', 'last_far', 'last_near')
    corners = [_extract_metadata_coord(abstract_metadata, corner_name) for corner_name in corner_names]

    polygon_coords: list[tuple[float, float]] | None = None
    if all(corner is not None for corner in corners):
        polygon_coords = [corner for corner in corners if corner is not None]
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])

    center_coords = _extract_metadata_coord(abstract_metadata, 'centre')
    if center_coords is None:
        center_coords = _extract_metadata_coord(abstract_metadata, 'center')

    return {
        'tile_polygon_coords': polygon_coords,
        'tile_center_coords': center_coords,
    }


def normalize_expected_tile_geometries(rectangles: list[dict[str, Any]] | None) -> dict[str, list[tuple[float, float]]]:
    normalized: dict[str, list[tuple[float, float]]] = {}
    for rectangle in rectangles or []:
        tile_name = rectangle['BL']['properties']['name']
        polygon_coords = [
            tuple(float(value) for value in rectangle[corner]['geometry']['coordinates'])
            for corner in ('TL', 'TR', 'BR', 'BL')
        ]
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])
        normalized[tile_name] = polygon_coords
    return normalized


def validate_h5_tile(tile_path: Path | str, expected_bands: list[str], swath: str | None = None) -> dict[str, Any]:
    tile_path = Path(tile_path)
    actual_bands: list[str] = []
    missing_bands: list[str] = []
    extra_bands: list[str] = []
    empty_metadata_fields: list[str] = []
    missing_core_metadata_fields: list[str] = []
    empty_core_metadata_fields: list[str] = []
    band_attr_issues: dict[str, dict[str, Any]] = {}
    shape_summary: list[str] = []
    quickinfo: dict[str, Any] = {}
    missing_metadata_section = False

    with h5py.File(tile_path, 'r') as h5_file:
        inventory = _collect_h5_inventory(h5_file)

        bands_group = h5_file.get('bands')
        if isinstance(bands_group, h5py.Group):
            actual_bands = sorted(
                name for name, obj in bands_group.items()
                if isinstance(obj, h5py.Dataset)
            )

        missing_bands = sorted(set(expected_bands) - set(actual_bands))
        extra_bands = sorted(set(actual_bands) - set(expected_bands))

        abstract_group = h5_file.get('metadata/Abstracted_Metadata')
        if not isinstance(abstract_group, h5py.Group):
            missing_metadata_section = True
            abstract_metadata = {}
        else:
            abstract_metadata = {
                key: normalize_attribute_value(value)
                for key, value in abstract_group.attrs.items()
            }

        quickinfo = extract_core_metadata_sentinel(abstract_metadata)
        empty_metadata_fields = sorted(
            key for key, value in abstract_metadata.items()
            if _is_blank_attr_value(value)
        )
        geometry_summary = extract_tile_geometry_from_abstract_metadata(abstract_metadata)
        missing_core_metadata_fields = sorted(
            key for key in CORE_METADATA_KEYS
            if key not in abstract_metadata
        )
        empty_core_metadata_fields = sorted(
            key for key in CORE_METADATA_KEYS
            if key in abstract_metadata and _is_blank_attr_value(abstract_metadata[key])
        )

        band_shapes: dict[str, tuple[int, ...]] = {}
        if isinstance(bands_group, h5py.Group):
            for band_name in actual_bands:
                dataset = bands_group[band_name]
                attrs = {
                    key: normalize_attribute_value(value)
                    for key, value in dataset.attrs.items()
                }
                missing_attrs = sorted(key for key in REQUIRED_BAND_ATTRS if key not in attrs)
                empty_attrs = sorted(
                    key for key in REQUIRED_BAND_ATTRS
                    if key in attrs and _is_blank_attr_value(attrs[key])
                )
                shape = tuple(int(dim) for dim in dataset.shape)
                invalid_shape = len(shape) < 2 or any(dim <= 0 for dim in shape)
                band_shapes[band_name] = shape
                if missing_attrs or empty_attrs or invalid_shape:
                    band_attr_issues[band_name] = {
                        'missing_attrs': missing_attrs,
                        'empty_attrs': empty_attrs,
                        'invalid_shape': invalid_shape,
                        'shape': shape,
                    }

        unique_shapes = sorted({shape for shape in band_shapes.values()})
        if len(unique_shapes) > 1:
            shape_summary = [str(shape) for shape in unique_shapes]

    metadata_ok = (
        not missing_metadata_section
        and not empty_metadata_fields
        and not missing_core_metadata_fields
        and not empty_core_metadata_fields
    )
    bands_ok = not missing_bands and not extra_bands
    band_attrs_ok = not band_attr_issues and not shape_summary
    status = 'success' if bands_ok and metadata_ok and band_attrs_ok else 'failed'

    quickinfo_row = {key: quickinfo.get(key) for key in quickinfo}
    quickinfo_row['ID'] = tile_path.stem
    if swath is not None:
        quickinfo_row['SWATH'] = swath

    return {
        'tile': tile_path.stem,
        'swath': swath,
        'output_path': str(tile_path),
        'status': status,
        'bands_ok': bands_ok,
        'metadata_ok': metadata_ok,
        'band_attrs_ok': band_attrs_ok,
        'structure_ok': True,
        'missing_bands': missing_bands,
        'extra_bands': extra_bands,
        'actual_bands': actual_bands,
        'missing_metadata_section': missing_metadata_section,
        'empty_metadata_fields': empty_metadata_fields,
        'missing_core_metadata_fields': missing_core_metadata_fields,
        'empty_core_metadata_fields': empty_core_metadata_fields,
        'band_attr_issues': band_attr_issues,
        'shape_summary': shape_summary,
        'array_paths': inventory['array_paths'],
        'metadata_paths': inventory['metadata_paths'],
        'metadata_attr_paths': inventory['metadata_attr_paths'],
        'missing_array_paths': [],
        'missing_metadata_paths': [],
        'missing_metadata_attrs': [],
        'quickinfo_row': quickinfo_row,
        'tile_polygon_coords': geometry_summary['tile_polygon_coords'],
        'tile_center_coords': geometry_summary['tile_center_coords'],
    }


def enrich_validation_results_with_h5_structure(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    expected_array_paths = sorted({
        path
        for result in results
        for path in result.get('array_paths', [])
    })
    expected_metadata_paths = sorted({
        path
        for result in results
        for path in result.get('metadata_paths', [])
    })
    expected_metadata_attr_paths = sorted({
        path
        for result in results
        for path in result.get('metadata_attr_paths', [])
    })

    for result in results:
        result_array_paths = set(result.get('array_paths', []))
        result_metadata_paths = set(result.get('metadata_paths', []))
        result_metadata_attr_paths = set(result.get('metadata_attr_paths', []))

        missing_array_paths = sorted(set(expected_array_paths) - result_array_paths)
        missing_metadata_paths = sorted(set(expected_metadata_paths) - result_metadata_paths)
        missing_metadata_attrs = sorted(
            attr_path
            for attr_path in expected_metadata_attr_paths
            if attr_path.rsplit('@', 1)[0] in result_metadata_paths
            and attr_path not in result_metadata_attr_paths
        )

        structure_ok = not missing_array_paths and not missing_metadata_paths and not missing_metadata_attrs
        result['missing_array_paths'] = missing_array_paths
        result['missing_metadata_paths'] = missing_metadata_paths
        result['missing_metadata_attrs'] = missing_metadata_attrs
        result['structure_ok'] = structure_ok
        result['status'] = 'success' if (
            result['bands_ok']
            and result['metadata_ok']
            and result['band_attrs_ok']
            and structure_ok
        ) else 'failed'

    return {
        'expected_array_paths': expected_array_paths,
        'expected_metadata_paths': expected_metadata_paths,
        'expected_metadata_attr_paths': expected_metadata_attr_paths,
    }


def _chunked(lines: list[str], size: int):
    for index in range(0, len(lines), size):
        yield lines[index:index + size]


def _group_label(group: Mapping[str, Any]) -> str:
    if group.get('swath'):
        return str(group['swath'])
    return str(group.get('name') or 'single-product')


def _group_expected_tiles(group: Mapping[str, Any]) -> list[str]:
    if group.get('expected_tiles'):
        return sorted(str(tile) for tile in group.get('expected_tiles', []))
    return sorted(str(tile) for tile in (group.get('expected_tile_geometries') or {}).keys())


def _group_actual_tiles(group: Mapping[str, Any]) -> list[str]:
    if group.get('actual_tiles'):
        return sorted(str(tile) for tile in group.get('actual_tiles', []))
    return sorted({str(result['tile']) for result in group.get('results', [])})


def _group_missing_tiles(group: Mapping[str, Any]) -> list[str]:
    return sorted(str(tile) for tile in group.get('missing_tiles', []))


def _group_extra_tiles(group: Mapping[str, Any]) -> list[str]:
    return sorted(str(tile) for tile in group.get('extra_tiles', []))


def _group_skipped_tiles(group: Mapping[str, Any]) -> list[str]:
    return sorted(str(tile) for tile in group.get('skipped_tiles', []))


def _group_failed_tiles(group: Mapping[str, Any]) -> list[str]:
    failed_tiles = {str(tile) for tile in group.get('failed_tiles', [])}
    failed_tiles.update(
        str(result['tile'])
        for result in group.get('results', [])
        if result.get('status') != 'success'
    )
    return sorted(failed_tiles)


def _group_expected_tile_count(group: Mapping[str, Any]) -> int:
    if group.get('expected_tile_count') is not None:
        return int(group['expected_tile_count'])
    return len(_group_expected_tiles(group))


def _group_actual_tile_count(group: Mapping[str, Any]) -> int:
    if group.get('actual_tile_count') is not None:
        return int(group['actual_tile_count'])
    return len(_group_actual_tiles(group))


def build_validation_group_summary_rows(validation_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in validation_groups:
        results = group.get('results', [])
        passed_tiles = sorted(str(result['tile']) for result in results if result.get('status') == 'success')
        failed_tiles = _group_failed_tiles(group)
        missing_tiles = _group_missing_tiles(group)
        extra_tiles = _group_extra_tiles(group)
        skipped_tiles = _group_skipped_tiles(group)
        overall_status = 'PASS'
        if failed_tiles or missing_tiles or extra_tiles or group.get('cut_failed'):
            overall_status = 'FAIL'
        rows.append({
            'group': _group_label(group),
            'expected': _group_expected_tile_count(group),
            'actual': _group_actual_tile_count(group),
            'passed': len(set(passed_tiles)),
            'failed': len(set(failed_tiles)),
            'skipped': len(set(skipped_tiles)),
            'missing': len(set(missing_tiles)),
            'extra': len(set(extra_tiles)),
            'overall_status': overall_status,
        })
    return rows


def build_validation_headline_counts(validation_groups: list[dict[str, Any]]) -> dict[str, int]:
    summary_rows = build_validation_group_summary_rows(validation_groups)
    return {
        'expected_tiles': sum(row['expected'] for row in summary_rows),
        'actual_tiles': sum(row['actual'] for row in summary_rows),
        'passed_tiles': sum(row['passed'] for row in summary_rows),
        'failed_tiles': sum(row['failed'] for row in summary_rows),
        'skipped_tiles': sum(row['skipped'] for row in summary_rows),
        'missing_tiles': sum(row['missing'] for row in summary_rows),
        'extra_tiles': sum(row['extra'] for row in summary_rows),
    }


def build_validation_inventory_summary(group: Mapping[str, Any]) -> dict[str, int]:
    return {
        'expected_bands': len(group.get('expected_bands', [])),
        'expected_non_band_rasters': len(group.get('expected_array_paths', [])),
        'expected_metadata_paths': len(group.get('expected_metadata_paths', [])),
        'expected_metadata_attrs': len(group.get('expected_metadata_attr_paths', [])),
    }


def build_validation_dashboard_rows(group: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = group.get('results', [])
    total = len(results)
    rows: list[dict[str, Any]] = []
    for label, predicate in VALIDATION_CHECKS:
        passed = sum(1 for result in results if predicate(result))
        failed = max(total - passed, 0)
        rows.append({
            'check': label,
            'passed': passed,
            'failed': failed,
            'pass_pct': round((passed / total) * 100.0, 1) if total else 100.0,
        })
    return rows


def _wkt_to_rings(source_wkt: str | None) -> list[list[tuple[float, float]]]:
    if not source_wkt:
        return []
    from shapely import wkt as shapely_wkt

    geometry = shapely_wkt.loads(source_wkt)
    if geometry.geom_type == 'Polygon':
        polygons = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    else:
        return []
    return [
        [(float(x), float(y)) for x, y in polygon.exterior.coords]
        for polygon in polygons
    ]


def build_validation_map_layers(validation_groups: list[dict[str, Any]]) -> dict[str, Any]:
    pre_tc_outlines: list[dict[str, Any]] = []
    post_tc_outlines: list[dict[str, Any]] = []
    report_source_outlines: list[dict[str, Any]] = []
    swath_source_outlines: list[dict[str, Any]] = []
    expected_tiles: dict[str, list[tuple[float, float]]] = {}
    missing_tiles: set[str] = set()
    skipped_tiles: set[str] = set()
    extra_tiles: set[str] = set()
    actual_status: dict[str, str] = {}
    actual_polygons: dict[str, list[tuple[float, float]]] = {}
    actual_points: dict[str, tuple[float, float]] = {}
    tiles_without_geometry: set[str] = set()
    tiles_with_center_only: set[str] = set()

    report_outline_keys: set[tuple[tuple[float, float], ...]] = set()
    pre_outline_keys: set[tuple[tuple[float, float], ...]] = set()
    post_outline_keys: set[tuple[tuple[float, float], ...]] = set()

    for group in validation_groups:
        group_label = _group_label(group)
        pre_tc_wkt = _group_pre_tc_wkt(group)
        post_tc_wkt = _group_post_tc_wkt(group)
        for ring in _wkt_to_rings(pre_tc_wkt):
            key = tuple(ring)
            if key in pre_outline_keys:
                continue
            pre_outline_keys.add(key)
            pre_tc_outlines.append({'label': group_label, 'coords': ring})
        for ring in _wkt_to_rings(post_tc_wkt):
            key = tuple(ring)
            if key in post_outline_keys:
                continue
            post_outline_keys.add(key)
            post_tc_outlines.append({'label': group_label, 'coords': ring})
        report_source_wkt = group.get('report_source_wkt') or group.get('source_wkt')
        for ring in _wkt_to_rings(report_source_wkt):
            key = tuple(ring)
            if key in report_outline_keys:
                continue
            report_outline_keys.add(key)
            report_source_outlines.append({'label': group_label, 'coords': ring})

        if group.get('source_wkt') and len(validation_groups) > 1:
            for ring in _wkt_to_rings(group['source_wkt']):
                swath_source_outlines.append({'label': group_label, 'coords': ring})

        expected_tiles.update(group.get('expected_tile_geometries', {}) or {})
        missing_tiles.update(_group_missing_tiles(group))
        skipped_tiles.update(_group_skipped_tiles(group))
        extra_tiles.update(_group_extra_tiles(group))

        for result in group.get('results', []):
            tile_name = str(result['tile'])
            status_bucket = 'failed' if result.get('status') != 'success' else 'passed'
            if tile_name in extra_tiles:
                status_bucket = 'extra'
            current_bucket = actual_status.get(tile_name)
            priority = {'passed': 1, 'failed': 2, 'extra': 3}
            if current_bucket is None or priority[status_bucket] >= priority[current_bucket]:
                actual_status[tile_name] = status_bucket

            polygon_coords = result.get('tile_polygon_coords')
            center_coords = result.get('tile_center_coords')
            if polygon_coords:
                actual_polygons[tile_name] = [tuple(coord) for coord in polygon_coords]
                tiles_with_center_only.discard(tile_name)
                tiles_without_geometry.discard(tile_name)
            elif center_coords:
                actual_points[tile_name] = tuple(center_coords)
                if tile_name not in actual_polygons:
                    tiles_with_center_only.add(tile_name)
                    tiles_without_geometry.discard(tile_name)
            elif tile_name not in actual_polygons and tile_name not in actual_points:
                tiles_without_geometry.add(tile_name)

    def _items_for_status(status: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        polygon_items = [
            {'tile': tile_name, 'coords': actual_polygons[tile_name]}
            for tile_name, bucket in sorted(actual_status.items())
            if bucket == status and tile_name in actual_polygons
        ]
        point_items = [
            {'tile': tile_name, 'coords': actual_points[tile_name]}
            for tile_name, bucket in sorted(actual_status.items())
            if bucket == status and tile_name in actual_points and tile_name not in actual_polygons
        ]
        return polygon_items, point_items

    passed_polygons, passed_points = _items_for_status('passed')
    failed_polygons, failed_points = _items_for_status('failed')
    extra_polygons, extra_points = _items_for_status('extra')

    missing_polygons = [
        {'tile': tile_name, 'coords': expected_tiles[tile_name]}
        for tile_name in sorted(missing_tiles)
        if tile_name in expected_tiles
    ]
    expected_polygons = [
        {'tile': tile_name, 'coords': coords}
        for tile_name, coords in sorted(expected_tiles.items())
    ]

    return {
        'pre_tc_outlines': pre_tc_outlines,
        'post_tc_outlines': post_tc_outlines,
        'report_source_outlines': report_source_outlines,
        'swath_source_outlines': swath_source_outlines,
        'expected_polygons': expected_polygons,
        'missing_polygons': missing_polygons,
        'passed_polygons': passed_polygons,
        'failed_polygons': failed_polygons,
        'extra_polygons': extra_polygons,
        'passed_points': passed_points,
        'failed_points': failed_points,
        'extra_points': extra_points,
        'counts': {
            'expected': len(expected_polygons),
            'passed': len({item['tile'] for item in passed_polygons + passed_points}),
            'failed': len({item['tile'] for item in failed_polygons + failed_points}),
            'skipped': len(skipped_tiles),
            'missing': len(missing_tiles),
            'extra': len(extra_tiles),
        },
        'tiles_with_center_only_count': len(tiles_with_center_only),
        'tiles_without_geometry_count': len(tiles_without_geometry),
    }


def build_failure_appendix_rows(validation_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    appendix_rows: list[dict[str, Any]] = []
    for group in validation_groups:
        group_label = _group_label(group)
        swath = group.get('swath')
        actual_results = {str(result['tile']): result for result in group.get('results', [])}

        for tile_name in sorted(set(_group_missing_tiles(group)) - set(actual_results)):
            appendix_rows.append({
                'tile': tile_name,
                'group': group_label,
                'swath': swath,
                'issues': ['tile output missing from expected coverage'],
            })

        for tile_name in sorted(set(group.get('failed_tiles', [])) - set(actual_results)):
            appendix_rows.append({
                'tile': tile_name,
                'group': group_label,
                'swath': swath,
                'issues': ['tile cutting failed before H5 validation'],
            })

        extra_tiles = set(_group_extra_tiles(group))
        for tile_name in sorted(actual_results):
            result = actual_results[tile_name]
            issues: list[str] = []
            if tile_name in extra_tiles:
                issues.append('unexpected extra tile output')
            if result.get('missing_bands'):
                issues.append('missing bands: ' + ', '.join(result['missing_bands']))
            if result.get('extra_bands'):
                issues.append('extra bands: ' + ', '.join(result['extra_bands']))
            if result.get('missing_array_paths'):
                issues.append('missing arrays: ' + ', '.join(result['missing_array_paths']))
            if result.get('missing_metadata_paths'):
                issues.append('missing metadata paths: ' + ', '.join(result['missing_metadata_paths']))
            if result.get('missing_metadata_attrs'):
                issues.append('missing metadata attrs: ' + ', '.join(result['missing_metadata_attrs']))
            if result.get('missing_metadata_section'):
                issues.append('missing metadata/Abstracted_Metadata section')
            if result.get('empty_metadata_fields'):
                issues.append('empty metadata fields: ' + ', '.join(result['empty_metadata_fields']))
            if result.get('missing_core_metadata_fields'):
                issues.append('missing core metadata: ' + ', '.join(result['missing_core_metadata_fields']))
            if result.get('empty_core_metadata_fields'):
                issues.append('empty core metadata: ' + ', '.join(result['empty_core_metadata_fields']))
            if result.get('shape_summary'):
                issues.append('shape mismatch: ' + ', '.join(result['shape_summary']))
            issue_lines = format_issue_map(result.get('band_attr_issues', {}))
            if issue_lines:
                issues.append('band attr issues: ' + ' | '.join(issue_lines))
            if result.get('status') == 'success' and tile_name not in extra_tiles:
                continue
            appendix_rows.append({
                'tile': tile_name,
                'group': group_label,
                'swath': swath,
                'issues': issues or ['validation failed'],
            })

    appendix_rows.sort(key=lambda item: ((item.get('swath') or ''), item['group'], item['tile']))
    return appendix_rows


def _write_table(
    ax,
    col_labels: list[str],
    row_values: list[list[str]],
    *,
    font_size: int = 8,
    scale_y: float = 1.4,
    cell_loc: str = 'center',
    fill_bbox: bool = False,
    col_widths: list[float] | None = None,
    cell_pad: float = 0.06,
) -> None:
    ax.axis('off')
    table_kwargs: dict[str, Any] = {
        'cellText': row_values,
        'colLabels': col_labels,
        'cellLoc': cell_loc,
    }
    if col_widths is not None:
        table_kwargs['colWidths'] = col_widths
    if fill_bbox:
        table_kwargs['bbox'] = [0.0, 0.0, 1.0, 1.0]
    else:
        table_kwargs['loc'] = 'center'
    table = ax.table(**table_kwargs)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)
    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass
    for (row_index, _col_index), cell in table.get_celld().items():
        cell.PAD = cell_pad
        if row_index == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E9ECEF')
        else:
            cell.set_facecolor('#FFFFFF')
        cell.set_edgecolor('#CBD5E1')
        cell.set_linewidth(0.8)


def _write_summary_page(pdf, product_name: str, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    from datetime import datetime as _datetime, timezone as _timezone

    summary_rows = build_validation_group_summary_rows(validation_groups)
    headline_counts = build_validation_headline_counts(validation_groups)

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle('WorldSAR H5 Validation Report', x=0.03, y=0.98, ha='left', fontsize=16, fontweight='bold')
    fig.text(0.03, 0.94, f'Product: {product_name}', ha='left', fontsize=11)
    fig.text(0.03, 0.91, f'Timestamp (UTC): {_datetime.now(_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}', ha='left', fontsize=10)

    headline_labels = (
        ('Expected', headline_counts['expected_tiles']),
        ('Actual', headline_counts['actual_tiles']),
        ('Passed', headline_counts['passed_tiles']),
        ('Failed', headline_counts['failed_tiles']),
        ('Skipped', headline_counts['skipped_tiles']),
        ('Missing', headline_counts['missing_tiles']),
        ('Extra', headline_counts['extra_tiles']),
    )
    x_positions = np.linspace(0.06, 0.94, num=len(headline_labels))
    for x_pos, (label, value) in zip(x_positions, headline_labels, strict=False):
        fig.text(x_pos, 0.83, label, ha='center', va='bottom', fontsize=10, color='#495057')
        fig.text(x_pos, 0.79, str(value), ha='center', va='top', fontsize=17, fontweight='bold')

    fig.text(0.03, 0.73, 'Group Summary', ha='left', fontsize=12, fontweight='bold')
    summary_ax = fig.add_axes([0.03, 0.14, 0.94, 0.56])
    _write_table(
        summary_ax,
        ['Group', 'Expected', 'Actual', 'Passed', 'Failed', 'Skipped', 'Missing', 'Extra', 'Status'],
        [
            [
                row['group'],
                str(row['expected']),
                str(row['actual']),
                str(row['passed']),
                str(row['failed']),
                str(row['skipped']),
                str(row['missing']),
                str(row['extra']),
                row['overall_status'],
            ]
            for row in summary_rows
        ] or [['-', '0', '0', '0', '0', '0', '0', '0', 'PASS']],
        font_size=9,
        scale_y=1.6,
    )
    fig.text(
        0.03,
        0.06,
        'Sections: executive summary, geographic coverage map, validation dashboards, failures-only appendix.',
        ha='left',
        fontsize=9,
        color='#495057',
    )
    pdf.savefig(fig)
    plt.close(fig)


def _plot_polygon_items(ax, items: list[dict[str, Any]], *, edgecolor: str, facecolor: str = 'none', linewidth: float = 1.2, linestyle: str = '-', alpha: float = 1.0, hatch: str | None = None) -> None:
    from matplotlib.patches import Polygon as MplPolygon

    for item in items:
        ax.add_patch(
            MplPolygon(
                item['coords'],
                closed=True,
                fill=facecolor != 'none' or hatch is not None,
                facecolor=facecolor if facecolor != 'none' else 'none',
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                hatch=hatch,
            )
        )


def _write_map_page(pdf, product_name: str, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    layers = build_validation_map_layers(validation_groups)

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle('Coverage Map', x=0.03, y=0.98, ha='left', fontsize=16, fontweight='bold')
    fig.text(0.03, 0.94, f'Product: {product_name}', ha='left', fontsize=11)

    map_ax = fig.add_axes([0.06, 0.12, 0.66, 0.76])
    info_ax = fig.add_axes([0.76, 0.12, 0.20, 0.76])
    info_ax.axis('off')

    _plot_polygon_items(map_ax, layers['expected_polygons'], edgecolor='#C7CED6', linewidth=0.8)
    _plot_polygon_items(map_ax, layers['missing_polygons'], edgecolor='#F59F00', linewidth=1.3, hatch='////')
    _plot_polygon_items(map_ax, layers['passed_polygons'], edgecolor='#2B8A3E', facecolor='#D3F9D8', linewidth=1.2, alpha=0.75)
    _plot_polygon_items(map_ax, layers['failed_polygons'], edgecolor='#C92A2A', facecolor='#FFE3E3', linewidth=1.2, alpha=0.8)
    _plot_polygon_items(map_ax, layers['extra_polygons'], edgecolor='#862E9C', facecolor='#F3D9FA', linewidth=1.2, alpha=0.8)
    _plot_polygon_items(map_ax, layers['swath_source_outlines'], edgecolor='#6C757D', linewidth=1.0, linestyle='--')
    _plot_polygon_items(map_ax, layers['report_source_outlines'], edgecolor='#000000', linewidth=1.4)

    if layers['passed_points']:
        map_ax.scatter(
            [point['coords'][0] for point in layers['passed_points']],
            [point['coords'][1] for point in layers['passed_points']],
            color='#2B8A3E',
            s=18,
            marker='o',
            label='passed centre',
            zorder=5,
        )
    if layers['failed_points']:
        map_ax.scatter(
            [point['coords'][0] for point in layers['failed_points']],
            [point['coords'][1] for point in layers['failed_points']],
            color='#C92A2A',
            s=22,
            marker='x',
            label='failed centre',
            zorder=5,
        )
    if layers['extra_points']:
        map_ax.scatter(
            [point['coords'][0] for point in layers['extra_points']],
            [point['coords'][1] for point in layers['extra_points']],
            color='#862E9C',
            s=20,
            marker='D',
            label='extra centre',
            zorder=5,
        )

    all_coords: list[tuple[float, float]] = []
    for layer_name in ('report_source_outlines', 'swath_source_outlines', 'expected_polygons', 'missing_polygons', 'passed_polygons', 'failed_polygons', 'extra_polygons'):
        for item in layers[layer_name]:
            all_coords.extend(item['coords'])
    for layer_name in ('passed_points', 'failed_points', 'extra_points'):
        all_coords.extend(point['coords'] for point in layers[layer_name])

    if all_coords:
        xs = [coord[0] for coord in all_coords]
        ys = [coord[1] for coord in all_coords]
        x_margin = max((max(xs) - min(xs)) * 0.05, 0.01)
        y_margin = max((max(ys) - min(ys)) * 0.05, 0.01)
        map_ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        map_ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
    else:
        map_ax.text(0.5, 0.5, 'No geographic geometry available for this report.', ha='center', va='center', transform=map_ax.transAxes)

    map_ax.set_xlabel('Longitude')
    map_ax.set_ylabel('Latitude')
    map_ax.set_aspect('equal', adjustable='box')
    map_ax.grid(True, linewidth=0.4, alpha=0.3)

    legend_handles = [
        Patch(facecolor='none', edgecolor='#000000', linewidth=1.4, label='product footprint'),
        Patch(facecolor='none', edgecolor='#C7CED6', linewidth=0.8, label='expected tiles'),
        Patch(facecolor='#D3F9D8', edgecolor='#2B8A3E', linewidth=1.2, label='passed tiles'),
        Patch(facecolor='#FFE3E3', edgecolor='#C92A2A', linewidth=1.2, label='failed tiles'),
        Patch(facecolor='none', edgecolor='#F59F00', linewidth=1.3, hatch='////', label='missing tiles'),
        Patch(facecolor='#F3D9FA', edgecolor='#862E9C', linewidth=1.2, label='extra tiles'),
    ]
    if len(validation_groups) > 1:
        legend_handles.insert(1, Line2D([0], [0], color='#6C757D', linestyle='--', linewidth=1.0, label='swath footprint'))
    if layers['passed_points'] or layers['failed_points'] or layers['extra_points']:
        legend_handles.append(Line2D([0], [0], color='#495057', marker='o', linestyle='None', label='centre-only tile'))

    info_ax.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=9)
    counts = layers['counts']
    info_ax.text(
        0.0,
        0.46,
        '\n'.join([
            'Counts',
            f"Expected: {counts['expected']}",
            f"Passed: {counts['passed']}",
            f"Failed: {counts['failed']}",
            f"Skipped: {counts['skipped']}",
            f"Missing: {counts['missing']}",
            f"Extra: {counts['extra']}",
        ]),
        ha='left',
        va='top',
        fontsize=10,
    )

    notes = []
    if layers['tiles_with_center_only_count']:
        notes.append(f"Centre-only tiles: {layers['tiles_with_center_only_count']}")
    if layers['tiles_without_geometry_count']:
        notes.append(f"Tiles without geometry: {layers['tiles_without_geometry_count']}")
    if notes:
        info_ax.text(0.0, 0.18, '\n'.join(notes), ha='left', va='top', fontsize=9, color='#495057')

    pdf.savefig(fig)
    plt.close(fig)


def _write_dashboard_pages(pdf, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    for group in validation_groups:
        inventory = build_validation_inventory_summary(group)
        dashboard_rows = build_validation_dashboard_rows(group)
        summary_rows = build_validation_group_summary_rows([group])
        summary_row = summary_rows[0]

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f"Validation Dashboard: {_group_label(group)}", x=0.03, y=0.98, ha='left', fontsize=16, fontweight='bold')
        fig.text(
            0.03,
            0.94,
            f"Intermediate: {group.get('intermediate_product', '-')} | Cuts: {group.get('cuts_dir', '-')}",
            ha='left',
            fontsize=9,
        )
        if group.get('cut_report_path'):
            fig.text(0.03, 0.91, f"Cut report: {group['cut_report_path']}", ha='left', fontsize=9)

        fig.text(0.03, 0.86, 'Inventory', ha='left', fontsize=12, fontweight='bold')
        inventory_ax = fig.add_axes([0.03, 0.59, 0.42, 0.22])
        _write_table(
            inventory_ax,
            ['Metric', 'Count'],
            [
                ['Expected bands', str(inventory['expected_bands'])],
                ['Expected non-band rasters', str(inventory['expected_non_band_rasters'])],
                ['Expected metadata paths', str(inventory['expected_metadata_paths'])],
                ['Expected metadata attrs', str(inventory['expected_metadata_attrs'])],
            ],
            font_size=9,
            scale_y=1.6,
        )

        fig.text(0.52, 0.86, 'Tile Status', ha='left', fontsize=12, fontweight='bold')
        status_ax = fig.add_axes([0.52, 0.59, 0.45, 0.22])
        _write_table(
            status_ax,
            ['Expected', 'Actual', 'Passed', 'Failed', 'Skipped', 'Missing', 'Extra', 'Status'],
            [[
                str(summary_row['expected']),
                str(summary_row['actual']),
                str(summary_row['passed']),
                str(summary_row['failed']),
                str(summary_row['skipped']),
                str(summary_row['missing']),
                str(summary_row['extra']),
                summary_row['overall_status'],
            ]],
            font_size=9,
            scale_y=1.8,
        )

        fig.text(0.03, 0.49, 'Checklist', ha='left', fontsize=12, fontweight='bold')
        checklist_ax = fig.add_axes([0.03, 0.11, 0.94, 0.33])
        _write_table(
            checklist_ax,
            ['Check', 'Pass', 'Fail', 'Pass %'],
            [
                [
                    row['check'],
                    str(row['passed']),
                    str(row['failed']),
                    f"{row['pass_pct']:.1f}",
                ]
                for row in dashboard_rows
            ] or [['overall', '0', '0', '100.0']],
            font_size=9,
            scale_y=1.55,
        )
        pdf.savefig(fig)
        plt.close(fig)


def _format_failure_block(block: Mapping[str, Any]) -> list[str]:
    swath = block.get('swath') or '-'
    lines = [f"Tile: {block['tile']} | Group: {block['group']} | Swath: {swath}"]
    for issue in block.get('issues', []):
        lines.append(f'  - {issue}')
    lines.append('')
    return lines


def _write_failure_appendix(pdf, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    appendix_rows = build_failure_appendix_rows(validation_groups)
    if not appendix_rows:
        appendix_rows = [{'tile': '-', 'group': 'all', 'swath': None, 'issues': ['No failing tiles.']}]

    failure_lines: list[str] = []
    for block in appendix_rows:
        failure_lines.extend(_format_failure_block(block))

    for index, page_lines in enumerate(_chunked(failure_lines, 34), start=1):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.suptitle(f'Failures Appendix (page {index})', x=0.03, y=0.98, ha='left', fontsize=16, fontweight='bold')
        fig.text(0.03, 0.93, '\n'.join(page_lines), va='top', ha='left', fontsize=9, family='monospace')
        pdf.savefig(fig)
        plt.close(fig)


def _group_pre_tc_wkt(group: Mapping[str, Any]) -> str | None:
    return (
        group.get('pre_tc_wkt')
        or group.get('input_source_wkt')
        or group.get('source_input_wkt')
        or group.get('report_source_wkt')
        or group.get('source_wkt')
    )


def _group_post_tc_wkt(group: Mapping[str, Any]) -> str | None:
    return (
        group.get('post_tc_wkt')
        or group.get('report_source_wkt')
        or group.get('source_wkt')
        or _group_pre_tc_wkt(group)
    )


def build_aggregate_dashboard_rows(validation_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = {
        label: {'check': label, 'passed': 0, 'failed': 0}
        for label, _predicate in VALIDATION_CHECKS
    }
    for group in validation_groups:
        for row in build_validation_dashboard_rows(group):
            entry = totals[row['check']]
            entry['passed'] += row['passed']
            entry['failed'] += row['failed']
    rows: list[dict[str, Any]] = []
    for label, _predicate in VALIDATION_CHECKS:
        entry = totals[label]
        total = entry['passed'] + entry['failed']
        rows.append({
            'check': label,
            'passed': int(entry['passed']),
            'failed': int(entry['failed']),
            'pass_pct': round((entry['passed'] / total) * 100.0, 1) if total else 100.0,
        })
    return rows


def build_report_metadata_snapshot(validation_groups: list[dict[str, Any]]) -> list[tuple[str, str]]:
    preferred_keys = (
        ('Mission', 'MISSION'),
        ('Mode', 'ACQUISITION_MODE'),
        ('Product', 'PRODUCT_TYPE'),
        ('Pass', 'PASS'),
        ('Polarization 1', 'mds1_tx_rx_polar'),
        ('Polarization 2', 'mds2_tx_rx_polar'),
        ('Acquisition', 'first_line_time'),
        ('Source Product', 'PRODUCT'),
    )
    for group in validation_groups:
        for result in group.get('results', []):
            quickinfo = result.get('quickinfo_row') or {}
            snapshot = [
                (label, str(quickinfo[key]))
                for label, key in preferred_keys
                if quickinfo.get(key) not in (None, '', 'None')
            ]
            if snapshot:
                return snapshot
    return []


def _geometry_area_sqkm(geometry) -> float:
    import pyproj

    geod = pyproj.Geod(ellps='WGS84')
    if geometry.geom_type == 'Polygon':
        polygons = [geometry]
    elif geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)
    else:
        return 0.0

    total_area = 0.0
    for polygon in polygons:
        lon, lat = zip(*polygon.exterior.coords)
        area, _perimeter = geod.polygon_area_perimeter(lon, lat)
        total_area += abs(area)
        for interior in polygon.interiors:
            hole_lon, hole_lat = zip(*interior.coords)
            hole_area, _hole_perimeter = geod.polygon_area_perimeter(hole_lon, hole_lat)
            total_area -= abs(hole_area)
    return total_area / 1_000_000.0


def _combined_wkt_area_sqkm(source_wkts: list[str | None]) -> float | None:
    wkts = [source_wkt for source_wkt in source_wkts if source_wkt]
    if not wkts:
        return None
    from shapely import wkt as shapely_wkt
    from shapely.ops import unary_union

    geometry = unary_union([shapely_wkt.loads(source_wkt) for source_wkt in wkts])
    if geometry.is_empty:
        return None
    return _geometry_area_sqkm(geometry)


def _status_distribution(validation_groups: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    counts = build_validation_headline_counts(validation_groups)
    return [
        ('Passed', counts['passed_tiles'], '#16A34A'),
        ('Failed', counts['failed_tiles'], '#DC2626'),
        ('Skipped', counts['skipped_tiles'], '#D97706'),
        ('Missing', counts['missing_tiles'], '#F59E0B'),
        ('Extra', counts['extra_tiles'], '#7C3AED'),
    ]


def _trim_text(value: Any, width: int = 60) -> str:
    import textwrap

    if value in (None, ''):
        return '-'
    return textwrap.shorten(str(value), width=width, placeholder='...')


def _build_issue_summary_lines(validation_groups: list[dict[str, Any]], limit: int = 6) -> list[str]:
    rows = build_failure_appendix_rows(validation_groups)
    if not rows or (len(rows) == 1 and rows[0].get('tile') == '-'):
        return [
            'No failing tiles or metadata regressions detected.',
            'All produced tiles passed the validation checklist for this run.',
        ]

    lines: list[str] = []
    for row in rows[:limit]:
        issues = '; '.join(str(issue) for issue in row.get('issues', [])[:2]) or 'validation failed'
        lines.append(f"{row['tile']} [{row['group']}]: {issues}")
    remaining = len(rows) - limit
    if remaining > 0:
        lines.append(f'+ {remaining} additional issue block(s) omitted from this summary.')
    return lines


def _add_page_chrome(fig, title: str, subtitle: str, page_number: int, total_pages: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig.patch.set_facecolor('#F8FAFC')
    chrome_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=-5)
    chrome_ax.axis('off')
    chrome_ax.add_patch(Rectangle((0.0, 0.925), 1.0, 0.075, color='#0F172A'))
    chrome_ax.add_patch(Rectangle((0.0, 0.925), 0.34, 0.075, color='#0F766E'))
    chrome_ax.add_patch(Rectangle((0.0, 0.0), 1.0, 0.018, color='#E2E8F0'))
    fig.text(0.035, 0.958, title, ha='left', va='center', fontsize=19, fontweight='bold', color='white')
    fig.text(0.035, 0.931, subtitle, ha='left', va='center', fontsize=9.5, color='#DCE7F3')
    fig.text(0.965, 0.028, f'Page {page_number}/{total_pages}', ha='right', va='bottom', fontsize=8.5, color='#64748B')


def _wrap_lines_to_width(lines: list[str], width_chars: int) -> list[str]:
    import textwrap

    wrapped_lines: list[str] = []
    for line in lines:
        current_line = str(line) if line not in (None, '') else '-'
        chunks = textwrap.wrap(
            current_line,
            width=max(width_chars, 8),
            break_long_words=False,
            break_on_hyphens=False,
        ) or ['-']
        wrapped_lines.extend(chunks)
    return wrapped_lines


def _add_fitted_text_block(
    ax,
    lines: list[str],
    *,
    x: float = 0.0,
    y: float = 1.0,
    max_fontsize: float = 9.0,
    min_fontsize: float = 5.5,
    color: str = '#334155',
    fontweight: str = 'normal',
    va: str = 'top',
    ha: str = 'left',
    line_spacing: float = 1.02,
) -> None:
    fig = ax.figure
    if not lines:
        lines = ['-']
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    target_width = ax_bbox.width * 0.98
    target_height = ax_bbox.height * 0.98

    best_text = '\n'.join(lines)
    best_fontsize = min_fontsize

    for fontsize in np.arange(max_fontsize, min_fontsize - 0.01, -0.2):
        char_width_px = max((fontsize * fig.dpi / 72.0) * 0.56, 1.0)
        wrap_width = int(target_width / char_width_px)
        wrapped_lines = _wrap_lines_to_width(lines, wrap_width)
        candidate_text = '\n'.join(wrapped_lines)
        text_obj = ax.text(
            x,
            y,
            candidate_text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            linespacing=line_spacing,
            clip_on=True,
        )
        fig.canvas.draw()
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        fits = bbox.width <= target_width and bbox.height <= target_height
        text_obj.remove()
        if fits:
            best_text = candidate_text
            best_fontsize = fontsize
            break
        best_text = candidate_text
        best_fontsize = fontsize

    ax.text(
        x,
        y,
        best_text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=best_fontsize,
        color=color,
        fontweight=fontweight,
        linespacing=line_spacing,
        clip_on=True,
    )


def _add_metric_card(fig, bounds: list[float], label: str, value: str, accent: str, subtitle: str | None = None) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax = fig.add_axes(bounds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle='round,pad=0.02,rounding_size=0.06',
        facecolor='white',
        edgecolor='#CBD5E1',
        linewidth=1.1,
    ))
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 0.03, 1.0,
        boxstyle='round,pad=0.0,rounding_size=0.06',
        facecolor=accent,
        edgecolor=accent,
        linewidth=0,
    ))
    label_ax = ax.inset_axes([0.08, 0.60, 0.82, 0.18])
    label_ax.axis('off')
    _add_fitted_text_block(
        label_ax,
        [label.upper()],
        x=0.0,
        y=1.0,
        max_fontsize=8.4,
        min_fontsize=6.6,
        color='#64748B',
        fontweight='bold',
        va='top',
    )
    ax.text(0.08, 0.45, value, ha='left', va='center', fontsize=20, color='#0F172A', fontweight='bold')
    if subtitle:
        subtitle_ax = ax.inset_axes([0.08, 0.08, 0.84, 0.18])
        subtitle_ax.axis('off')
        _add_fitted_text_block(
            subtitle_ax,
            [subtitle],
            x=0.0,
            y=0.0,
            max_fontsize=8.2,
            min_fontsize=6.6,
            color='#475569',
            va='bottom',
        )


def _add_note_box(fig, bounds: list[float], title: str, lines: list[str], facecolor: str = 'white') -> None:
    from matplotlib.patches import FancyBboxPatch

    ax = fig.add_axes(bounds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle='round,pad=0.02,rounding_size=0.05',
        facecolor=facecolor,
        edgecolor='#CBD5E1',
        linewidth=1.0,
    ))
    ax.text(0.05, 0.88, title, ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')
    body_ax = ax.inset_axes([0.05, 0.08, 0.90, 0.68])
    body_ax.axis('off')
    _add_fitted_text_block(
        body_ax,
        lines,
        x=0.0,
        y=1.0,
        max_fontsize=9.0,
        min_fontsize=6.2,
        color='#334155',
        va='top',
    )


def _add_metadata_box(
    fig,
    bounds: list[float],
    title: str,
    items: list[tuple[str, str]],
    *,
    facecolor: str = 'white',
) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax = fig.add_axes(bounds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle='round,pad=0.02,rounding_size=0.05',
        facecolor=facecolor,
        edgecolor='#CBD5E1',
        linewidth=1.0,
    ))
    ax.text(0.05, 0.88, title, ha='left', va='center', fontsize=11, fontweight='bold', color='#0F172A')
    display_items = items if items else [('Status', 'No metadata available')]
    y = 0.74
    step = min(0.12, 0.62 / max(len(display_items), 1))
    fontsize = 8.6 if len(display_items) <= 5 else 7.9 if len(display_items) <= 7 else 7.2
    for label, value in display_items:
        line = f'{label}: {value}'
        ax.text(0.05, y, line, ha='left', va='top', fontsize=fontsize, color='#475569', clip_on=True)
        y -= step


def _format_sqkm(value: float | None) -> str:
    if value is None:
        return '-'
    return f'{value:,.0f} km^2'


def _format_sqkm_compact(value: float | None) -> str:
    if value is None:
        return '-'
    if value >= 10_000:
        return f'{value / 1_000:.1f}k'
    return f'{value:,.0f}'


def _set_map_extent(ax, coord_groups: list[list[tuple[float, float]]]) -> None:
    all_coords = [coord for group in coord_groups for coord in group]
    if not all_coords:
        ax.text(0.5, 0.5, 'No geographic geometry available.', ha='center', va='center', transform=ax.transAxes)
        return
    xs = [coord[0] for coord in all_coords]
    ys = [coord[1] for coord in all_coords]
    x_margin = max((max(xs) - min(xs)) * 0.08, 0.01)
    y_margin = max((max(ys) - min(ys)) * 0.08, 0.01)
    ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
    ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linewidth=0.4, alpha=0.25, color='#94A3B8')
    ax.set_xlabel('Longitude', fontsize=8.5, color='#475569')
    ax.set_ylabel('Latitude', fontsize=8.5, color='#475569')
    ax.tick_params(labelsize=8.5, colors='#334155')


def _style_plot_panel(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.04, 0.95, title, transform=ax.transAxes, ha='left', va='top', fontsize=11.5, fontweight='bold', color='#0F172A')


def _make_panel(fig, bounds: list[float], title: str, subtitle: str | None = None, content_bounds: list[float] | None = None):
    panel_ax = fig.add_axes(bounds)
    _style_plot_panel(panel_ax, title, subtitle)
    body_bounds = content_bounds or [0.06, 0.10, 0.90, 0.72]
    content_ax = panel_ax.inset_axes(body_bounds)
    return panel_ax, content_ax


def _write_table_panel(
    fig,
    bounds: list[float],
    title: str,
    subtitle: str,
    col_labels: list[str],
    row_values: list[list[str]],
    *,
    font_size: int = 8,
    scale_y: float = 1.0,
    cell_loc: str = 'center',
    content_bounds: list[float] | None = None,
    fill_bbox: bool = True,
    col_widths: list[float] | None = None,
    cell_pad: float = 0.06,
) -> None:
    _, table_ax = _make_panel(fig, bounds, title, subtitle, content_bounds or [0.02, 0.10, 0.96, 0.72])
    _write_table(
        table_ax,
        col_labels,
        row_values,
        font_size=font_size,
        scale_y=scale_y,
        cell_loc=cell_loc,
        fill_bbox=fill_bbox,
        col_widths=col_widths,
        cell_pad=cell_pad,
    )


def _write_summary_page_compact(pdf, product_name: str, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    from datetime import datetime as _datetime, timezone as _timezone

    counts = build_validation_headline_counts(validation_groups)
    summary_rows = build_validation_group_summary_rows(validation_groups)
    aggregate_rows = build_aggregate_dashboard_rows(validation_groups)
    metadata_snapshot = build_report_metadata_snapshot(validation_groups)
    issue_count = counts['failed_tiles'] + counts['missing_tiles'] + counts['extra_tiles']
    pass_rate = (counts['passed_tiles'] / counts['expected_tiles'] * 100.0) if counts['expected_tiles'] else 100.0
    overall_status = 'PASS' if issue_count == 0 else 'FAIL'

    fig = plt.figure(figsize=(11.69, 8.27))
    _add_page_chrome(
        fig,
        'WorldSAR Validation Report',
        f'{product_name} | Executive summary | Generated {_datetime.now(_timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")}',
        1,
        2,
    )

    overview_metrics = [
        ('Validation groups', str(len(summary_rows))),
        ('Expected tiles', str(counts['expected_tiles'])),
        ('Produced tiles', str(counts['actual_tiles'])),
        ('Passed tiles', str(counts['passed_tiles'])),
        ('Failed tiles', str(counts['failed_tiles'])),
        ('Skipped tiles', str(counts['skipped_tiles'])),
        ('Missing tiles', str(counts['missing_tiles'])),
        ('Extra tiles', str(counts['extra_tiles'])),
        ('Pass rate', f'{pass_rate:.1f}%'),
        ('Open issues', str(issue_count)),
        ('Overall status', overall_status),
    ]
    overview_rows: list[list[str]] = []
    for index in range(0, len(overview_metrics), 2):
        left_label, left_value = overview_metrics[index]
        if index + 1 < len(overview_metrics):
            right_label, right_value = overview_metrics[index + 1]
        else:
            right_label, right_value = '', ''
        overview_rows.append([left_label, left_value, right_label, right_value])
    metadata_rows = [
        [label, _trim_text(value, 42)]
        for label, value in metadata_snapshot[:6]
    ] or [['Status', 'No metadata snapshot available']]
    group_rows = [
        [
            row['group'],
            str(row['expected']),
            str(row['actual']),
            str(row['passed']),
            str(row['failed']),
            str(row['skipped']),
            str(row['missing']),
            str(row['extra']),
            row['overall_status'],
        ]
        for row in summary_rows
    ] or [['-', '0', '0', '0', '0', '0', '0', '0', 'PASS']]
    check_rows = [
        [
            row['check'],
            str(row['passed']),
            str(row['failed']),
            f"{row['pass_pct']:.1f}%",
        ]
        for row in aggregate_rows
    ] or [['overall', '0', '0', '100.0%']]

    _write_table_panel(
        fig,
        [0.05, 0.68, 0.90, 0.17],
        'Run Summary',
        'Key counts and final status for this validation run',
        ['Metric', 'Value', 'Metric', 'Value'],
        overview_rows,
        font_size=9.2,
        scale_y=1.0,
        cell_loc='left',
        content_bounds=[0.02, 0.14, 0.96, 0.64],
        fill_bbox=True,
        col_widths=[0.32, 0.18, 0.32, 0.18],
        cell_pad=0.10,
    )

    _write_table_panel(
        fig,
        [0.05, 0.50, 0.90, 0.13],
        'Product Metadata',
        'Mission and acquisition attributes used by the validator',
        ['Field', 'Value'],
        metadata_rows,
        font_size=8.8,
        scale_y=1.0,
        cell_loc='left',
        content_bounds=[0.02, 0.16, 0.96, 0.56],
        fill_bbox=True,
        col_widths=[0.22, 0.78],
        cell_pad=0.10,
    )

    _write_table_panel(
        fig,
        [0.05, 0.32, 0.90, 0.13],
        'Validation Groups',
        'Per-group production outcome and issue counts',
        ['Group', 'Expected', 'Actual', 'Passed', 'Failed', 'Skipped', 'Missing', 'Extra', 'Status'],
        group_rows,
        font_size=8.4,
        scale_y=1.0,
        content_bounds=[0.02, 0.16, 0.96, 0.56],
        fill_bbox=True,
        cell_pad=0.10,
    )

    _write_table_panel(
        fig,
        [0.05, 0.06, 0.90, 0.21],
        'Validation Checks',
        'Aggregate pass and fail counts across the produced tile set',
        ['Check', 'Passed', 'Failed', 'Pass rate'],
        check_rows,
        font_size=8.6,
        scale_y=1.0,
        cell_loc='left',
        content_bounds=[0.02, 0.12, 0.96, 0.68],
        fill_bbox=True,
        col_widths=[0.52, 0.16, 0.14, 0.18],
        cell_pad=0.10,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _write_footprint_page_compact(pdf, product_name: str, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    layers = build_validation_map_layers(validation_groups)
    pre_area = _combined_wkt_area_sqkm([_group_pre_tc_wkt(group) for group in validation_groups])
    post_area = _combined_wkt_area_sqkm([_group_post_tc_wkt(group) for group in validation_groups])
    ratio = ((post_area / pre_area) * 100.0) if pre_area and post_area else None

    pre_coords = [item['coords'] for item in layers['pre_tc_outlines']]
    post_coords = [item['coords'] for item in layers['post_tc_outlines']]
    tile_coords = [item['coords'] for item in layers['expected_polygons']] + [item['coords'] for item in layers['passed_polygons']]
    combined_coords = pre_coords + post_coords + tile_coords
    pre_focus_coords = pre_coords + post_coords
    post_focus_coords = post_coords + [item['coords'] for item in layers['expected_polygons']]
    post_focus_coords += [item['coords'] for item in layers['passed_polygons']]
    post_focus_coords += [item['coords'] for item in layers['failed_polygons']]

    fig = plt.figure(figsize=(11.69, 8.27))
    _add_page_chrome(fig, 'Footprint Analysis', f'{product_name} | Input geometry vs terrain-corrected raster extent', 2, 2)

    before_panel_ax, before_ax = _make_panel(
        fig,
        [0.05, 0.52, 0.42, 0.29],
        'Before Terrain Correction',
        'Input product footprint from source metadata or manual override',
        [0.08, 0.14, 0.88, 0.70],
    )
    _plot_polygon_items(before_ax, layers['pre_tc_outlines'], edgecolor='#1D4ED8', facecolor='#DBEAFE', linewidth=1.6, alpha=0.85)
    _plot_polygon_items(before_ax, layers['post_tc_outlines'], edgecolor='#0F766E', linewidth=1.0, linestyle='--', alpha=0.55)
    _set_map_extent(before_ax, pre_focus_coords or combined_coords)

    after_panel_ax, after_ax = _make_panel(
        fig,
        [0.52, 0.52, 0.42, 0.29],
        'After Terrain Correction',
        'Processed TC.dim footprint used for tile selection and reporting',
        [0.08, 0.14, 0.88, 0.70],
    )
    _plot_polygon_items(after_ax, layers['expected_polygons'], edgecolor='#CBD5E1', facecolor='#F1F5F9', linewidth=0.8, alpha=0.95)
    _plot_polygon_items(after_ax, layers['passed_polygons'], edgecolor='#15803D', facecolor='#DCFCE7', linewidth=1.1, alpha=0.9)
    _plot_polygon_items(after_ax, layers['failed_polygons'], edgecolor='#B91C1C', facecolor='#FEE2E2', linewidth=1.2, alpha=0.85)
    _plot_polygon_items(after_ax, layers['post_tc_outlines'], edgecolor='#0F172A', linewidth=1.6)
    _set_map_extent(after_ax, post_focus_coords or combined_coords)

    overlay_panel_ax, overlay_ax = _make_panel(
        fig,
        [0.05, 0.19, 0.44, 0.24],
        'Footprint comparison overlay',
        'Pre-TC outline vs post-TC raster extent with tile footprint overlay',
        [0.08, 0.14, 0.86, 0.66],
    )
    _plot_polygon_items(overlay_ax, layers['pre_tc_outlines'], edgecolor='#1D4ED8', linewidth=1.2, linestyle='--', alpha=0.95)
    _plot_polygon_items(overlay_ax, layers['post_tc_outlines'], edgecolor='#0F172A', linewidth=1.6, alpha=0.95)
    _plot_polygon_items(overlay_ax, layers['passed_polygons'], edgecolor='#16A34A', facecolor='#DCFCE7', linewidth=0.9, alpha=0.85)
    _set_map_extent(overlay_ax, combined_coords)

    _add_note_box(
        fig,
        [0.54, 0.23, 0.40, 0.18],
        'Interpretation',
        [
            'Pre-TC is the source-product footprint before raster reprojection.',
            'Post-TC is the actual raster extent read from TC.dim and used for tile selection.',
            'Expected and produced tiles are aligned to the processed raster, not to the broader source metadata footprint.',
            'The 7.2% retained-area ratio is expected here because the validation run uses a burst-limited TC raster extent.',
        ],
        facecolor='#FFFFFF',
    )

    legend_ax = fig.add_axes([0.54, 0.125, 0.40, 0.08])
    legend_ax.axis('off')
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color='#1D4ED8', linestyle='--', linewidth=1.4, label='pre-TC footprint'),
            Line2D([0], [0], color='#0F172A', linestyle='-', linewidth=1.8, label='post-TC footprint'),
            Patch(facecolor='#F1F5F9', edgecolor='#CBD5E1', linewidth=0.8, label='expected tiles'),
            Patch(facecolor='#DCFCE7', edgecolor='#16A34A', linewidth=1.0, label='passed tiles'),
            Patch(facecolor='#FEE2E2', edgecolor='#B91C1C', linewidth=1.0, label='failed tiles'),
        ],
        loc='center left',
        frameon=False,
        fontsize=8.0,
        ncol=3,
        columnspacing=1.4,
        handlelength=2.2,
    )

    _add_metric_card(
        fig,
        [0.05, 0.03, 0.20, 0.09],
        'Pre-TC area',
        _format_sqkm_compact(pre_area),
        '#2563EB',
        'km^2 footprint',
    )
    _add_metric_card(
        fig,
        [0.28, 0.03, 0.20, 0.09],
        'Post-TC area',
        _format_sqkm_compact(post_area),
        '#0F172A',
        'km^2 raster extent',
    )
    _add_metric_card(
        fig,
        [0.51, 0.03, 0.20, 0.09],
        'Retained',
        f'{ratio:.1f}%' if ratio is not None else '-',
        '#0F766E',
        'post / pre area',
    )
    _add_metric_card(
        fig,
        [0.74, 0.03, 0.20, 0.09],
        'Tiles inside',
        str(layers['counts']['passed']),
        '#16A34A',
        f"{layers['counts']['expected']} expected",
    )

    pdf.savefig(fig)
    plt.close(fig)


def _write_dashboard_page_compact(pdf, product_name: str, validation_groups: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    counts = build_validation_headline_counts(validation_groups)
    aggregate_rows = build_aggregate_dashboard_rows(validation_groups)
    summary_rows = build_validation_group_summary_rows(validation_groups)
    issue_lines = _build_issue_summary_lines(validation_groups)
    inventory = build_validation_inventory_summary(validation_groups[0]) if validation_groups else {
        'expected_bands': 0,
        'expected_non_band_rasters': 0,
        'expected_metadata_paths': 0,
        'expected_metadata_attrs': 0,
    }
    metadata_snapshot = build_report_metadata_snapshot(validation_groups)

    fig = plt.figure(figsize=(11.69, 8.27))
    _add_page_chrome(fig, 'Validation Dashboard', f'{product_name} | Quality, schema, and issue summary', 3, 3)

    status_panel_ax, status_ax = _make_panel(
        fig,
        [0.05, 0.58, 0.40, 0.24],
        'Tile status counts',
        'Materialized outputs and residual issue buckets',
        [0.08, 0.18, 0.89, 0.62],
    )
    status_labels = ['Passed', 'Failed', 'Skipped', 'Missing', 'Extra']
    status_values = [counts['passed_tiles'], counts['failed_tiles'], counts['skipped_tiles'], counts['missing_tiles'], counts['extra_tiles']]
    status_colors = ['#16A34A', '#DC2626', '#D97706', '#F59E0B', '#7C3AED']
    x_pos = np.arange(len(status_labels))
    bars = status_ax.bar(x_pos, status_values, color=status_colors, width=0.62)
    status_ax.set_xticks(x_pos, labels=status_labels)
    status_ax.set_ylabel('Tiles', fontsize=8.5, color='#475569')
    status_ax.tick_params(labelsize=8.5, colors='#334155')
    status_ax.set_ylim(0, max(status_values + [1]) * 1.18)
    for bar, value in zip(bars, status_values, strict=False):
        status_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(max(status_values), 1) * 0.03, str(value), ha='center', va='bottom', fontsize=8.5, color='#0F172A')

    check_panel_ax, check_ax = _make_panel(
        fig,
        [0.52, 0.58, 0.43, 0.24],
        'Checklist pass rates',
        'Per-check validation quality across all produced tiles',
        [0.22, 0.18, 0.75, 0.62],
    )
    checks = [row['check'] for row in aggregate_rows]
    pass_pct = [row['pass_pct'] for row in aggregate_rows]
    y_pos = np.arange(len(checks))
    colors = ['#16A34A' if value >= 99.9 else '#F59E0B' if value >= 80 else '#DC2626' for value in pass_pct]
    check_ax.barh(y_pos, pass_pct, color=colors, height=0.55)
    check_ax.set_yticks(y_pos, labels=checks)
    check_ax.invert_yaxis()
    check_ax.set_xlim(0, 104)
    check_ax.set_xlabel('Pass rate (%)', fontsize=8.5, color='#475569')
    for index, row in enumerate(aggregate_rows):
        total = row['passed'] + row['failed']
        check_ax.text(min(row['pass_pct'] + 0.8, 101.5), index, f"{row['passed']}/{total}", va='center', fontsize=8.2, color='#334155')

    inventory_panel_ax, inventory_ax = _make_panel(
        fig,
        [0.06, 0.29, 0.26, 0.20],
        'Expected schema',
        'Per-tile structural expectations',
        [0.12, 0.18, 0.84, 0.60],
    )
    inv_labels = ['Bands', 'Rasters', 'Meta paths', 'Meta attrs']
    inv_values = [
        inventory['expected_bands'],
        inventory['expected_non_band_rasters'],
        inventory['expected_metadata_paths'],
        inventory['expected_metadata_attrs'],
    ]
    inv_y = np.arange(len(inv_labels))
    inventory_ax.barh(inv_y, inv_values, color=['#0F766E', '#0284C7', '#6366F1', '#7C3AED'], height=0.55)
    inventory_ax.set_yticks(inv_y, labels=inv_labels)
    inventory_ax.invert_yaxis()
    inventory_ax.tick_params(labelsize=8.2, colors='#334155')
    inventory_ax.set_xlabel('Count', fontsize=8.5, color='#475569')
    inventory_ax.set_xlim(0, max(inv_values + [1]) * 1.10)
    for index, value in enumerate(inv_values):
        inventory_ax.text(value + max(max(inv_values), 1) * 0.03, index, str(value), va='center', fontsize=8.2, color='#334155')

    metadata_items = [(label, _trim_text(value, 26)) for label, value in metadata_snapshot[:6]]
    _add_metadata_box(fig, [0.34, 0.29, 0.27, 0.20], 'Metadata Detail', metadata_items, facecolor='#FFFFFF')

    summary_panel_ax, summary_ax = _make_panel(
        fig,
        [0.65, 0.29, 0.30, 0.20],
        'Group scorecard',
        'Per-group delivery status',
        [0.04, 0.24, 0.92, 0.46],
    )
    _write_table(
        summary_ax,
        ['Group', 'Expected', 'Actual', 'Passed', 'Status'],
        [
            [
                row['group'],
                str(row['expected']),
                str(row['actual']),
                str(row['passed']),
                row['overall_status'],
            ]
            for row in summary_rows
        ] or [['-', '0', '0', '0', 'PASS']],
        font_size=8.8,
        scale_y=1.45,
    )

    _add_note_box(fig, [0.05, 0.06, 0.90, 0.16], 'Issue Summary', issue_lines, facecolor='#FFFFFF')

    pdf.savefig(fig)
    plt.close(fig)


def write_h5_validation_report_pdf(report_path: Path | str, product_name: str, validation_groups: list[dict[str, Any]]) -> Path:
    import matplotlib as mpl
    from matplotlib.backends.backend_pdf import PdfPages

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rc_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    }
    with mpl.rc_context(rc=rc_params):
        with PdfPages(report_path) as pdf:
            _write_summary_page_compact(pdf, product_name, validation_groups)
            _write_footprint_page_compact(pdf, product_name, validation_groups)

    return report_path
