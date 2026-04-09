from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

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


def _write_pdf_text_page(pdf, title: str, lines: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.03, 0.97, title, va='top', ha='left', fontsize=14, fontweight='bold', family='monospace')
    fig.text(0.03, 0.93, '\n'.join(lines), va='top', ha='left', fontsize=8, family='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def write_h5_validation_report_pdf(report_path: Path | str, product_name: str, validation_groups: list[dict[str, Any]]) -> Path:
    from datetime import datetime as _datetime
    from textwrap import wrap

    from matplotlib.backends.backend_pdf import PdfPages

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for group in validation_groups:
        all_results.extend(group['results'])

    passed = sum(1 for result in all_results if result['status'] == 'success')
    failed = len(all_results) - passed
    total_expected_tiles = sum(group.get('expected_tile_count', 0) for group in validation_groups)
    total_actual_tiles = sum(group.get('actual_tile_count', len(group['results'])) for group in validation_groups)
    tiles_with_missing_arrays = sum(1 for result in all_results if result.get('missing_array_paths'))
    tiles_with_missing_metadata_paths = sum(1 for result in all_results if result.get('missing_metadata_paths'))
    tiles_with_missing_metadata_attrs = sum(1 for result in all_results if result.get('missing_metadata_attrs'))

    summary_lines = [
        f'Timestamp (UTC): {_datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}',
        f'Product name: {product_name}',
        f'Validated tile groups: {len(validation_groups)}',
        f'Expected tiles: {total_expected_tiles}',
        f'Actual tiles: {total_actual_tiles}',
        f'Validated H5 files: {len(all_results)}',
        f'Passed tiles: {passed}',
        f'Failed tiles: {failed}',
        f'Tiles with missing non-band arrays: {tiles_with_missing_arrays}',
        f'Tiles with missing metadata paths: {tiles_with_missing_metadata_paths}',
        f'Tiles with missing metadata attrs: {tiles_with_missing_metadata_attrs}',
        '',
        'Group summary:',
    ]
    for group in validation_groups:
        group_failed = sum(1 for result in group['results'] if result['status'] != 'success')
        label = group['swath'] or 'single-product'
        summary_lines.append(
            f"- {label}: expected={group.get('expected_tile_count', len(group['results']))} "
            f"actual={group.get('actual_tile_count', len(group['results']))} "
            f"failed={group_failed} intermediate={group['intermediate_product']}"
        )
        summary_lines.append(f"  cuts={group['cuts_dir']}")
        if group.get('cut_report_path'):
            summary_lines.append(f"  cut_report={group['cut_report_path']}")
        summary_lines.append('  expected bands:')
        for line in wrap(', '.join(group['expected_bands']), width=110):
            summary_lines.append(f'    {line}')
        if group.get('expected_array_paths'):
            summary_lines.append('  expected non-band arrays:')
            for line in wrap(', '.join(group['expected_array_paths']), width=110):
                summary_lines.append(f'    {line}')
        if group.get('expected_metadata_paths'):
            summary_lines.append('  expected metadata paths:')
            for line in wrap(', '.join(group['expected_metadata_paths']), width=110):
                summary_lines.append(f'    {line}')
        if group.get('expected_metadata_attr_paths'):
            summary_lines.append('  expected metadata attrs:')
            for line in wrap(', '.join(group['expected_metadata_attr_paths']), width=110):
                summary_lines.append(f'    {line}')

    table_lines = [
        'tile                 swath      bands  metadata  attrs  struct  overall',
        '-------------------  ---------  -----  --------  -----  ------  -------',
    ]
    for result in sorted(all_results, key=lambda item: ((item.get('swath') or ''), item['tile'])):
        table_lines.append(
            f"{result['tile'][:19]:19}  "
            f"{(result.get('swath') or '-'):9}  "
            f"{'PASS' if result['bands_ok'] else 'FAIL':5}  "
            f"{'PASS' if result['metadata_ok'] else 'FAIL':8}  "
            f"{'PASS' if result['band_attrs_ok'] else 'FAIL':5}  "
            f"{'PASS' if result['structure_ok'] else 'FAIL':6}  "
            f"{'PASS' if result['status'] == 'success' else 'FAIL'}"
        )

    failure_lines = ['No failing tiles.'] if failed == 0 else []
    if failed:
        for result in sorted((item for item in all_results if item['status'] != 'success'), key=lambda item: ((item.get('swath') or ''), item['tile'])):
            failure_lines.extend([
                f"Tile: {result['tile']}  Swath: {result.get('swath') or '-'}",
                f"  Missing bands: {result['missing_bands'] or '[]'}",
                f"  Extra bands: {result['extra_bands'] or '[]'}",
                f"  Missing arrays: {result.get('missing_array_paths') or '[]'}",
                f"  Missing metadata paths: {result.get('missing_metadata_paths') or '[]'}",
                f"  Missing metadata attrs: {result.get('missing_metadata_attrs') or '[]'}",
                f"  Missing metadata section: {result['missing_metadata_section']}",
                f"  Empty metadata fields: {result['empty_metadata_fields'] or '[]'}",
                f"  Missing core metadata: {result['missing_core_metadata_fields'] or '[]'}",
                f"  Empty core metadata: {result['empty_core_metadata_fields'] or '[]'}",
            ])
            if result['shape_summary']:
                failure_lines.append(f"  Shape mismatch: {result['shape_summary']}")
            issue_lines = format_issue_map(result['band_attr_issues'])
            if issue_lines:
                failure_lines.append('  Band attribute issues:')
                failure_lines.extend(f'    - {line}' for line in issue_lines)
            failure_lines.append('')

    with PdfPages(report_path) as pdf:
        for page_lines in _chunked(summary_lines, 42):
            _write_pdf_text_page(pdf, 'WorldSAR H5 validation summary', page_lines)
        for index, page_lines in enumerate(_chunked(table_lines, 44), start=1):
            _write_pdf_text_page(pdf, f'WorldSAR H5 validation table (page {index})', page_lines)
        for index, page_lines in enumerate(_chunked(failure_lines, 40), start=1):
            _write_pdf_text_page(pdf, f'WorldSAR H5 validation failures (page {index})', page_lines)

    return report_path
