"""Legacy compatibility entrypoint for tests and older scripts."""

from __future__ import annotations

import argparse
def create_parser() -> argparse.ArgumentParser:
    from sarpyx.cli.worldsar import add_worldsar_arguments

    parser = argparse.ArgumentParser(description="Legacy WorldSAR compatibility entrypoint.")
    add_worldsar_arguments(parser)
    return parser


def main() -> None:
    from sarpyx.cli.worldsar import main as worldsar_main

    worldsar_main()


def merge_iq_into_pdec(*args, **kwargs):
    from sarpyx.cli.worldsar import merge_iq_into_pdec as impl

    return impl(*args, **kwargs)


def _apply_sentinel_orbit_file(
    op,
    orbit_type='Sentinel Precise (Auto Download)',
    orbit_continue_on_fail=False,
):
    orbit_product = op.ApplyOrbitFile(
        orbit_type=orbit_type,
        continue_on_fail=orbit_continue_on_fail,
    )
    if orbit_product is not None:
        return orbit_product

    error_summary = op.last_error_summary()
    normalized_error = error_summary.lower()
    offline_orbit_failure_markers = (
        'network is unreachable',
        'unable to connect to http://step.esa.int/auxdata/orbits/',
        'unable to connect to https://step.esa.int/auxdata/orbits/',
    )
    missing_orbit_file_markers = (
        'no valid orbit file found',
        'orbit files may be downloaded from copernicus dataspaces',
    )
    recoverable_offline_failure = (
        any(marker in normalized_error for marker in offline_orbit_failure_markers)
        or all(marker in normalized_error for marker in missing_orbit_file_markers)
    )
    if orbit_continue_on_fail or recoverable_offline_failure:
        print(f'WARNING: Apply-Orbit-File failed but continuing without orbit correction: {error_summary}')
        return op.prod_path

    raise RuntimeError(f'Apply-Orbit-File failed: {error_summary}')


def _sentinel_post_chain(
    op,
    product_path,
    orbit_type='Sentinel Precise (Auto Download)',
    orbit_continue_on_fail=False,
    sentinel_tc_source_band=None,
):
    fp_orb = _apply_sentinel_orbit_file(
        op,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
    )
    fp_cal = op.Calibration(output_complex=True)
    if fp_cal is None:
        raise RuntimeError(f'Calibration failed: {op.last_error_summary()}')
    fp_deramp = op.TopsarDerampDemod()
    if fp_deramp is None:
        raise RuntimeError(f'TOPSAR-DerampDemod failed: {op.last_error_summary()}')
    fp_deb = op.Deburst()
    if fp_deb is None:
        raise RuntimeError(f'TOPSAR-Deburst failed: {op.last_error_summary()}')

    op.do_subaps(
        dim_path=op.prod_path,
        safe_path=product_path,
        n_decompositions=[2],
        byte_order=1,
        VERBOSE=False,
        update_dim=False,
        tops_iw_mode=True,
        iw_apply_spectrum_normalization=False,
        iw_energy_compensation=True,
        iw_flip_output=True,
        iw_row_equalization=False,
        iw_doppler_centroid_correction=True,
        iw_dc_smooth_win=129,
        iw_equal_energy_split=True,
        iw_crosslook_row_balance=True,
        iw_crosslook_row_balance_smooth_win=257,
        iw_crosslook_row_balance_clip=1.5,
    )

    fp_pdec = op.polarimetric_decomposition(decomposition="H-Alpha Dual Pol Decomposition", window_size=5)
    if fp_pdec is None:
        raise RuntimeError(f'Polarimetric decomposition failed: {op.last_error_summary()}')
    try:
        merge_iq_into_pdec(
            src_dim=fp_deb,
            pdec_dim=fp_pdec,
            is_tops=True,
            overwrite_copied_files=False,
            backup=False,
        )
    except ModuleNotFoundError as exc:
        if getattr(exc, 'name', None) != 'merge_iq_into_pdec':
            raise
        raise RuntimeError(
            'merge_iq_into_pdec module is required for TOPS flow. '
            'TOPS fallback to DIM metadata rewrite is disabled to avoid malformed DEB metadata.'
        )
    fp_tc = op.TerrainCorrection(
        map_projection='AUTO:42001',
        pixel_spacing_in_meter=10.0,
        source_bands=[sentinel_tc_source_band] if sentinel_tc_source_band else None,
        save_selected_source_band=True,
    )
    if fp_tc is None:
        raise RuntimeError(f'Terrain Correction failed: {op.last_error_summary()}')
    return op.prod_path


def pipeline_sentinel(
    product_path, output_dir, is_TOPS=False,
    gpt_memory=None, gpt_parallelism=None, gpt_timeout=None,
    orbit_type='Sentinel Precise (Auto Download)',
    orbit_continue_on_fail=False,
    sentinel_swath=None,
    sentinel_first_burst=1,
    sentinel_last_burst=9999,
    sentinel_tc_source_band=None,
    **_,
):
    from pathlib import Path

    gpt_kw = dict(gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout)
    op = _create_gpt_operator(product_path, output_dir, 'BEAM-DIMAP', **gpt_kw)

    if is_TOPS:
        results = {}
        swaths = (sentinel_swath,) if sentinel_swath else ('IW1', 'IW2', 'IW3')
        for swath in swaths:
            sw_op = _create_gpt_operator(Path(op.prod_path), output_dir / swath, 'BEAM-DIMAP', **gpt_kw)
            split_result = sw_op.TopsarSplit(
                subswath=swath,
                first_burst_index=sentinel_first_burst,
                last_burst_index=sentinel_last_burst,
            )
            if split_result is None:
                raise RuntimeError(f'TOPSAR-Split failed for {swath}: {sw_op.last_error_summary()}')
            if not Path(split_result).exists():
                raise FileNotFoundError(f'TOPSAR-Split output missing for {swath}: {split_result}')
            results[swath] = _sentinel_post_chain(
                op=sw_op,
                product_path=product_path,
                orbit_type=orbit_type,
                orbit_continue_on_fail=orbit_continue_on_fail,
                sentinel_tc_source_band=sentinel_tc_source_band,
            )
        return results

    orbit_product = _apply_sentinel_orbit_file(
        op,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
    )
    fp_cal = op.Calibration(output_complex=True)
    if fp_cal is None:
        raise RuntimeError(f'Calibration failed: {op.last_error_summary()}')

    op.do_subaps(
        safe_path=product_path,
        dim_path=op.prod_path,
        n_decompositions=[3],
        byte_order=1,
        VERBOSE=False,
        update_dim=False,
    )
    fp_pdec = op.polarimetric_decomposition(decomposition="H-Alpha Dual Pol Decomposition", window_size=5)
    if fp_pdec is None:
        raise RuntimeError(f'Polarimetric decomposition failed: {op.last_error_summary()}')
    try:
        merge_iq_into_pdec(
            src_dim=fp_cal,
            pdec_dim=fp_pdec,
            is_tops=False,
            overwrite_copied_files=False,
            backup=False,
        )
    except ModuleNotFoundError as exc:
        if getattr(exc, 'name', None) != 'merge_iq_into_pdec':
            raise
        fp_cal = update_dim_add_bands_from_data_dir(fp_cal, verbose=False)
        fp_merged = op.BandMerge(
            source_products=[fp_pdec, fp_cal],
            output_name=f'{Path(fp_pdec).stem}_MERGED',
        )
        if fp_merged is None:
            raise RuntimeError(f'BandMerge failed: {op.last_error_summary()}')
    fp_tc = op.TerrainCorrection(
        map_projection='AUTO:42001',
        pixel_spacing_in_meter=10.0,
        source_bands=[sentinel_tc_source_band] if sentinel_tc_source_band else None,
        save_selected_source_band=True,
    )
    if fp_tc is None:
        raise RuntimeError(f'Terrain Correction failed: {op.last_error_summary()}')
    return op.prod_path


def _resolve_tiling_wkt(product_wkt, source_product, intermediate_product, product_mode, swath=None):
    from pathlib import Path
    from sarpyx.cli.worldsar import _dim_footprint_wkt

    intermediate_path = Path(intermediate_product)
    if intermediate_path.suffix.lower() == '.dim':
        try:
            return _dim_footprint_wkt(intermediate_path)
        except Exception as exc:
            print(f'[WARN] Failed to derive raster footprint from {intermediate_path}: {type(exc).__name__}: {exc}')
    if product_mode == 'S1TOPS' and swath:
        derived_wkt = sentinel1_swath_wkt_extractor_safe(source_product, swath, display_results=False, verbose=False)
        if derived_wkt:
            return derived_wkt
    return product_wkt


def _run_tops_swath_tiling(product_wkt, grid_geoj_path, product_path, intermediate, cuts_outdir, product_mode, gpt_kwargs):
    swath_tiling_errors = {}
    swath_wkts = {}
    validation_groups = []
    report_name = None

    for swath, swath_product in intermediate.items():
        name = swath_product.stem
        swath_wkt = _resolve_tiling_wkt(product_wkt, product_path, swath_product, product_mode, swath=swath)
        swath_wkts[swath] = swath_wkt

        if tiling:
            tiling_result = _run_tiling(
                swath_wkt, grid_geoj_path, product_path,
                swath_product, cuts_outdir / swath, product_mode, **gpt_kwargs,
            )
            tiling_result['pre_tc_wkt'] = product_wkt
            tiling_result['post_tc_wkt'] = swath_wkt
            name = tiling_result['name']
            report_name = report_name or name
            if tiling_result['cut_failed']:
                swath_tiling_errors[swath] = RuntimeError(f"Tile cutting failed; report: {tiling_result['report_path']}")
            validation_group = _validate_tile_group(
                tiling_result['cuts_dir'],
                swath_product,
                swath=swath,
                tiling_result=tiling_result,
            )
            validation_groups.append(validation_group)
            if validation_group['rows']:
                _run_db_indexing(validation_group['rows'], name, swath=swath)
        else:
            validation_group = _validate_tile_group(
                cuts_outdir / swath / name,
                swath_product,
                swath=swath,
                tiling_result={
                    'source_wkt': swath_wkt,
                    'report_source_wkt': swath_wkt,
                },
            )
            validation_groups.append(validation_group)
            if validation_group['rows']:
                _run_db_indexing(validation_group['rows'], name, swath=swath)

    if validation_groups:
        pdf_path = cuts_outdir / f'{report_name or validation_groups[0]["name"]}_h5_validation_report.pdf'
        _write_h5_validation_report_pdf(pdf_path, report_name or validation_groups[0]['name'], validation_groups)

    if swath_tiling_errors:
        _verify_tops_tile_coverage(
            product_wkt, grid_geoj_path, cuts_outdir, intermediate, swath_wkts=swath_wkts,
        )
    if any(result['status'] != 'success' for group in validation_groups for result in group['results']):
        raise RuntimeError(f'H5 validation failed; report: {pdf_path}')

if __name__ == "__main__":
    main()
