"""WorldSAR preprocessing orchestration for package pipeline recipes."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from sarpyx.pipelines.single_product import biomass, nisar, s1_strip, s1_tops, tsx
from sarpyx.snapflow.gpt import create_gpt_operator
from sarpyx.snapflow.product import (
    _read_terrasar_metadata,
    _resolve_terrasar_product_xml_if_supported,
    _terrasar_is_complex,
    _terrasar_is_detected,
    _terrasar_is_geocoded,
)
from sarpyx.snapflow.runtime import PipelineStep, make_context, run_step, run_steps
from sarpyx.snapflow.tiling_runtime import finalize_tops_tiling

ISOLATED_PREPROCESSING_PREFIX = "worldsar_tmp_"
_PREPROCESSING_ROOT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _gpt_kwargs(gpt_memory=None, gpt_parallelism=None, gpt_timeout=None, gpt_cache_size=None) -> dict:
    return dict(gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout, gpt_cache_size=gpt_cache_size)


def _tiling_metadata(
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    product_mode=None,
    gpt_kwargs=None,
    tile_writer="zarr",
    pre_write_hook=None,
    report_outdir=None,
    product_name=None,
) -> dict:
    if grid_path is None or cuts_outdir is None or product_mode is None:
        return {}
    metadata = {
        "grid_path": grid_path,
        "cuts_outdir": cuts_outdir,
        "product_mode": product_mode,
        "gpt_kwargs": dict(gpt_kwargs or {}),
        "tile_writer": tile_writer,
        "pre_write_hook": pre_write_hook,
    }
    if product_wkt is not None:
        metadata["product_wkt"] = product_wkt
    if report_outdir is not None:
        metadata["report_outdir"] = report_outdir
    if product_name is not None:
        metadata["product_name"] = product_name
    return metadata


def _dimap_data_dir(dim_path: Path) -> Path:
    return dim_path.with_suffix(".data")


def _create_isolated_preprocessing_root(output_dir, root_name=None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if root_name is None:
        return Path(tempfile.mkdtemp(prefix=ISOLATED_PREPROCESSING_PREFIX, dir=output_dir))

    root_name = str(root_name)
    if not _PREPROCESSING_ROOT_NAME_RE.fullmatch(root_name):
        raise ValueError(
            "preprocessing root name must be a basename containing only "
            "letters, digits, '.', '_' and '-': "
            f"{root_name!r}"
        )

    root = output_dir / f"{ISOLATED_PREPROCESSING_PREFIX}{root_name}"
    if root.is_symlink():
        raise ValueError(f"refusing to replace preprocessing root symlink: {root}")
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"preprocessing root exists and is not a directory: {root}")
        shutil.rmtree(root)
    root.mkdir()
    return root


def _isolated_preprocessing_root_for_path(path, output_dir) -> Path | None:
    output_root = Path(output_dir).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(output_root)
    except ValueError:
        return None
    if not relative.parts:
        return None
    root = output_root / relative.parts[0]
    if root.name.startswith(ISOLATED_PREPROCESSING_PREFIX):
        return root
    return None


def _is_isolated_preprocessing_path(path, output_dir) -> bool:
    return _isolated_preprocessing_root_for_path(path, output_dir) is not None


def _cleanup_isolated_preprocessing_root(path, output_dir) -> None:
    root = _isolated_preprocessing_root_for_path(path, output_dir)
    if root is not None and root.is_dir():
        shutil.rmtree(root)


def _cleanup_intermediates(output_dir, keep_product, keep_intermediate: bool = True) -> None:
    if keep_intermediate or keep_product is None:
        return
    output_dir = Path(output_dir)
    keep_dim = Path(keep_product).resolve()
    for dim_path in output_dir.glob("*.dim"):
        if dim_path.resolve() == keep_dim:
            continue
        data_dir = _dimap_data_dir(dim_path)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        if dim_path.exists():
            dim_path.unlink()


def _tiling_created_final_tiles(tiling_result) -> bool:
    if not isinstance(tiling_result, dict) or tiling_result.get("error"):
        return False
    cut_result = tiling_result.get("tiling_result") or {}
    actual_tiles = cut_result.get("actual_tiles") or []
    validation = tiling_result.get("validation_group") or {}
    validation_results = validation.get("results") or []
    return bool(actual_tiles) and bool(validation_results) and all(item.get("status") == "success" for item in validation_results)


def run_sentinel_tops_pipeline(
    product_path,
    output_dir,
    gpt_memory=None,
    gpt_parallelism=None,
    gpt_timeout=None,
    gpt_cache_size=None,
    orbit_type="Sentinel Precise (Auto Download)",
    orbit_continue_on_fail=False,
    sentinel_swath=None,
    sentinel_first_burst=None,
    sentinel_last_burst=None,
    sentinel_tc_source_band=None,
    sentinel_subap_decompositions=None,
    create_operator=create_gpt_operator,
    merge_func=None,
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    product_mode="S1TOPS",
    keep_intermediate=True,
    **_,
):
    output_dir = Path(output_dir)
    gpt_kwargs = _gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size)
    recipe = s1_tops.steps(
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
        sentinel_tc_source_band=sentinel_tc_source_band,
        sentinel_subap_decompositions=sentinel_subap_decompositions,
    )
    first_burst = sentinel_first_burst or s1_tops.DEFAULT_FIRST_BURST
    last_burst = sentinel_last_burst or s1_tops.DEFAULT_LAST_BURST
    swath = sentinel_swath or s1_tops.DEFAULT_SWATH
    swaths = (swath,) if swath else s1_tops.DEFAULT_SWATHS
    results = {}
    tiling_results = []
    tiling_metadata = _tiling_metadata(
        product_wkt,
        grid_path,
        cuts_outdir,
        product_mode,
        gpt_kwargs,
        _.get("tile_writer", "zarr"),
        _.get("pre_write_hook"),
        _.get("report_outdir"),
        _.get("product_name"),
    )
    processing_root = (
        _create_isolated_preprocessing_root(output_dir, _.get("preprocessing_root_name"))
        if tiling_metadata
        else output_dir
    )
    base_ctx = make_context(product_path, processing_root, "BEAM-DIMAP", gpt_kwargs, create_operator=create_operator)
    for swath in swaths:
        sw_ctx = make_context(
            Path(base_ctx.op.prod_path),
            processing_root / swath,
            "BEAM-DIMAP",
            gpt_kwargs,
            create_operator=create_operator,
        )
        sw_ctx.original_product = product_path
        sw_ctx.metadata.update(tiling_metadata)
        sw_ctx.metadata["swath"] = swath
        sw_ctx.metadata["keep_intermediate"] = keep_intermediate
        if merge_func is not None:
            sw_ctx.metadata["merge_iq_into_pdec"] = merge_func
        split_result = run_step(
            sw_ctx,
            PipelineStep(
                "TopsarSplit",
                {"subswath": swath, "first_burst_index": first_burst, "last_burst_index": last_burst},
                "split",
            ),
        )
        if not Path(split_result).exists():
            raise FileNotFoundError(f"TOPSAR-Split output missing for {swath}: {split_result}")
        run_steps(sw_ctx, recipe[1:])
        if "tiling" in sw_ctx.saved:
            tiling_results.append(sw_ctx.saved["tiling"])
        results[swath] = sw_ctx.op.prod_path
    if tiling_metadata and tiling_results:
        finalize_tops_tiling(
            product_wkt,
            grid_path,
            cuts_outdir,
            results,
            tiling_results,
            report_outdir=_.get("report_outdir"),
            product_name=_.get("product_name"),
        )
    if (
        tiling_metadata
        and not keep_intermediate
        and tiling_results
        and all(_tiling_created_final_tiles(result) for result in tiling_results)
    ):
        _cleanup_isolated_preprocessing_root(processing_root, output_dir)
    return results


def run_sentinel_strip_pipeline(
    product_path,
    output_dir,
    gpt_memory=None,
    gpt_parallelism=None,
    gpt_timeout=None,
    gpt_cache_size=None,
    orbit_type="Sentinel Precise (Auto Download)",
    orbit_continue_on_fail=False,
    sentinel_tc_source_band=None,
    sentinel_subap_decompositions=None,
    create_operator=create_gpt_operator,
    merge_func=None,
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    product_mode="S1STRIP",
    keep_intermediate=True,
    **_,
):
    gpt_kwargs = _gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size)
    metadata = _tiling_metadata(product_wkt, grid_path, cuts_outdir, product_mode, gpt_kwargs, _.get("tile_writer", "zarr"), _.get("pre_write_hook"), _.get("report_outdir"), _.get("product_name"))
    if merge_func is not None:
        metadata["merge_iq_into_pdec"] = merge_func
    ctx = make_context(
        product_path,
        output_dir,
        "BEAM-DIMAP",
        gpt_kwargs,
        create_operator=create_operator,
        metadata=metadata,
    )
    run_steps(
        ctx,
        s1_strip.steps(
            orbit_type=orbit_type,
            orbit_continue_on_fail=orbit_continue_on_fail,
            sentinel_tc_source_band=sentinel_tc_source_band,
            sentinel_subap_decompositions=sentinel_subap_decompositions,
        ),
    )
    if _tiling_created_final_tiles(ctx.saved.get("tiling")):
        _cleanup_intermediates(ctx.output_dir, ctx.op.prod_path, keep_intermediate)
    return ctx.op.prod_path


def run_tsx_csg_pipeline(
    product_path,
    output_dir,
    gpt_memory=None,
    gpt_parallelism=None,
    gpt_timeout=None,
    gpt_cache_size=None,
    create_operator=create_gpt_operator,
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    product_mode="TSX",
    keep_intermediate=True,
    **_,
):
    product_xml = _resolve_terrasar_product_xml_if_supported(product_path)
    is_terrasar = product_xml is not None
    gpt_product_path = product_xml if is_terrasar else product_path
    metadata = _read_terrasar_metadata(Path(gpt_product_path)) if is_terrasar else {}
    if is_terrasar:
        print(
            "TerraSAR-X/TanDEM-X metadata: "
            f"variant={metadata.get('variant') or 'UNKNOWN'}, "
            f"product_type={metadata.get('product_type') or 'UNKNOWN'}, "
            f"imaging_mode={metadata.get('imaging_mode') or 'UNKNOWN'}, "
            f"image_data_type={metadata.get('image_data_type') or 'UNKNOWN'}, "
            f"projection={metadata.get('projection') or 'UNKNOWN'}"
        )
    geocoded = is_terrasar and _terrasar_is_geocoded(metadata)
    if geocoded:
        print("TerraSAR-X/TanDEM-X geocoded product detected; normalizing to BEAM-DIMAP with Write.")
    elif is_terrasar and not (_terrasar_is_complex(metadata) or _terrasar_is_detected(metadata)):
        print("WARNING: TerraSAR-X/TanDEM-X product variant could not be classified; using complex calibration.")
    output_complex = not (is_terrasar and _terrasar_is_detected(metadata) and not _terrasar_is_complex(metadata))
    ctx = make_context(
        gpt_product_path,
        output_dir,
        "BEAM-DIMAP",
        _gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size),
        create_operator=create_operator,
        metadata=_tiling_metadata(
            product_wkt,
            grid_path,
            cuts_outdir,
            product_mode,
            _gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size),
            _.get("tile_writer", "zarr"),
            _.get("pre_write_hook"),
            _.get("report_outdir"),
            _.get("product_name"),
        ),
    )
    output_file = Path(output_dir) / f"{Path(gpt_product_path).stem}_WRITE.dim"
    run_steps(ctx, tsx.steps(geocoded=geocoded, output_complex=output_complex, output_file=output_file))
    if _tiling_created_final_tiles(ctx.saved.get("tiling")):
        _cleanup_intermediates(ctx.output_dir, ctx.op.prod_path, keep_intermediate)
    return ctx.op.prod_path


def run_biomass_pipeline(
    product_path,
    output_dir,
    gpt_memory=None,
    gpt_parallelism=None,
    gpt_timeout=None,
    gpt_cache_size=None,
    create_operator=create_gpt_operator,
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    product_mode="BM",
    keep_intermediate=True,
    **_,
):
    gpt_kwargs = _gpt_kwargs(gpt_memory, gpt_parallelism, gpt_timeout, gpt_cache_size)
    ctx = make_context(
        product_path,
        output_dir,
        "GDAL-GTiff-WRITER",
        gpt_kwargs,
        create_operator=create_operator,
        metadata=_tiling_metadata(product_wkt, grid_path, cuts_outdir, product_mode, gpt_kwargs, _.get("tile_writer", "zarr"), _.get("pre_write_hook"), _.get("report_outdir"), _.get("product_name")),
    )
    run_steps(ctx, biomass.steps())
    if _tiling_created_final_tiles(ctx.saved.get("tiling")):
        _cleanup_intermediates(ctx.output_dir, ctx.op.prod_path, keep_intermediate)
    return ctx.op.prod_path


def run_nisar_pipeline(product_path, output_dir=None, product_wkt=None, grid_path=None, cuts_outdir=None, product_mode="NISAR", **_):
    if Path(product_path).suffix.lower() != ".h5":
        raise ValueError("NISAR products must be in .h5 format.")
    ctx = make_context(
        product_path,
        output_dir or Path(product_path).parent,
        "HDF5",
        {},
        metadata=_tiling_metadata(product_wkt, grid_path, cuts_outdir, product_mode, {}, _.get("tile_writer", "zarr"), _.get("pre_write_hook"), _.get("report_outdir"), _.get("product_name")),
    )
    run_steps(ctx, nisar.steps())
    return product_path
