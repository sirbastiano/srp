"""WorldSAR CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from sarpyx.hooks.subap_features import SubapFeatureConfig
from sarpyx.hooks.worldsar import make_worldsar_zarr_tile_hook, product_output_name
from sarpyx.utils.worldsar_h5 import convert_tile_h5_to_zarr
from sarpyx.snapflow import config
from sarpyx.snapflow.locks import worldsar_product_lock
from sarpyx.snapflow.tile_writers import normalize_tile_writer
from sarpyx.pipelines.single_product import s1_tops
from sarpyx.snapflow.preprocessing import (
    ISOLATED_PREPROCESSING_PREFIX,
    _is_isolated_preprocessing_path,
    run_biomass_pipeline,
    run_nisar_pipeline,
    run_sentinel_strip_pipeline,
    run_sentinel_tops_pipeline,
    run_tsx_csg_pipeline,
)
from sarpyx.snapflow.product import infer_product_mode, resolve_product_wkt
from sarpyx.snapflow.runtime import PipelineContext, PipelineStep, run_step

ROUTER = {
    "S1TOPS": run_sentinel_tops_pipeline,
    "S1STRIP": run_sentinel_strip_pipeline,
    "TSX": run_tsx_csg_pipeline,
    "CSG": run_tsx_csg_pipeline,
    "BM": run_biomass_pipeline,
    "NISAR": run_nisar_pipeline,
}


def _find_existing_intermediates(output_dir: Path, product_mode: str, sentinel_swath: str | None = None) -> dict | Path:
    if product_mode == "S1TOPS":
        result = {}
        swath = sentinel_swath or s1_tops.DEFAULT_SWATH
        swaths = (swath,) if swath else s1_tops.DEFAULT_SWATHS
        for swath in swaths:
            swath_dirs = [
                output_dir / swath,
                *sorted(output_dir.glob(f"{ISOLATED_PREPROCESSING_PREFIX}*/{swath}")),
            ]
            dims = sorted(
                (dim for swath_dir in swath_dirs for dim in swath_dir.glob("*.dim")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not dims:
                raise FileNotFoundError(f"No .dim intermediate found for {swath} under {output_dir}")
            if len(dims) > 1:
                print(f"[WARN] Multiple .dim files for {swath} under {output_dir}, using most recent: {dims[0].name}")
            result[swath] = dims[0]
            print(f"Reusing intermediate {swath}: {dims[0]}")
        return result
    dims = sorted(output_dir.glob("*.dim"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dims:
        raise FileNotFoundError(f"No .dim intermediate found in {output_dir}")
    print(f"Reusing intermediate: {dims[0]}")
    return dims[0]


def _run_preprocessing(
    product_path,
    output_dir,
    product_mode,
    orbit_type,
    orbit_continue_on_fail,
    gpt_memory,
    gpt_parallelism,
    gpt_timeout,
    gpt_cache_size,
    sentinel_swath=None,
    sentinel_first_burst=None,
    sentinel_last_burst=None,
    sentinel_tc_source_band=None,
    sentinel_subap_decompositions=None,
    product_wkt=None,
    grid_path=None,
    cuts_outdir=None,
    tile_writer="zarr",
    pre_write_hook=None,
    report_outdir=None,
    product_name=None,
    preprocessing_root_name=None,
    keep_intermediate=True,
    skip=False,
):
    gpt_kwargs = dict(gpt_memory=gpt_memory, gpt_parallelism=gpt_parallelism, gpt_timeout=gpt_timeout, gpt_cache_size=gpt_cache_size)
    if not config.prepro or skip:
        if skip:
            intermediate = _find_existing_intermediates(output_dir, product_mode, sentinel_swath=sentinel_swath)
            _run_existing_intermediate_tiling(
                product_path,
                intermediate,
                product_mode,
                product_wkt,
                grid_path,
                cuts_outdir,
                gpt_kwargs,
                tile_writer=tile_writer,
                pre_write_hook=pre_write_hook,
                report_outdir=report_outdir,
                product_name=product_name,
            )
            return intermediate
        return product_path
    result = ROUTER[product_mode](
        product_path,
        output_dir,
        orbit_type=orbit_type,
        orbit_continue_on_fail=orbit_continue_on_fail,
        sentinel_swath=sentinel_swath,
        sentinel_first_burst=sentinel_first_burst,
        sentinel_last_burst=sentinel_last_burst,
        sentinel_tc_source_band=sentinel_tc_source_band,
        sentinel_subap_decompositions=sentinel_subap_decompositions,
        gpt_memory=gpt_memory,
        gpt_parallelism=gpt_parallelism,
        gpt_timeout=gpt_timeout,
        gpt_cache_size=gpt_cache_size,
        product_wkt=product_wkt,
        grid_path=grid_path,
        cuts_outdir=cuts_outdir,
        product_mode=product_mode,
        tile_writer=tile_writer,
        pre_write_hook=pre_write_hook,
        report_outdir=report_outdir,
        product_name=product_name,
        preprocessing_root_name=preprocessing_root_name,
        keep_intermediate=keep_intermediate,
    )
    if isinstance(result, dict):
        for swath, path in result.items():
            print(f"Intermediate {swath}: {path}")
            if path is None:
                raise RuntimeError(f"Intermediate product for {swath} was not returned.")
            if not Path(path).exists() and not (
                not keep_intermediate and _is_isolated_preprocessing_path(path, output_dir)
            ):
                raise FileNotFoundError(f"Intermediate product {path} ({swath}) does not exist.")
        return {swath: Path(path) for swath, path in result.items()}
    print(f"Intermediate processed product located at: {result}")
    if result is None:
        raise RuntimeError(f"No intermediate product was returned for mode {product_mode}.")
    if not Path(result).exists() and not (
        not keep_intermediate and _is_isolated_preprocessing_path(result, output_dir)
    ):
        raise FileNotFoundError(f"Intermediate product {result} does not exist.")
    return Path(result)


def _run_existing_intermediate_tiling(
    product_path,
    intermediate,
    product_mode,
    product_wkt,
    grid_path,
    cuts_outdir,
    gpt_kwargs,
    tile_writer="zarr",
    pre_write_hook=None,
    report_outdir=None,
    product_name=None,
):
    if product_wkt is None or grid_path is None or cuts_outdir is None:
        return
    from sarpyx.snapflow.tiling_runtime import finalize_tops_tiling

    metadata = {
        "product_wkt": product_wkt,
        "grid_path": grid_path,
        "cuts_outdir": cuts_outdir,
        "product_mode": product_mode,
        "gpt_kwargs": dict(gpt_kwargs),
        "tile_writer": tile_writer,
        "pre_write_hook": pre_write_hook,
    }
    if report_outdir is not None:
        metadata["report_outdir"] = report_outdir
    if product_name is not None:
        metadata["product_name"] = product_name
    if isinstance(intermediate, dict):
        results = []
        for swath, path in intermediate.items():
            ctx = PipelineContext(product_path, path, Path(path).parent, None, {"tc": path}, {**metadata, "swath": swath}, dict(gpt_kwargs))
            results.append(run_step(ctx, PipelineStep("WorldSARTiling", {"intermediate_ref": "tc", "collect": True}, "tiling")))
        finalize_tops_tiling(product_wkt, grid_path, cuts_outdir, intermediate, results, report_outdir=report_outdir, product_name=product_name)
        return
    ctx = PipelineContext(product_path, intermediate, Path(intermediate).parent, None, {"tc": intermediate}, metadata, dict(gpt_kwargs))
    run_step(ctx, PipelineStep("WorldSARTiling", {"intermediate_ref": "tc"}, "tiling"))


def _run_h5_to_zarr_only(product_path, output_path, chunk_size, overwrite):
    converted = convert_tile_h5_to_zarr(input_path=product_path, output_path=output_path, chunk_size=tuple(chunk_size), overwrite=overwrite)
    summary = {"input": str(Path(product_path).expanduser().absolute()), "output": str(converted), "chunk_size": list(chunk_size), "zarr_format": 3}
    print(json.dumps(summary, indent=2))
    return converted


def _default_output_dir(product_path: Path) -> Path:
    return product_path.parent / "output"


def _cleanup_final_intermediates(intermediate, output_dir: Path) -> None:
    products = list(intermediate.values()) if isinstance(intermediate, dict) else [intermediate]
    for path in products:
        _delete_dimap_product(path, output_dir)
    _delete_generated_tiling_intermediates(output_dir, products)
    _delete_nonfinal_report_artifacts(output_dir)


def _cleanup_generated_roots(products, output_dir: Path) -> list[Path]:
    output_root = Path(output_dir).resolve()
    roots: set[Path] = set()
    for product in products:
        try:
            resolved = Path(product).resolve()
        except TypeError:
            continue
        try:
            relative = resolved.relative_to(output_root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        candidate = output_root / relative.parts[0]
        if candidate.name.startswith(ISOLATED_PREPROCESSING_PREFIX):
            roots.add(candidate)
        else:
            roots.add(resolved.parent)
    return sorted(roots)


def _delete_generated_tiling_intermediates(output_dir: Path, products) -> None:
    output_root = Path(output_dir).resolve()
    for root in _cleanup_generated_roots(products, output_dir):
        if output_root != root and output_root not in root.parents:
            continue
        for folder_name in ("worldsar_tc_epsg", "worldsar_reprojected", "worldsar_subap_tc"):
            for folder in root.rglob(folder_name):
                resolved = folder.resolve()
                if root == resolved or root not in resolved.parents:
                    continue
                if folder.is_dir():
                    import shutil

                    shutil.rmtree(folder)


def _delete_nonfinal_report_artifacts(output_dir: Path) -> None:
    pdf_root = Path(output_dir) / "pdfs"
    if not pdf_root.is_dir():
        return
    for path in pdf_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
            path.unlink()
    for folder in sorted((path for path in pdf_root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        try:
            folder.rmdir()
        except OSError:
            pass


def _delete_dimap_product(product_path, output_dir: Path) -> None:
    product = Path(product_path)
    if product.suffix.lower() != ".dim":
        return
    output_root = Path(output_dir).resolve()
    resolved = product.resolve()
    if output_root != resolved.parent and output_root not in resolved.parents:
        return
    data_dir = resolved.with_suffix(".data")
    if data_dir.exists() and data_dir.is_dir():
        import shutil

        shutil.rmtree(data_dir)
    resolved.unlink(missing_ok=True)


def run(args) -> int:
    config.validate_runtime_args(args)
    product_path = config._ensure_existing_path(args.product_path, "Input product")
    if args.h5_to_zarr_only:
        _run_h5_to_zarr_only(product_path=product_path, output_path=args.output_dir, chunk_size=args.zarr_chunk_size, overwrite=args.overwrite_zarr)
        return 0
    config.apply_runtime_overrides(args)
    output_dir = config._expand_path(args.output_dir) if args.output_dir else _default_output_dir(product_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cuts_outdir_value = args.cuts_outdir or config.CUTS_OUTDIR
    if not cuts_outdir_value:
        cuts_outdir = output_dir / "tiles"
    else:
        cuts_outdir = config._expand_path(cuts_outdir_value)
    cuts_outdir.mkdir(parents=True, exist_ok=True)
    report_outdir = output_dir / "pdfs"
    report_outdir.mkdir(parents=True, exist_ok=True)
    if config.db_indexing:
        db_dir = config._expand_path(config.DB_DIR) if config.DB_DIR else output_dir / "db"
        config.DB_DIR = str(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
    with worldsar_product_lock(product_path, output_dir, timeout=args.lock_timeout) as lock_path:
        print(f"WorldSAR product lock: {lock_path}")
        base_path = config._expand_path(config.BASE_PATH)
        grid_path = config._expand_path(config.GRID_PATH) if config.GRID_PATH else base_path / "grid" / "grid_10km.geojson"
        grid_path = config.ensure_grid_file(grid_path, base_path)
        product_mode = infer_product_mode(product_path)
        print(f"Inferred product mode: {product_mode}")
        tile_writer = normalize_tile_writer(args.tile_writer)
        product_name = product_output_name(product_path)
        pre_write_hook = None
        if tile_writer == "zarr":
            pre_write_hook = make_worldsar_zarr_tile_hook(
                product_path,
                product_mode=product_mode,
                product_name=product_name,
                chunk_size=tuple(args.zarr_chunk_size),
                subap_features=SubapFeatureConfig(
                    enabled=product_mode.startswith("S1"),
                    window_size=args.sentinel_subap_feature_window_size,
                ),
            )
        product_wkt = resolve_product_wkt(args, product_path, product_mode)
        gpt_kwargs = dict(gpt_memory=args.gpt_memory, gpt_parallelism=args.gpt_parallelism, gpt_timeout=args.gpt_timeout, gpt_cache_size=args.gpt_cache_size)
        intermediate = _run_preprocessing(
            product_path,
            output_dir,
            product_mode,
            orbit_type=args.orbit_type,
            orbit_continue_on_fail=args.orbit_continue_on_fail,
            sentinel_swath=args.sentinel_swath,
            sentinel_first_burst=args.sentinel_first_burst or s1_tops.DEFAULT_FIRST_BURST,
            sentinel_last_burst=args.sentinel_last_burst or s1_tops.DEFAULT_LAST_BURST,
            sentinel_tc_source_band=args.sentinel_tc_source_band,
            sentinel_subap_decompositions=args.sentinel_subap_decompositions,
            product_wkt=product_wkt,
            grid_path=grid_path,
            cuts_outdir=cuts_outdir,
            tile_writer=tile_writer,
            pre_write_hook=pre_write_hook,
            report_outdir=report_outdir,
            product_name=product_name,
            preprocessing_root_name=args.preprocessing_root_name,
            keep_intermediate=args.keep_intermediate,
            skip=args.skip_preprocessing,
            **gpt_kwargs,
        )
        if not args.keep_intermediate and not args.skip_preprocessing:
            _cleanup_final_intermediates(intermediate, output_dir)
    return 0
