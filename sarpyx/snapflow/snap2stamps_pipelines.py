"""snap2stamps-equivalent pipelines implemented with :mod:`sarpyx.snapflow` operators.

This module captures the operator chains used by the ``snap2stamps`` graph
collection (v2.0.1) and exposes them as reusable Python pipelines powered by
:class:`sarpyx.snapflow.engine.GPT` methods.

Design goals
------------
- Keep each graph-like workflow as a named, composable pipeline.
- Allow parameter overrides per operator.
- Provide higher-level StaMPS-ready end-to-end pipelines (PSI/SBAS style).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from sarpyx.snapflow.engine import GPT


@dataclass(frozen=True)
class PipelineStep:
    """Single operator call in a SNAP pipeline."""

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)


# Graph-level pipelines (snap2stamps style) mapped to GPT methods.
SNAP2STAMPS_GRAPH_PIPELINES: dict[str, tuple[PipelineStep, ...]] = {
    "split_orbit": (
        PipelineStep("topsar_split", {"subswath": "IW3", "selected_polarisations": ["VV"]}),
        PipelineStep("apply_orbit_file"),
    ),
    "deburst": (
        PipelineStep("deburst"),
    ),
    "coregistration": (
        PipelineStep("back_geocoding"),
        PipelineStep("enhanced_spectral_diversity"),
    ),
    "interferogram": (
        PipelineStep("interferogram", {"subtract_flat_earth_phase": True}),
    ),
    "topo_phase_removal": (
        PipelineStep("topo_phase_removal"),
    ),
    "goldstein_filtering": (
        PipelineStep("goldstein_phase_filtering"),
    ),
    "multilook": (
        PipelineStep("multilook", {"n_rg_looks": 4, "n_az_looks": 1}),
    ),
    "subset": (
        PipelineStep("subset"),
    ),
    "snaphu_export": (
        PipelineStep("snaphu_export"),
    ),
    "snaphu_import": (
        PipelineStep("snaphu_import"),
    ),
    "phase_to_displacement": (
        PipelineStep("phase_to_displacement"),
    ),
    "terrain_correction": (
        PipelineStep("terrain_correction"),
    ),
    "stamps_export": (PipelineStep("stamps_export", {"psi_format": True}),),
}


# End-to-end pipelines composed from graph-level blocks.
SNAP2STAMPS_WORKFLOWS: dict[str, tuple[str, ...]] = {
    # "psi_prep": (
    #     "split_orbit",
    #     "deburst",
    #     "coregistration",
    #     "interferogram",
    #     "topo_phase_removal",
    #     "goldstein_filtering",
    #     "multilook",
    #     "subset",
    #     "snaphu_export",
    # ),
    # "psi_post_unwrap": (
    #     "snaphu_import",
    #     "phase_to_displacement",
    #     "terrain_correction",
    #     "stamps_export",
    # ),
    # "psi_full": (
    #     "split_orbit",
    #     "deburst",
    #     "coregistration",
    #     "interferogram",
    #     "topo_phase_removal",
    #     "goldstein_filtering",
    #     "multilook",
    #     "subset",
    #     "snaphu_export",
    #     "snaphu_import",
    #     "phase_to_displacement",
    #     "terrain_correction",
    #     "stamps_export",
    # ),
    # "sbas_full": (
    #     "split_orbit",
    #     "deburst",
    #     "coregistration",
    #     "interferogram",
    #     "topo_phase_removal",
    #     "goldstein_filtering",
    #     "multilook",
    #     "subset",
    #     "snaphu_export",
    #     "snaphu_import",
    #     "phase_to_displacement",
    #     "terrain_correction",
    # ),
    "test_correct": (
        "split_orbit",
        "deburst",
        "coregistration",
        "interferogram",
        "topo_phase_removal",
        "subset",
        "terrain_correction",
    ),
}


def _clone_overrides(overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    if not overrides:
        return {}
    return {step_name: dict(step_kwargs) for step_name, step_kwargs in overrides.items()}


def _overrides_with_subset_aoi(
    overrides: dict[str, dict[str, Any]] | None = None,
    aoi_wkt: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    if not aoi_wkt:
        return overrides

    merged = _clone_overrides(overrides)
    subset_overrides = dict(merged.get("subset", {}))
    subset_overrides.setdefault("geo_region", aoi_wkt)
    merged["subset"] = subset_overrides
    return merged


def _run_back_geocoding_with_source_products(
    gpt: GPT,
    source_products: Sequence[str | Path],
    kwargs: dict[str, Any],
) -> str:
    if len(source_products) < 2:
        raise ValueError(
            "back_geocoding source_products override requires at least two products "
            "(master and one secondary)."
        )

    dem_name = kwargs.pop("dem_name", "SRTM 3Sec")
    dem_resampling_method = kwargs.pop("dem_resampling_method", "BICUBIC_INTERPOLATION")
    external_dem_file = kwargs.pop("external_dem_file", None)
    external_dem_no_data_value = kwargs.pop("external_dem_no_data_value", 0.0)
    resampling_type = kwargs.pop("resampling_type", "BISINC_5_POINT_INTERPOLATION")
    mask_out_area_without_elevation = kwargs.pop("mask_out_area_without_elevation", True)
    output_range_azimuth_offset = kwargs.pop("output_range_azimuth_offset", False)
    output_deramp_demod_phase = kwargs.pop("output_deramp_demod_phase", False)
    disable_reramp = kwargs.pop("disable_reramp", False)
    output_name = kwargs.pop("output_name", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(
            "Unsupported parameter(s) for multi-source back_geocoding override: "
            f"{unknown}"
        )

    gpt._reset_command()
    products = ",".join(Path(prod).as_posix() for prod in source_products)
    for index, token in enumerate(gpt.current_cmd):
        if token.startswith("-Ssource="):
            gpt.current_cmd[index] = f"-SsourceProducts={products}"
            break

    cmd_params = [
        f'-PdemName="{dem_name}"',
        f"-PdemResamplingMethod={dem_resampling_method}",
        f"-PexternalDEMNoDataValue={external_dem_no_data_value}",
        f"-PresamplingType={resampling_type}",
        f"-PmaskOutAreaWithoutElevation={str(mask_out_area_without_elevation).lower()}",
        f"-PoutputRangeAzimuthOffset={str(output_range_azimuth_offset).lower()}",
        f"-PoutputDerampDemodPhase={str(output_deramp_demod_phase).lower()}",
        f"-PdisableReramp={str(disable_reramp).lower()}",
    ]
    if external_dem_file:
        cmd_params.append(f"-PexternalDEMFile={Path(external_dem_file).as_posix()}")

    gpt.current_cmd.append(f'Back-Geocoding {" ".join(cmd_params)}')
    result = gpt._call(suffix="BGEO", output_name=output_name)
    if result is None:
        raise RuntimeError(f"Operator 'back_geocoding' failed: {gpt.last_error_summary()}")
    return result


def _run_step(gpt: GPT, step: PipelineStep, overrides: dict[str, dict[str, Any]] | None = None) -> str:
    method: Callable[..., str | None] | None = getattr(gpt, step.method, None)
    if method is None:
        raise AttributeError(f"GPT does not expose method '{step.method}'")

    kwargs = dict(step.kwargs)
    if overrides and step.method in overrides:
        kwargs.update(overrides[step.method])

    if step.method == "back_geocoding" and "source_products" in kwargs:
        source_products = kwargs.pop("source_products")
        if isinstance(source_products, (str, Path)):
            raise TypeError("source_products must be a sequence of product paths, not a single path.")
        return _run_back_geocoding_with_source_products(gpt=gpt, source_products=source_products, kwargs=kwargs)

    if step.method == "stamps_export" and "target_folder" not in kwargs:
        kwargs["target_folder"] = (gpt.outdir / "stamps_export").as_posix()
    if step.method == "snaphu_export":
        kwargs.setdefault("snaphu_processing_location", gpt.outdir.as_posix())

    result = method(**kwargs)
    if result is None:
        raise RuntimeError(f"Operator '{step.method}' failed: {gpt.last_error_summary()}")
    return result


def _run_graph_sequence(
    gpt: GPT,
    graph_names: Sequence[str],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    output = gpt.prod_path.as_posix()
    for graph_name in graph_names:
        output = run_graph_pipeline(gpt=gpt, graph_name=graph_name, overrides=overrides)
    return output


def run_graph_pipeline(
    gpt: GPT,
    graph_name: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a single snap2stamps graph-equivalent pipeline."""
    try:
        steps = SNAP2STAMPS_GRAPH_PIPELINES[graph_name]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_GRAPH_PIPELINES))
        raise KeyError(f"Unknown graph pipeline '{graph_name}'. Available: {available}") from exc

    output = gpt.prod_path.as_posix()
    for step in steps:
        output = _run_step(gpt, step, overrides)
    return output


def run_workflow(
    gpt: GPT | Sequence[GPT],
    workflow: str,
    overrides: dict[str, dict[str, Any]] | None = None,
    *,
    aoi_wkt: str | None = None,
) -> str:
    """Run a composed snap2stamps workflow (PSI/SBAS variants)."""
    try:
        graph_names = SNAP2STAMPS_WORKFLOWS[workflow]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_WORKFLOWS))
        raise KeyError(f"Unknown workflow '{workflow}'. Available: {available}") from exc

    workflow_overrides = _overrides_with_subset_aoi(overrides=overrides, aoi_wkt=aoi_wkt)

    if workflow == "test_correct":
        if not isinstance(gpt, Sequence):
            raise ValueError(
                "Workflow 'test_correct' requires exactly two GPT objects (master and secondary)."
            )
        if len(gpt) != 2:
            raise ValueError("Workflow 'test_correct' requires exactly two GPT objects.")

        master_gpt, secondary_gpt = gpt
        master_preprocessed = _run_graph_sequence(
            gpt=master_gpt,
            graph_names=("split_orbit", "deburst"),
            overrides=workflow_overrides,
        )
        secondary_preprocessed = _run_graph_sequence(
            gpt=secondary_gpt,
            graph_names=("split_orbit", "deburst"),
            overrides=workflow_overrides,
        )

        coreg_overrides = _clone_overrides(workflow_overrides)
        coreg_step_overrides = dict(coreg_overrides.get("back_geocoding", {}))
        coreg_step_overrides["source_products"] = [master_preprocessed, secondary_preprocessed]
        coreg_overrides["back_geocoding"] = coreg_step_overrides

        return _run_graph_sequence(
            gpt=master_gpt,
            graph_names=("coregistration", "interferogram", "topo_phase_removal", "subset", "terrain_correction"),
            overrides=coreg_overrides,
        )

    if isinstance(gpt, Sequence):
        raise TypeError(
            f"Workflow '{workflow}' expects a single GPT object. "
            "Only 'test_correct' accepts a sequence of GPT objects."
        )
    return _run_graph_sequence(gpt=gpt, graph_names=graph_names, overrides=workflow_overrides)


def build_gpt(
    product: str | Path,
    outdir: str | Path,
    *,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
) -> GPT:
    """Convenience constructor aligned with snap2stamps-like batch processing."""
    return GPT(
        product=product,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
    )
