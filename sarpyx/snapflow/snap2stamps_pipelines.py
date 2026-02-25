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
from typing import Any, Callable

from sarpyx.snapflow.engine import GPT


@dataclass(frozen=True)
class PipelineStep:
    """Single operator call in a SNAP pipeline."""

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)


# Graph-level pipelines (snap2stamps style) mapped to GPT methods.
SNAP2STAMPS_GRAPH_PIPELINES: dict[str, tuple[PipelineStep, ...]] = {
    "split_orbit": (
        PipelineStep("topsar_split"),
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
    "psi_prep": (
        "split_orbit",
        "deburst",
        "coregistration",
        "interferogram",
        "topo_phase_removal",
        "goldstein_filtering",
        "multilook",
        "subset",
        "snaphu_export",
    ),
    "psi_post_unwrap": (
        "snaphu_import",
        "phase_to_displacement",
        "terrain_correction",
        "stamps_export",
    ),
    "psi_full": (
        "split_orbit",
        "deburst",
        "coregistration",
        "interferogram",
        "topo_phase_removal",
        "goldstein_filtering",
        "multilook",
        "subset",
        "snaphu_export",
        "snaphu_import",
        "phase_to_displacement",
        "terrain_correction",
        "stamps_export",
    ),
    "sbas_full": (
        "split_orbit",
        "deburst",
        "coregistration",
        "interferogram",
        "topo_phase_removal",
        "goldstein_filtering",
        "multilook",
        "subset",
        "snaphu_export",
        "snaphu_import",
        "phase_to_displacement",
        "terrain_correction",
    ),
}


def _run_step(gpt: GPT, step: PipelineStep, overrides: dict[str, dict[str, Any]] | None = None) -> str:
    method: Callable[..., str | None] | None = getattr(gpt, step.method, None)
    if method is None:
        raise AttributeError(f"GPT does not expose method '{step.method}'")

    kwargs = dict(step.kwargs)
    if overrides and step.method in overrides:
        kwargs.update(overrides[step.method])

    if step.method == "stamps_export" and "target_folder" not in kwargs:
        kwargs["target_folder"] = (gpt.outdir / "stamps_export").as_posix()
    if step.method == "snaphu_export":
        kwargs.setdefault("snaphu_processing_location", gpt.outdir.as_posix())

    result = method(**kwargs)
    if result is None:
        raise RuntimeError(f"Operator '{step.method}' failed: {gpt.last_error_summary()}")
    return result


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
    gpt: GPT,
    workflow: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a composed snap2stamps workflow (PSI/SBAS variants)."""
    try:
        graph_names = SNAP2STAMPS_WORKFLOWS[workflow]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_WORKFLOWS))
        raise KeyError(f"Unknown workflow '{workflow}'. Available: {available}") from exc

    output = gpt.prod_path.as_posix()
    for graph_name in graph_names:
        output = run_graph_pipeline(gpt=gpt, graph_name=graph_name, overrides=overrides)
    return output


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
