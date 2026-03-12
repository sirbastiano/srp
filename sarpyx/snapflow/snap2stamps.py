"""Processing-only SNAP2StaMPS workflows implemented on top of :mod:`snapflow`.

This module covers the upstream SNAP2StaMPS processing branches without
recreating the original script/config runner UX. The focus is:

- processing branch selection for TOPSAR and Stripmap data
- reusable Python-callable helpers for the graph families used upstream
- backward-compatible access to the narrower workflow helpers that previously
  lived in ``snap2stamps_pipelines.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from sarpyx.snapflow.engine import GPT

BranchName = Literal["topsar", "stripmap", "generic"]
InputKind = Literal["single", "pair", "multi"]
WorkflowInputKind = Literal["single", "pair"]


@dataclass(frozen=True)
class PipelineStep:
    """Single operator call in a SNAP pipeline."""

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairProducts:
    """Master/slave product pair used for pairwise InSAR preparation."""

    master: str | Path
    slave: str | Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "master", Path(self.master))
        object.__setattr__(self, "slave", Path(self.slave))


@dataclass(frozen=True)
class PipelineDefinition:
    """Description of an upstream-equivalent processing pipeline variant."""

    name: str
    branch: BranchName
    input_kind: InputKind
    stages: tuple[str, ...]
    upstream_graph: str | None = None
    description: str = ""


@dataclass(frozen=True)
class TopsarCoregIfgResult:
    """Outputs of the TOPSAR coregistration/interferogram branch."""

    coreg_path: str
    ifg_path: str
    pipeline_name: str


def _as_path_list(products: Iterable[str | Path]) -> list[Path]:
    paths = [Path(product) for product in products]
    if not paths:
        raise ValueError("At least one product path is required")
    return paths


def _external_dem_name(external_dem_file: str | Path | None, default: str) -> str:
    return "External DEM" if external_dem_file else default


def _step_kwargs(
    method_name: str,
    default_kwargs: dict[str, Any],
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    kwargs = dict(default_kwargs)
    if overrides and method_name in overrides:
        kwargs.update(overrides[method_name])
    return kwargs


SNAP2STAMPS_GRAPH_PIPELINES: dict[str, tuple[PipelineStep, ...]] = {
    "split_orbit": (
        PipelineStep("topsar_split"),
        PipelineStep("apply_orbit_file"),
    ),
    "deburst": (
        PipelineStep("deburst"),
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
    "stamps_export": (
        PipelineStep("stamps_export", {"psi_format": True}),
    ),
}


PAIRWISE_GRAPH_NAMES = frozenset({"coregistration"})


SNAP2STAMPS_PIPELINES: dict[str, PipelineDefinition] = {
    "topsar_split_applyorbit": PipelineDefinition(
        name="topsar_split_applyorbit",
        branch="topsar",
        input_kind="single",
        stages=("topsar_split", "apply_orbit_file"),
        upstream_graph="topsar_master_split_applyorbit.xml / topsar_secondaries_split_applyorbit.xml",
        description="Split and orbit-correct a single-slice TOPSAR product.",
    ),
    "topsar_assemble_split_applyorbit": PipelineDefinition(
        name="topsar_assemble_split_applyorbit",
        branch="topsar",
        input_kind="multi",
        stages=("slice_assembly", "topsar_split", "apply_orbit_file"),
        upstream_graph="topsar_master_assemble_split_applyorbit.xml / topsar_secondaries_assemble_split_applyorbit.xml",
        description="Assemble multi-slice TOPSAR data before split/apply-orbit.",
    ),
    "topsar_coreg_ifg": PipelineDefinition(
        name="topsar_coreg_ifg",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "enhanced_spectral_diversity", "deburst", "interferogram", "deburst"),
        upstream_graph="topsar_coreg_ifg_computation.xml",
        description="TOPSAR pair processing with ESD and no subset/topo stage.",
    ),
    "topsar_coreg_ifg_ext_dem": PipelineDefinition(
        name="topsar_coreg_ifg_ext_dem",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "enhanced_spectral_diversity", "deburst", "interferogram", "deburst"),
        upstream_graph="topsar_coreg_ifg_computation_extDEM.xml",
        description="TOPSAR pair processing with ESD and external DEM.",
    ),
    "topsar_coreg_ifg_subset": PipelineDefinition(
        name="topsar_coreg_ifg_subset",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "enhanced_spectral_diversity", "deburst", "interferogram", "deburst", "topo_phase_removal", "subset"),
        upstream_graph="topsar_coreg_ifg_computation_subset.xml",
        description="TOPSAR pair processing with subset/topo branch enabled.",
    ),
    "topsar_coreg_ifg_subset_ext_dem": PipelineDefinition(
        name="topsar_coreg_ifg_subset_ext_dem",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "enhanced_spectral_diversity", "deburst", "interferogram", "deburst", "topo_phase_removal", "subset"),
        upstream_graph="topsar_coreg_ifg_computation_subset_extDEM.xml",
        description="TOPSAR subset branch with external DEM.",
    ),
    "topsar_coreg_ifg_no_esd": PipelineDefinition(
        name="topsar_coreg_ifg_no_esd",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "deburst", "interferogram", "deburst"),
        upstream_graph="topsar_coreg_ifg_computation_noESD.xml",
        description="TOPSAR pair processing without ESD for single-burst cases.",
    ),
    "topsar_coreg_ifg_no_esd_ext_dem": PipelineDefinition(
        name="topsar_coreg_ifg_no_esd_ext_dem",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "deburst", "interferogram", "deburst"),
        upstream_graph="topsar_coreg_ifg_computation_noESD_extDEM.xml",
        description="TOPSAR no-ESD branch with external DEM.",
    ),
    "topsar_coreg_ifg_subset_no_esd": PipelineDefinition(
        name="topsar_coreg_ifg_subset_no_esd",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "deburst", "interferogram", "deburst", "topo_phase_removal", "subset"),
        upstream_graph="topsar_coreg_ifg_computation_subset_noESD.xml",
        description="TOPSAR single-burst branch with subset/topo stage.",
    ),
    "topsar_coreg_ifg_subset_no_esd_ext_dem": PipelineDefinition(
        name="topsar_coreg_ifg_subset_no_esd_ext_dem",
        branch="topsar",
        input_kind="pair",
        stages=("back_geocoding", "deburst", "interferogram", "deburst", "topo_phase_removal", "subset"),
        upstream_graph="topsar_coreg_ifg_computation_subset_noESD_extDEM.xml",
        description="TOPSAR single-burst subset branch with external DEM.",
    ),
    "topsar_export": PipelineDefinition(
        name="topsar_export",
        branch="topsar",
        input_kind="pair",
        stages=("stamps_export_pair",),
        upstream_graph="topsar_export.xml",
        description="Direct StaMPS export for a single-IW TOPSAR pair.",
    ),
    "topsar_export_mergeiw_subset": PipelineDefinition(
        name="topsar_export_mergeiw_subset",
        branch="topsar",
        input_kind="multi",
        stages=("topsar_merge_products", "topo_phase_removal", "subset", "stamps_export_pair"),
        upstream_graph="topsar_export_mergeIW_subset.xml",
        description="Merge multi-IW TOPSAR products before StaMPS export.",
    ),
    "topsar_export_mergeiw_subset_ext_dem": PipelineDefinition(
        name="topsar_export_mergeiw_subset_ext_dem",
        branch="topsar",
        input_kind="multi",
        stages=("topsar_merge_products", "topo_phase_removal", "subset", "stamps_export_pair"),
        upstream_graph="topsar_export_mergeIW_subset_extDEM.xml",
        description="Merge multi-IW TOPSAR products before StaMPS export using an external DEM.",
    ),
    "stripmap_subset": PipelineDefinition(
        name="stripmap_subset",
        branch="stripmap",
        input_kind="single",
        stages=("subset",),
        upstream_graph="stripmap_TSX_Subset.xml",
        description="Subset Stripmap products prior to coregistration.",
    ),
    "stripmap_dem_assisted_coregistration": PipelineDefinition(
        name="stripmap_dem_assisted_coregistration",
        branch="stripmap",
        input_kind="pair",
        stages=("dem_assisted_coregistration_pair",),
        upstream_graph="stripmap_DEM_Assisted_Coregistration.xml",
        description="DEM-assisted Stripmap coregistration.",
    ),
    "stripmap_dem_assisted_coregistration_ext_dem": PipelineDefinition(
        name="stripmap_dem_assisted_coregistration_ext_dem",
        branch="stripmap",
        input_kind="pair",
        stages=("dem_assisted_coregistration_pair",),
        upstream_graph="stripmap_DEM_Assisted_Coregistration_extDEM.xml",
        description="DEM-assisted Stripmap coregistration using an external DEM.",
    ),
    "stripmap_interferogram_topophase": PipelineDefinition(
        name="stripmap_interferogram_topophase",
        branch="stripmap",
        input_kind="single",
        stages=("interferogram", "topo_phase_removal"),
        upstream_graph="stripmap_Interferogram_TopoPhase.xml",
        description="Stripmap interferogram generation followed by topographic phase removal.",
    ),
    "stripmap_interferogram_topophase_ext_dem": PipelineDefinition(
        name="stripmap_interferogram_topophase_ext_dem",
        branch="stripmap",
        input_kind="single",
        stages=("interferogram", "topo_phase_removal"),
        upstream_graph="stripmap_Interferogram_TopoPhase_extDEM.xml",
        description="Stripmap interferogram generation with external DEM topo removal.",
    ),
    "stripmap_export": PipelineDefinition(
        name="stripmap_export",
        branch="stripmap",
        input_kind="pair",
        stages=("stamps_export_pair",),
        upstream_graph="stripmap_Export.xml",
        description="StaMPS export for a Stripmap coreg/IFG pair.",
    ),
}


SNAP2STAMPS_WORKFLOWS: dict[str, tuple[str, ...]] = {
    "stamps_prep": (
        "split_orbit",
        "coregistration",
        "deburst",
        "interferogram",
        "topo_phase_removal",
        "subset",
        "terrain_correction",
    ),
    "psi_prep": (
        "split_orbit",
        "coregistration",
        "deburst",
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
        "coregistration",
        "deburst",
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
        "coregistration",
        "deburst",
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

SNAP2STAMPS_WORKFLOW_INPUTS: dict[str, WorkflowInputKind] = {
    "stamps_prep": "pair",
    "psi_prep": "pair",
    "psi_post_unwrap": "single",
    "psi_full": "pair",
    "sbas_full": "pair",
}


def list_pipeline_names(branch: BranchName | None = None) -> tuple[str, ...]:
    """Return known processing pipeline names, optionally filtered by branch."""
    names = SNAP2STAMPS_PIPELINES.keys()
    if branch is None:
        return tuple(sorted(names))
    return tuple(
        sorted(name for name, definition in SNAP2STAMPS_PIPELINES.items() if definition.branch == branch)
    )


def get_pipeline_definition(name: str) -> PipelineDefinition:
    """Return a pipeline definition by name."""
    try:
        return SNAP2STAMPS_PIPELINES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_PIPELINES))
        raise KeyError(f"Unknown pipeline '{name}'. Available: {available}") from exc


def pipeline_requires_pair(name: str) -> bool:
    """Return ``True`` when a processing branch requires explicit master/slave inputs."""
    return get_pipeline_definition(name).input_kind == "pair"


def pipeline_requires_multi_input(name: str) -> bool:
    """Return ``True`` when a processing branch requires more than one source product."""
    return get_pipeline_definition(name).input_kind == "multi"


def select_topsar_split_pipeline(source_count: int) -> str:
    """Select the upstream-equivalent TOPSAR split/apply-orbit branch."""
    if source_count < 1:
        raise ValueError("source_count must be >= 1")
    return "topsar_assemble_split_applyorbit" if source_count > 1 else "topsar_split_applyorbit"


def select_topsar_coreg_ifg_pipeline(
    *,
    master_count: int,
    burst_count: int,
    external_dem_file: str | Path | None = None,
) -> str:
    """Select the upstream-equivalent TOPSAR coreg/IFG pipeline variant."""
    if master_count < 1:
        raise ValueError("master_count must be >= 1")
    if burst_count < 1:
        raise ValueError("burst_count must be >= 1")

    pipeline = "topsar_coreg_ifg"
    if master_count == 1:
        pipeline += "_subset"
    if burst_count == 1:
        pipeline += "_no_esd"
    if external_dem_file:
        pipeline += "_ext_dem"
    return pipeline


def select_topsar_export_pipeline(
    *,
    master_count: int,
    external_dem_file: str | Path | None = None,
) -> str:
    """Select the upstream-equivalent TOPSAR StaMPS export branch."""
    if master_count < 1:
        raise ValueError("master_count must be >= 1")
    if master_count == 1:
        return "topsar_export"
    return (
        "topsar_export_mergeiw_subset_ext_dem"
        if external_dem_file else
        "topsar_export_mergeiw_subset"
    )


def select_stripmap_coreg_pipeline(external_dem_file: str | Path | None = None) -> str:
    """Select the upstream-equivalent Stripmap coregistration branch."""
    return (
        "stripmap_dem_assisted_coregistration_ext_dem"
        if external_dem_file else
        "stripmap_dem_assisted_coregistration"
    )


def select_stripmap_ifg_pipeline(external_dem_file: str | Path | None = None) -> str:
    """Select the upstream-equivalent Stripmap IFG/topo branch."""
    return (
        "stripmap_interferogram_topophase_ext_dem"
        if external_dem_file else
        "stripmap_interferogram_topophase"
    )


def _run_step(
    gpt: GPT,
    step: PipelineStep,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    method: Callable[..., str | None] | None = getattr(gpt, step.method, None)
    if method is None:
        raise AttributeError(f"GPT does not expose method '{step.method}'")

    kwargs = _step_kwargs(step.method, step.kwargs, overrides)
    if step.method == "stamps_export" and "target_folder" not in kwargs:
        kwargs["target_folder"] = (gpt.outdir / "stamps_export").as_posix()
    if step.method == "snaphu_export":
        kwargs.setdefault("snaphu_processing_location", gpt.outdir.as_posix())

    result = method(**kwargs)
    if result is None:
        raise RuntimeError(f"Operator '{step.method}' failed: {gpt.last_error_summary()}")
    return result


def build_gpt(
    product: str | Path,
    outdir: str | Path,
    *,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> GPT:
    """Convenience constructor aligned with SNAP2StaMPS-like batch processing."""
    return GPT(
        product=product,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )


def run_graph_pipeline(
    gpt: GPT,
    graph_name: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a single-input stage from the legacy graph-pipeline catalog."""
    if graph_name in PAIRWISE_GRAPH_NAMES:
        raise ValueError(
            f"Graph pipeline '{graph_name}' requires a master/slave pair. "
            "Use run_pair_graph_pipeline()."
        )

    try:
        steps = SNAP2STAMPS_GRAPH_PIPELINES[graph_name]
    except KeyError as exc:
        available = ", ".join(sorted(set(SNAP2STAMPS_GRAPH_PIPELINES) | set(PAIRWISE_GRAPH_NAMES)))
        raise KeyError(f"Unknown graph pipeline '{graph_name}'. Available: {available}") from exc

    output = gpt.prod_path.as_posix()
    for step in steps:
        output = _run_step(gpt, step, overrides)
    return output


def run_pair_graph_pipeline(
    gpt: GPT,
    graph_name: str,
    pair: PairProducts,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a graph pipeline that consumes explicit master and slave products."""
    if graph_name != "coregistration":
        raise KeyError(f"Unknown pair graph pipeline '{graph_name}'. Available: coregistration")

    kwargs = overrides.get("topsar_coregistration", {}).copy() if overrides else {}
    result = gpt.topsar_coregistration(
        master_product=pair.master,
        slave_product=pair.slave,
        **kwargs,
    )
    if result is None:
        raise RuntimeError(f"Operator 'topsar_coregistration' failed: {gpt.last_error_summary()}")
    return result


def prepare_pair(
    pair: PairProducts,
    outdir: str | Path,
    *,
    preprocess_graphs: tuple[str, ...] = ("split_orbit",),
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> PairProducts:
    """Prepare master and slave independently for pairwise TOPSAR processing."""
    prepared: dict[str, Path] = {}
    for role, product in (("master", pair.master), ("slave", pair.slave)):
        gpt = build_gpt(
            product=product,
            outdir=Path(outdir) / role,
            format=format,
            gpt_path=gpt_path,
            memory=memory,
            parallelism=parallelism,
            timeout=timeout,
            snap_userdir=snap_userdir,
        )
        for graph_name in preprocess_graphs:
            run_graph_pipeline(gpt=gpt, graph_name=graph_name, overrides=overrides)
        prepared[role] = Path(gpt.prod_path)
    return PairProducts(master=prepared["master"], slave=prepared["slave"])


def run_pair_workflow(
    pair: PairProducts,
    outdir: str | Path,
    workflow: str = "stamps_prep",
    *,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a pairwise TOPSAR workflow from split/deburst through final output."""
    try:
        graph_names = SNAP2STAMPS_WORKFLOWS[workflow]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_WORKFLOWS))
        raise KeyError(f"Unknown workflow '{workflow}'. Available: {available}") from exc
    if SNAP2STAMPS_WORKFLOW_INPUTS.get(workflow) != "pair":
        raise ValueError(
            f"Workflow '{workflow}' is not pair-based. "
            "Use run_workflow() for single-input workflows."
        )

    prepared_pair = prepare_pair(
        pair=pair,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
        overrides=overrides,
    )
    gpt = build_gpt(
        product=prepared_pair.master,
        outdir=Path(outdir) / "pair",
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    output = gpt.prod_path.as_posix()
    for graph_name in graph_names:
        if graph_name == "split_orbit":
            continue
        if graph_name in PAIRWISE_GRAPH_NAMES:
            output = run_pair_graph_pipeline(
                gpt=gpt,
                graph_name=graph_name,
                pair=prepared_pair,
                overrides=overrides,
            )
            continue
        output = run_graph_pipeline(gpt=gpt, graph_name=graph_name, overrides=overrides)
    return output


def run_workflow(
    gpt: GPT,
    workflow: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a composed single-input workflow from the convenience catalog."""
    try:
        graph_names = SNAP2STAMPS_WORKFLOWS[workflow]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_WORKFLOWS))
        raise KeyError(f"Unknown workflow '{workflow}'. Available: {available}") from exc

    if SNAP2STAMPS_WORKFLOW_INPUTS.get(workflow) != "single":
        raise ValueError(
            f"Workflow '{workflow}' requires pair inputs. "
            "Use run_pair_workflow() with PairProducts."
        )

    output = gpt.prod_path.as_posix()
    for graph_name in graph_names:
        output = run_graph_pipeline(gpt=gpt, graph_name=graph_name, overrides=overrides)
    return output


def run_topsar_split_apply_orbit(
    source_products: Iterable[str | Path],
    outdir: str | Path,
    *,
    subswath: str,
    polarisation: str | None = None,
    polygon_wkt: str | None = None,
    output_name: str | None = None,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
    apply_orbit_kwargs: dict[str, Any] | None = None,
) -> str:
    """Run the upstream-equivalent TOPSAR split/apply-orbit branch."""
    products = _as_path_list(source_products)
    gpt = build_gpt(
        product=products[0],
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    selected_pols = [polarisation] if polarisation else None
    if select_topsar_split_pipeline(len(products)) == "topsar_assemble_split_applyorbit":
        assembled = gpt.slice_assembly(
            source_products=products,
            selected_polarisations=selected_pols,
            output_name=f"{output_name}_assembled" if output_name else None,
        )
        if assembled is None:
            raise RuntimeError(gpt.last_error_summary())
    split_path = gpt.topsar_split(
        subswath=subswath,
        selected_polarisations=selected_pols,
        wkt_aoi=polygon_wkt,
        output_name=f"{output_name}_split" if output_name else None,
    )
    if split_path is None:
        raise RuntimeError(gpt.last_error_summary())
    orbit_kwargs = dict(apply_orbit_kwargs or {})
    orbit_path = gpt.apply_orbit_file(
        output_name=output_name,
        **orbit_kwargs,
    )
    if orbit_path is None:
        raise RuntimeError(gpt.last_error_summary())
    return orbit_path


def run_processing_pipeline(
    pipeline_name: str,
    *,
    outdir: str | Path,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
    source_product: str | Path | None = None,
    source_products: Iterable[str | Path] | None = None,
    pair: PairProducts | None = None,
    target_folder: str | Path | None = None,
    **kwargs: Any,
) -> str | TopsarCoregIfgResult:
    """Run a named upstream-equivalent processing branch from :data:`SNAP2STAMPS_PIPELINES`."""
    definition = get_pipeline_definition(pipeline_name)

    common = {
        "outdir": outdir,
        "format": format,
        "gpt_path": gpt_path,
        "memory": memory,
        "parallelism": parallelism,
        "timeout": timeout,
        "snap_userdir": snap_userdir,
    }

    if pipeline_name in {"topsar_split_applyorbit", "topsar_assemble_split_applyorbit"}:
        products = source_products or (() if source_product is None else (source_product,))
        if not products:
            raise ValueError(f"{pipeline_name} requires source_products or source_product")
        return run_topsar_split_apply_orbit(products, **common, **kwargs)

    if pipeline_name.startswith("topsar_coreg_ifg"):
        if pair is None:
            raise ValueError(f"{pipeline_name} requires pair=PairProducts(...)")
        return run_topsar_coreg_ifg(
            pair=pair,
            pipeline_name=pipeline_name,
            **common,
            **kwargs,
        )

    if pipeline_name.startswith("topsar_export"):
        if target_folder is None:
            raise ValueError(f"{pipeline_name} requires target_folder")
        coreg_products = kwargs.pop("coreg_products", None)
        ifg_products = kwargs.pop("ifg_products", None)
        if coreg_products is None or ifg_products is None:
            raise ValueError(f"{pipeline_name} requires coreg_products and ifg_products")
        return run_topsar_export(
            coreg_products=coreg_products,
            ifg_products=ifg_products,
            target_folder=target_folder,
            **common,
            **kwargs,
        )

    if pipeline_name == "stripmap_subset":
        if source_product is None:
            raise ValueError("stripmap_subset requires source_product")
        return run_stripmap_subset(product=source_product, **common, **kwargs)

    if pipeline_name.startswith("stripmap_dem_assisted_coregistration"):
        if pair is None:
            raise ValueError(f"{pipeline_name} requires pair=PairProducts(...)")
        return run_stripmap_coreg(pair=pair, **common, **kwargs)

    if pipeline_name.startswith("stripmap_interferogram_topophase"):
        if source_product is None:
            raise ValueError(f"{pipeline_name} requires source_product")
        return run_stripmap_ifg(coreg_product=source_product, **common, **kwargs)

    if pipeline_name == "stripmap_export":
        if target_folder is None:
            raise ValueError("stripmap_export requires target_folder")
        if pair is None:
            raise ValueError("stripmap_export requires pair=PairProducts(...) with coreg/ifg paths")
        return run_stripmap_export(
            coreg_product=pair.master,
            ifg_product=pair.slave,
            target_folder=target_folder,
            **common,
        )

    raise NotImplementedError(
        f"Pipeline '{definition.name}' is registered but has no execution dispatcher"
    )


def run_topsar_coreg_ifg(
    pair: PairProducts,
    outdir: str | Path,
    *,
    master_count: int | None = None,
    burst_count: int | None = None,
    pipeline_name: str | None = None,
    polygon_wkt: str | None = None,
    subset_region: str = "0,0,0,0",
    external_dem_file: str | Path | None = None,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
    output_name_prefix: str = "pair",
) -> TopsarCoregIfgResult:
    """Run the upstream-equivalent TOPSAR coregistration/interferogram branch."""
    if pipeline_name is None:
        if master_count is None or burst_count is None:
            raise ValueError("master_count and burst_count are required when pipeline_name is not provided")
        pipeline_name = select_topsar_coreg_ifg_pipeline(
            master_count=master_count,
            burst_count=burst_count,
            external_dem_file=external_dem_file,
        )

    definition = get_pipeline_definition(pipeline_name)
    if definition.branch != "topsar" or not pipeline_name.startswith("topsar_coreg_ifg"):
        raise ValueError(f"Pipeline '{pipeline_name}' is not a TOPSAR coregistration/interferogram pipeline")

    requires_external_dem = "_ext_dem" in pipeline_name
    if requires_external_dem and external_dem_file is None:
        raise ValueError(f"Pipeline '{pipeline_name}' requires external_dem_file")
    if not requires_external_dem and external_dem_file is not None and pipeline_name.endswith("_ext_dem") is False:
        # Allow callers to pass an external DEM only when running an ext_dem pipeline.
        raise ValueError(
            f"Pipeline '{pipeline_name}' does not use external_dem_file. "
            "Select an '_ext_dem' variant or omit external_dem_file."
        )

    stages = set(definition.stages)
    subset_outputs = "subset" in stages
    use_esd = "enhanced_spectral_diversity" in stages
    outdir_path = Path(outdir)

    gpt = build_gpt(
        product=pair.master,
        outdir=outdir_path,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    coreg_path = gpt.topsar_coregistration(
        master_product=pair.master,
        slave_product=pair.slave,
        use_esd=use_esd,
        dem_name=_external_dem_name(external_dem_file, "SRTM 1Sec HGT"),
        external_dem_file=external_dem_file,
        output_name=f"{output_name_prefix}_coreg",
    )
    if coreg_path is None:
        raise RuntimeError(gpt.last_error_summary())

    coreg_stack_gpt = build_gpt(
        product=coreg_path,
        outdir=outdir_path,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    coreg_deburst = coreg_stack_gpt.deburst(output_name=f"{output_name_prefix}_coreg_deb")
    if coreg_deburst is None:
        raise RuntimeError(coreg_stack_gpt.last_error_summary())

    ifg_gpt = build_gpt(
        product=coreg_deburst,
        outdir=outdir_path,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    ifg_path = ifg_gpt.interferogram(output_name=f"{output_name_prefix}_ifg_raw")
    if ifg_path is None:
        raise RuntimeError(ifg_gpt.last_error_summary())
    ifg_deburst = ifg_gpt.deburst(output_name=f"{output_name_prefix}_ifg")
    if ifg_deburst is None:
        raise RuntimeError(ifg_gpt.last_error_summary())

    final_coreg = coreg_deburst
    final_ifg = ifg_deburst
    if subset_outputs:
        topo_gpt = build_gpt(
            product=ifg_deburst,
            outdir=outdir_path,
            format=format,
            gpt_path=gpt_path,
            memory=memory,
            parallelism=parallelism,
            timeout=timeout,
            snap_userdir=snap_userdir,
        )
        topo_path = topo_gpt.topo_phase_removal(
            dem_name=_external_dem_name(external_dem_file, "SRTM 1Sec HGT"),
            external_dem_file=external_dem_file,
            output_name=f"{output_name_prefix}_ifg_topo",
        )
        if topo_path is None:
            raise RuntimeError(topo_gpt.last_error_summary())
        final_ifg = topo_gpt.subset(
            region=subset_region,
            geo_region=polygon_wkt,
            copy_metadata=True,
            output_name=f"{output_name_prefix}_ifg_subset",
        )
        if final_ifg is None:
            raise RuntimeError(topo_gpt.last_error_summary())

        subset_coreg_gpt = build_gpt(
            product=coreg_deburst,
            outdir=outdir_path,
            format=format,
            gpt_path=gpt_path,
            memory=memory,
            parallelism=parallelism,
            timeout=timeout,
            snap_userdir=snap_userdir,
        )
        final_coreg = subset_coreg_gpt.subset(
            region=subset_region,
            geo_region=polygon_wkt,
            copy_metadata=True,
            output_name=f"{output_name_prefix}_coreg_subset",
        )
        if final_coreg is None:
            raise RuntimeError(subset_coreg_gpt.last_error_summary())

    return TopsarCoregIfgResult(
        coreg_path=final_coreg,
        ifg_path=final_ifg,
        pipeline_name=pipeline_name,
    )


def run_topsar_export(
    coreg_products: Iterable[str | Path],
    ifg_products: Iterable[str | Path],
    outdir: str | Path,
    *,
    target_folder: str | Path,
    master_count: int,
    polygon_wkt: str | None = None,
    subset_region: str = "0,0,0,0",
    external_dem_file: str | Path | None = None,
    selected_polarisations: list[str] | None = None,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> str:
    """Run the upstream-equivalent TOPSAR StaMPS export branch."""
    coreg_paths = _as_path_list(coreg_products)
    ifg_paths = _as_path_list(ifg_products)
    pipeline_name = select_topsar_export_pipeline(
        master_count=master_count,
        external_dem_file=external_dem_file,
    )
    gpt = build_gpt(
        product=coreg_paths[0],
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    if pipeline_name == "topsar_export":
        if len(coreg_paths) != 1 or len(ifg_paths) != 1:
            raise ValueError("topsar_export expects exactly one coreg product and one IFG product")
        exported = gpt.stamps_export_pair(
            coreg_product=coreg_paths[0],
            ifg_product=ifg_paths[0],
            target_folder=target_folder,
            output_name="topsar_export",
        )
        if exported is None:
            raise RuntimeError(gpt.last_error_summary())
        return exported

    merged_coreg = gpt.topsar_merge_products(
        source_products=coreg_paths,
        selected_polarisations=selected_polarisations,
        output_name="topsar_export_coreg_merge",
    )
    if merged_coreg is None:
        raise RuntimeError(gpt.last_error_summary())
    merged_ifg = gpt.topsar_merge_products(
        source_products=ifg_paths,
        selected_polarisations=selected_polarisations,
        output_name="topsar_export_ifg_merge",
    )
    if merged_ifg is None:
        raise RuntimeError(gpt.last_error_summary())

    topo_gpt = build_gpt(
        product=merged_ifg,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    topo_ifg = topo_gpt.topo_phase_removal(
        dem_name=_external_dem_name(external_dem_file, "SRTM 1Sec HGT"),
        external_dem_file=external_dem_file,
        output_name="topsar_export_ifg_topo",
    )
    if topo_ifg is None:
        raise RuntimeError(topo_gpt.last_error_summary())
    subset_ifg = topo_gpt.subset(
        region=subset_region,
        geo_region=polygon_wkt,
        copy_metadata=True,
        output_name="topsar_export_ifg_subset",
    )
    if subset_ifg is None:
        raise RuntimeError(topo_gpt.last_error_summary())

    subset_coreg_gpt = build_gpt(
        product=merged_coreg,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    subset_coreg = subset_coreg_gpt.subset(
        region=subset_region,
        geo_region=polygon_wkt,
        copy_metadata=True,
        output_name="topsar_export_coreg_subset",
    )
    if subset_coreg is None:
        raise RuntimeError(subset_coreg_gpt.last_error_summary())

    export_gpt = build_gpt(
        product=subset_coreg,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    exported = export_gpt.stamps_export_pair(
        coreg_product=subset_coreg,
        ifg_product=subset_ifg,
        target_folder=target_folder,
        output_name="topsar_export_mergeiw",
    )
    if exported is None:
        raise RuntimeError(export_gpt.last_error_summary())
    return exported


def run_stripmap_subset(
    product: str | Path,
    outdir: str | Path,
    *,
    region: str | None = None,
    polygon_wkt: str | None = None,
    output_name: str | None = None,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> str:
    """Run the Stripmap subset branch."""
    gpt = build_gpt(
        product=product,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    subset_path = gpt.subset(
        region=region,
        geo_region=polygon_wkt,
        copy_metadata=True,
        output_name=output_name,
    )
    if subset_path is None:
        raise RuntimeError(gpt.last_error_summary())
    return subset_path


def run_stripmap_coreg(
    pair: PairProducts,
    outdir: str | Path,
    *,
    external_dem_file: str | Path | None = None,
    output_name: str | None = None,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> str:
    """Run the Stripmap DEM-assisted coregistration branch."""
    gpt = build_gpt(
        product=pair.master,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    coreg_path = gpt.dem_assisted_coregistration_pair(
        master_product=pair.master,
        slave_product=pair.slave,
        dem_name=_external_dem_name(external_dem_file, "SRTM 1Sec HGT"),
        external_dem_file=external_dem_file,
        output_name=output_name,
    )
    if coreg_path is None:
        raise RuntimeError(gpt.last_error_summary())
    return coreg_path


def run_stripmap_ifg(
    coreg_product: str | Path,
    outdir: str | Path,
    *,
    external_dem_file: str | Path | None = None,
    output_name_prefix: str = "stripmap",
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> str:
    """Run the Stripmap interferogram/topographic-phase branch."""
    gpt = build_gpt(
        product=coreg_product,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    ifg_path = gpt.interferogram(
        dem_name="SRTM 3Sec",
        output_name=f"{output_name_prefix}_ifg",
    )
    if ifg_path is None:
        raise RuntimeError(gpt.last_error_summary())
    topo_path = gpt.topo_phase_removal(
        dem_name=_external_dem_name(external_dem_file, "SRTM 1Sec HGT"),
        external_dem_file=external_dem_file,
        output_name=f"{output_name_prefix}_topo",
    )
    if topo_path is None:
        raise RuntimeError(gpt.last_error_summary())
    return topo_path


def run_stripmap_export(
    coreg_product: str | Path,
    ifg_product: str | Path,
    outdir: str | Path,
    *,
    target_folder: str | Path,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> str:
    """Run the Stripmap StaMPS export branch."""
    gpt = build_gpt(
        product=coreg_product,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
    exported = gpt.stamps_export_pair(
        coreg_product=coreg_product,
        ifg_product=ifg_product,
        target_folder=target_folder,
        output_name="stripmap_export",
    )
    if exported is None:
        raise RuntimeError(gpt.last_error_summary())
    return exported


__all__ = [
    "PAIRWISE_GRAPH_NAMES",
    "PairProducts",
    "PipelineDefinition",
    "PipelineStep",
    "SNAP2STAMPS_GRAPH_PIPELINES",
    "SNAP2STAMPS_PIPELINES",
    "SNAP2STAMPS_WORKFLOWS",
    "SNAP2STAMPS_WORKFLOW_INPUTS",
    "TopsarCoregIfgResult",
    "build_gpt",
    "get_pipeline_definition",
    "list_pipeline_names",
    "pipeline_requires_multi_input",
    "pipeline_requires_pair",
    "prepare_pair",
    "run_graph_pipeline",
    "run_pair_graph_pipeline",
    "run_pair_workflow",
    "run_processing_pipeline",
    "run_stripmap_coreg",
    "run_stripmap_export",
    "run_stripmap_ifg",
    "run_stripmap_subset",
    "run_topsar_coreg_ifg",
    "run_topsar_export",
    "run_topsar_split_apply_orbit",
    "run_workflow",
    "select_stripmap_coreg_pipeline",
    "select_stripmap_ifg_pipeline",
    "select_topsar_coreg_ifg_pipeline",
    "select_topsar_export_pipeline",
    "select_topsar_split_pipeline",
]
