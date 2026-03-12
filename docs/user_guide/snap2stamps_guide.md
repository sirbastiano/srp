# SNAP2StaMPS Guide

This guide explains the processing-only `snap2stamps` implementation in `sarpyx.snapflow`.

## Scope

The canonical module is:

- `sarpyx.snapflow.snap2stamps`

It implements the upstream processing branches as Python helpers and branch selectors:

- Sentinel-1 TOPSAR split/apply-orbit, coregistration/interferogram, and export variants
- TerraSAR-X/TanDEM-X Stripmap subset, coregistration, interferogram/topo-phase, and export variants
- Legacy pair-workflow helpers kept for notebook and test compatibility

It does not recreate the upstream CLI/config runner, plotting scripts, or download automation.

## Quick Start

```python
from sarpyx.snapflow.snap2stamps import (
    PairProducts,
    list_pipeline_names,
    run_processing_pipeline,
)

print(list_pipeline_names("topsar"))

pair = PairProducts("master_orbit.dim", "slave_orbit.dim")
result = run_processing_pipeline(
    "topsar_coreg_ifg_subset",
    pair=pair,
    outdir="data/output/insar",
    master_count=1,
    burst_count=3,
    subset_region="0,0,1024,1024",
)

print(result.pipeline_name, result.coreg_path, result.ifg_path)
```

## TOPSAR Branches

Branch selection follows the upstream behavior:

- `select_topsar_split_pipeline(source_count)` chooses single-slice vs assemble+split
- `select_topsar_coreg_ifg_pipeline(master_count, burst_count, external_dem_file=...)` chooses subset, no-ESD, and external-DEM variants
- `select_topsar_export_pipeline(master_count, external_dem_file=...)` chooses direct export vs merge-IW export

Main helpers:

- `run_topsar_split_apply_orbit(...)`
- `run_topsar_coreg_ifg(...)`
- `run_topsar_export(...)`

## Stripmap Branches

Stripmap processing is exposed through:

- `run_stripmap_subset(...)`
- `run_stripmap_coreg(...)`
- `run_stripmap_ifg(...)`
- `run_stripmap_export(...)`

Selectors:

- `select_stripmap_coreg_pipeline(external_dem_file=...)`
- `select_stripmap_ifg_pipeline(external_dem_file=...)`

## Legacy Pair Workflows

The older convenience workflow helpers remain available:

- `prepare_pair(...)`
- `run_pair_workflow(...)`
- `run_graph_pipeline(...)`
- `run_pair_graph_pipeline(...)`

The deprecated import path `sarpyx.snapflow.snap2stamps_pipelines` re-exports the same API.

## Notes

- Use `list_pipeline_names()` and `get_pipeline_definition()` to inspect available branches.
- Use `pipeline_requires_pair()` and `pipeline_requires_multi_input()` to validate caller inputs before dispatch.
- SNAP `gpt` and any DEM inputs must still be provided by the runtime environment.
