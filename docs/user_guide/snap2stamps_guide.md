# SNAP2StaMPS Guide

This guide explains how to run SNAP2StaMPS-like workflows in `sarpyx` using the `snapflow` engine.

## Scope

The implementation is provided by:

- `sarpyx.snapflow.snap2stamps_pipelines.Snap2StampsRunner`

It supports:

- Sentinel-1 TOPSAR flow (prepare, split, coreg/ifg, plotting, export)
- TerraSAR-X/TanDEM-X stripmap flow (unpack, subset, coreg, ifg, plotting, export)
- Graph-template execution from `snap2stamps/graphs/*.xml`

## Prerequisites

- SNAP installed with `gpt` available
- Project config file (`project_topsar.conf` or `project_stripmap.conf`) with required sections
- Optional: `asf_search` if you use Sentinel auto-download (`autoDownload=Y`)

## Quick Start

```python
from sarpyx.snapflow.snap2stamps_pipelines import Snap2StampsRunner

runner = Snap2StampsRunner.from_project_file(
    "snap2stamps/bin/project_topsar.conf",
    gpt_path="/usr/local/snap/bin/gpt",
    memory="16G",
    parallelism=8,
    timeout=7200,
)

# Full automatic run based on SENSOR in config
runner.run_auto()
```

## TOPSAR Usage

### Full automatic run

```python
runner.run_topsar_auto()
```

### Step-by-step run

```python
runner.prepare_topsar_secondaries()
runner.select_topsar_master(mode="AUTO")  # AUTO/FIRST/LAST/MANUAL or numeric index
runner.run_topsar_split_master()
runner.run_topsar_split_secondaries()
runner.run_topsar_coreg_ifg()
runner.run_topsar_plotting("ifg")         # optional
runner.run_topsar_plotting("coreg")       # optional
runner.run_topsar_export()
```

### Optional ASF download

If `autoDownload=Y` in config:

```python
count = runner.download_asf_s1()
print("Downloaded scenes:", count)
```

## Stripmap Usage

### Full automatic run

```python
runner.run_stripmap_auto(master_date="20200101")  # optional master date
```

### Step-by-step run

```python
runner.run_stripmap_unpack()
runner.run_stripmap_prepare_secondaries()
runner.run_stripmap_subset()
runner.run_stripmap_manual_master_selection("20200101")  # optional
runner.run_stripmap_coreg()
runner.run_stripmap_ifg()
runner.run_stripmap_plotting("split")  # optional
runner.run_stripmap_plotting("ifg")    # optional
runner.run_stripmap_plotting("coreg")  # optional
runner.run_stripmap_export()
```

## Graph Coverage Check

Use this to verify all XML templates are mapped:

```python
from sarpyx.snapflow.snap2stamps_pipelines import verify_graph_coverage
report = verify_graph_coverage("snap2stamps/graphs")
print(report["ok"], report["graph_count"])
```

## Notes

- This workflow module is currently not wired into `worldsar` CLI by default.
- You can run all operations directly from Python with `Snap2StampsRunner`.
- For reproducibility, keep project config and graph templates versioned with your data run.
