from pathlib import Path

from sarpyx.snapflow.snap2stamps_pipelines import (
    Snap2StampsRunner,
    verify_graph_coverage,
)


def _write_minimal_project_conf(path: Path) -> None:
    path.write_text(
        """
[PROJECT_DEFINITION]
PROJECTFOLDER = {project}
GRAPHSFOLDER = {graphs}

[PROC_OPTIONS]
OVERWRITE = N
SMARTHDD = Y
PLOTTING = N

[PROC_PARAMETERS]
SENSOR = S1
POLARISATION = VV
MASTER = {project}/Master
MASTERSEL = AUTO
EXTDEM =

[AOI_DEFINITION]
AOI_MODE = BBOX
LONMIN = 0.0
LATMIN = 0.0
LONMAX = 1.0
LATMAX = 1.0

[COMPUTING_RESOURCES]
CPU = 4
CACHE = 8G
""".strip().format(project=path.parent.as_posix(), graphs=(Path(__file__).resolve().parents[1] / "snap2stamps" / "graphs").as_posix()),
        encoding="utf-8",
    )


def test_verify_graph_coverage_complete():
    repo_root = Path(__file__).resolve().parents[1]
    graph_dir = repo_root / "snap2stamps" / "graphs"
    report = verify_graph_coverage(graph_dir)

    assert report["graph_count"] > 0
    assert report["ok"] is True
    assert report["unmapped_graphs"] == []
    assert report["missing_template_files"] == []


def test_runner_builds_bbox_wkt(tmp_path: Path):
    conf = tmp_path / "project_topsar.conf"
    _write_minimal_project_conf(conf)

    runner = Snap2StampsRunner.from_project_file(conf)
    wkt = runner._resolve_aoi_wkt()

    assert wkt.startswith("POLYGON")
    assert "0.0 0.0" in wkt


def test_render_graph_template(tmp_path: Path):
    conf = tmp_path / "project_topsar.conf"
    _write_minimal_project_conf(conf)
    runner = Snap2StampsRunner.from_project_file(conf)

    generated = tmp_path / "graphs" / "subset_2run.xml"
    out = runner._render_graph(
        graph_name="stripmap_tsx_subset",
        replacements={
            "INPUTXML": "/tmp/input.xml",
            "OUTPUTSUBSETFOLDER": "/tmp/out",
            "OUTPUTFILE": "scene_sub.dim",
            "POLYGON": "POLYGON ((0 0,1 0,1 1,0 1,0 0))",
        },
        generated_graph_path=generated,
    )

    text = out.read_text(encoding="utf-8")
    assert "INPUTXML" not in text
    assert "scene_sub.dim" in text
    assert "/tmp/input.xml" in text
