from pathlib import Path

import pytest

from sarpyx.snapflow import (
    SNAP2STAMPS_PIPELINES,
    get_pipeline_definition,
    list_pipeline_names,
    pipeline_requires_multi_input,
    pipeline_requires_pair,
    run_processing_pipeline,
)
from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.snap2stamps import (
    PAIRWISE_GRAPH_NAMES,
    PairProducts,
    SNAP2STAMPS_WORKFLOWS,
    SNAP2STAMPS_WORKFLOW_INPUTS,
    prepare_pair,
    run_graph_pipeline,
    run_pair_graph_pipeline,
    run_pair_workflow,
    select_stripmap_coreg_pipeline,
    select_stripmap_ifg_pipeline,
    select_topsar_coreg_ifg_pipeline,
    select_topsar_export_pipeline,
    select_topsar_split_pipeline,
)
from sarpyx.snapflow.snap2stamps_pipelines import PairProducts as CompatPairProducts


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    return path


def test_run_graph_supports_parameters(tmp_path: Path):
    product = _touch(tmp_path / "input.SAFE")
    gpt = GPT(product=product, outdir=tmp_path / "out")
    graph = _touch(tmp_path / "graph.xml")
    target = tmp_path / "out" / "target.dim"

    def fake_execute() -> bool:
        target.write_text("done", encoding="utf-8")
        return True

    gpt._execute_command = fake_execute  # type: ignore[method-assign]
    out = gpt.run_graph(
        graph_path=graph,
        output_path=target,
        parameters={"master": product, "enabled": True, "count": 3},
    )

    assert out == target.as_posix()
    assert any(part.startswith('-Pmaster="') for part in gpt.current_cmd)
    assert "-Penabled=true" in gpt.current_cmd
    assert "-Pcount=3" in gpt.current_cmd


def test_topsar_coregistration_writes_pair_graph(tmp_path: Path):
    master = _touch(tmp_path / "master.dim")
    slave = _touch(tmp_path / "slave.dim")
    gpt = GPT(product=master, outdir=tmp_path / "out")

    def fake_run_graph(graph_path, output_path, delete_graph=False, parameters=None):
        text = Path(graph_path).read_text(encoding="utf-8")
        assert "<operator>Back-Geocoding</operator>" in text
        assert '<sourceProduct.1 refid="ReadSlave"/>' in text
        assert "${master}" in text
        assert "${slave}" in text
        assert parameters == {
            "master": master,
            "slave": slave,
            "target": Path(output_path),
        }
        Path(output_path).write_text("coreg", encoding="utf-8")
        gpt.prod_path = Path(output_path)
        return Path(output_path).as_posix()

    gpt.run_graph = fake_run_graph  # type: ignore[method-assign]
    out = gpt.topsar_coregistration(
        master_product=master,
        slave_product=slave,
        output_name="pair_coreg",
        keep_graph=True,
    )

    assert out.endswith("pair_coreg.dim")
    assert Path(out).exists()


def test_pipeline_registry_covers_topsar_and_stripmap():
    assert "topsar_coreg_ifg_subset_no_esd_ext_dem" in SNAP2STAMPS_PIPELINES
    assert "stripmap_dem_assisted_coregistration_ext_dem" in SNAP2STAMPS_PIPELINES
    assert list_pipeline_names("topsar")
    assert list_pipeline_names("stripmap")
    assert get_pipeline_definition("topsar_export").input_kind == "pair"
    assert pipeline_requires_pair("topsar_export") is True
    assert pipeline_requires_multi_input("topsar_export_mergeiw_subset") is True


def test_selector_logic_matches_upstream_branching():
    assert select_topsar_split_pipeline(1) == "topsar_split_applyorbit"
    assert select_topsar_split_pipeline(2) == "topsar_assemble_split_applyorbit"
    assert (
        select_topsar_coreg_ifg_pipeline(master_count=1, burst_count=1, external_dem_file="dem.tif")
        == "topsar_coreg_ifg_subset_no_esd_ext_dem"
    )
    assert (
        select_topsar_coreg_ifg_pipeline(master_count=2, burst_count=3, external_dem_file=None)
        == "topsar_coreg_ifg"
    )
    assert select_topsar_export_pipeline(master_count=1) == "topsar_export"
    assert select_topsar_export_pipeline(master_count=2, external_dem_file="dem.tif") == (
        "topsar_export_mergeiw_subset_ext_dem"
    )
    assert select_stripmap_coreg_pipeline("dem.tif") == "stripmap_dem_assisted_coregistration_ext_dem"
    assert select_stripmap_ifg_pipeline(None) == "stripmap_interferogram_topophase"


def test_single_input_runner_rejects_pair_graph(tmp_path: Path):
    product = _touch(tmp_path / "input.SAFE")
    gpt = GPT(product=product, outdir=tmp_path / "out")

    with pytest.raises(ValueError, match="requires a master/slave pair"):
        run_graph_pipeline(gpt=gpt, graph_name="coregistration")


def test_run_pair_graph_pipeline_uses_topsar_coregistration(tmp_path: Path):
    product = _touch(tmp_path / "input.SAFE")
    gpt = GPT(product=product, outdir=tmp_path / "out")
    pair = PairProducts(master=_touch(tmp_path / "master.dim"), slave=_touch(tmp_path / "slave.dim"))

    calls: list[dict[str, Path]] = []

    def fake_coreg(master_product, slave_product, **kwargs):
        calls.append({"master": Path(master_product), "slave": Path(slave_product)})
        return (tmp_path / "out" / "coreg.dim").as_posix()

    gpt.topsar_coregistration = fake_coreg  # type: ignore[method-assign]
    out = run_pair_graph_pipeline(
        gpt=gpt,
        graph_name="coregistration",
        pair=pair,
        overrides={"topsar_coregistration": {"dem_name": "Copernicus 30m Global DEM"}},
    )

    assert out.endswith("coreg.dim")
    assert calls == [{"master": pair.master, "slave": pair.slave}]


def test_prepare_pair_defaults_to_split_only_for_master_and_slave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    master = _touch(tmp_path / "master.SAFE")
    slave = _touch(tmp_path / "slave.SAFE")
    pair = PairProducts(master=master, slave=slave)
    calls: list[tuple[Path, str]] = []

    class FakeGPT:
        def __init__(self, product, outdir, **kwargs):
            self.prod_path = Path(product)
            self.outdir = Path(outdir)

    def fake_build_gpt(product, outdir, **kwargs):
        return FakeGPT(product=product, outdir=outdir)

    def fake_run_graph_pipeline(gpt, graph_name, overrides=None):
        calls.append((gpt.outdir, graph_name))
        gpt.prod_path = gpt.outdir / f"{Path(gpt.prod_path).stem}_{graph_name}.dim"
        return gpt.prod_path.as_posix()

    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.build_gpt", fake_build_gpt)
    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.run_graph_pipeline", fake_run_graph_pipeline)

    prepared = prepare_pair(pair=pair, outdir=tmp_path / "proc")

    assert calls == [
        (tmp_path / "proc" / "master", "split_orbit"),
        (tmp_path / "proc" / "slave", "split_orbit"),
    ]
    assert prepared.master.name.endswith("_split_orbit.dim")
    assert prepared.slave.name.endswith("_split_orbit.dim")


def test_prepare_pair_can_opt_in_to_deburst(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pair = PairProducts(master=_touch(tmp_path / "master.SAFE"), slave=_touch(tmp_path / "slave.SAFE"))
    calls: list[tuple[Path, str]] = []

    class FakeGPT:
        def __init__(self, product, outdir, **kwargs):
            self.prod_path = Path(product)
            self.outdir = Path(outdir)

    def fake_build_gpt(product, outdir, **kwargs):
        return FakeGPT(product=product, outdir=outdir)

    def fake_run_graph_pipeline(gpt, graph_name, overrides=None):
        calls.append((gpt.outdir, graph_name))
        gpt.prod_path = gpt.outdir / f"{Path(gpt.prod_path).stem}_{graph_name}.dim"
        return gpt.prod_path.as_posix()

    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.build_gpt", fake_build_gpt)
    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.run_graph_pipeline", fake_run_graph_pipeline)

    prepared = prepare_pair(
        pair=pair,
        outdir=tmp_path / "proc",
        preprocess_graphs=("split_orbit", "deburst"),
    )

    assert calls == [
        (tmp_path / "proc" / "master", "split_orbit"),
        (tmp_path / "proc" / "master", "deburst"),
        (tmp_path / "proc" / "slave", "split_orbit"),
        (tmp_path / "proc" / "slave", "deburst"),
    ]
    assert prepared.master.name.endswith("_deburst.dim")
    assert prepared.slave.name.endswith("_deburst.dim")


def test_run_pair_workflow_uses_requested_stamps_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    assert PAIRWISE_GRAPH_NAMES == {"coregistration"}
    assert SNAP2STAMPS_WORKFLOWS["stamps_prep"] == (
        "split_orbit",
        "coregistration",
        "deburst",
        "interferogram",
        "topo_phase_removal",
        "subset",
        "terrain_correction",
    )

    pair = PairProducts(master=_touch(tmp_path / "master.SAFE"), slave=_touch(tmp_path / "slave.SAFE"))
    execution: list[str] = []

    class FakeGPT:
        def __init__(self, product, outdir, **kwargs):
            self.prod_path = Path(product)
            self.outdir = Path(outdir)

    monkeypatch.setattr(
        "sarpyx.snapflow.snap2stamps.prepare_pair",
        lambda **kwargs: PairProducts(
            master=tmp_path / "proc" / "master" / "master_deburst.dim",
            slave=tmp_path / "proc" / "slave" / "slave_deburst.dim",
        ),
    )
    monkeypatch.setattr(
        "sarpyx.snapflow.snap2stamps.build_gpt",
        lambda product, outdir, **kwargs: FakeGPT(product=product, outdir=outdir),
    )

    def fake_run_pair_graph_pipeline(gpt, graph_name, pair, overrides=None):
        execution.append(graph_name)
        gpt.prod_path = gpt.outdir / "pair_coreg.dim"
        return gpt.prod_path.as_posix()

    def fake_run_graph_pipeline(gpt, graph_name, overrides=None):
        execution.append(graph_name)
        gpt.prod_path = gpt.outdir / f"{graph_name}.dim"
        return gpt.prod_path.as_posix()

    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.run_pair_graph_pipeline", fake_run_pair_graph_pipeline)
    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.run_graph_pipeline", fake_run_graph_pipeline)

    out = run_pair_workflow(pair=pair, outdir=tmp_path / "proc", workflow="stamps_prep")

    assert execution == [
        "coregistration",
        "deburst",
        "interferogram",
        "topo_phase_removal",
        "subset",
        "terrain_correction",
    ]
    assert out.endswith("terrain_correction.dim")


def test_workflow_registry_declares_pair_and_single_inputs():
    assert SNAP2STAMPS_WORKFLOW_INPUTS["stamps_prep"] == "pair"
    assert SNAP2STAMPS_WORKFLOW_INPUTS["psi_full"] == "pair"
    assert SNAP2STAMPS_WORKFLOW_INPUTS["psi_post_unwrap"] == "single"


def test_run_processing_pipeline_dispatches_named_topsar_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    pair = PairProducts(master=tmp_path / "master.dim", slave=tmp_path / "slave.dim")
    calls: list[dict[str, object]] = []

    def fake_run_topsar_coreg_ifg(*, pair, outdir, **kwargs):
        calls.append({"pair": pair, "outdir": Path(outdir), **kwargs})
        return "result"

    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.run_topsar_coreg_ifg", fake_run_topsar_coreg_ifg)

    out = run_processing_pipeline(
        "topsar_coreg_ifg_subset",
        pair=pair,
        outdir=tmp_path / "out",
        master_count=1,
        burst_count=2,
    )

    assert out == "result"
    assert len(calls) == 1
    assert calls[0]["pair"] == pair
    assert calls[0]["outdir"] == tmp_path / "out"
    assert calls[0]["pipeline_name"] == "topsar_coreg_ifg_subset"
    assert calls[0]["master_count"] == 1
    assert calls[0]["burst_count"] == 2


def test_run_topsar_coreg_ifg_honors_named_no_esd_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    pair = PairProducts(master=tmp_path / "master.dim", slave=tmp_path / "slave.dim")
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeGPT:
        def __init__(self, product, outdir, **kwargs):
            self.prod_path = Path(product)
            self.outdir = Path(outdir)

        def last_error_summary(self):
            return "error"

        def topsar_coregistration(self, **kwargs):
            calls.append(("coregistration", kwargs))
            self.prod_path = self.outdir / "coreg.dim"
            return self.prod_path.as_posix()

        def deburst(self, output_name=None):
            calls.append(("deburst", {"output_name": output_name, "product": self.prod_path}))
            self.prod_path = self.outdir / f"{output_name}.dim"
            return self.prod_path.as_posix()

        def interferogram(self, output_name=None):
            calls.append(("interferogram", {"output_name": output_name}))
            self.prod_path = self.outdir / f"{output_name}.dim"
            return self.prod_path.as_posix()

        def subset(self, **kwargs):
            calls.append(("subset", kwargs))
            self.prod_path = self.outdir / f"{kwargs['output_name']}.dim"
            return self.prod_path.as_posix()

        def topo_phase_removal(self, **kwargs):
            calls.append(("topo_phase_removal", kwargs))
            self.prod_path = self.outdir / f"{kwargs['output_name']}.dim"
            return self.prod_path.as_posix()

    monkeypatch.setattr("sarpyx.snapflow.snap2stamps.build_gpt", lambda *args, **kwargs: FakeGPT(*args, **kwargs))

    result = run_processing_pipeline(
        "topsar_coreg_ifg_no_esd",
        pair=pair,
        outdir=tmp_path / "out",
        burst_count=3,
        master_count=2,
    )

    assert result.pipeline_name == "topsar_coreg_ifg_no_esd"
    assert calls[0][0] == "coregistration"
    assert calls[0][1]["use_esd"] is False
    assert all(name != "subset" for name, _ in calls)
    assert all(name != "topo_phase_removal" for name, _ in calls)


def test_run_topsar_coreg_ifg_ext_dem_requires_dem(tmp_path: Path):
    pair = PairProducts(master=tmp_path / "master.dim", slave=tmp_path / "slave.dim")

    with pytest.raises(ValueError, match="requires external_dem_file"):
        run_processing_pipeline(
            "topsar_coreg_ifg_subset_ext_dem",
            pair=pair,
            outdir=tmp_path / "out",
            master_count=1,
            burst_count=2,
        )


def test_run_workflow_rejects_pair_workflows(tmp_path: Path):
    gpt = GPT(product=_touch(tmp_path / "input.SAFE"), outdir=tmp_path / "out")

    with pytest.raises(ValueError, match="requires pair inputs"):
        from sarpyx.snapflow.snap2stamps import run_workflow

        run_workflow(gpt, "psi_full")


def test_run_pair_workflow_rejects_single_input_workflows(tmp_path: Path):
    pair = PairProducts(master=_touch(tmp_path / "master.SAFE"), slave=_touch(tmp_path / "slave.SAFE"))

    with pytest.raises(ValueError, match="not pair-based"):
        run_pair_workflow(pair=pair, outdir=tmp_path / "proc", workflow="psi_post_unwrap")


def test_compatibility_alias_reexports_canonical_types():
    assert CompatPairProducts is PairProducts
