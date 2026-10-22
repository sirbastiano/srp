from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from sarpyx.snapflow import runtime as runtime_mod


def test_terrain_correction_can_preserve_subap_source_bands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeOp:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def TerrainCorrection(self, **kwargs):
            self.calls.append(kwargs)
            return tmp_path / "tc.dim"

    def run(include_subap_source_bands: bool) -> dict[str, object]:
        op = FakeOp()
        ctx = runtime_mod.PipelineContext(
            tmp_path / "input.SAFE",
            tmp_path / "source.dim",
            tmp_path,
            op,
            {},
            {},
            {},
        )
        monkeypatch.setattr(runtime_mod, "materialized_band_names", lambda _path: ["Sigma0_VH", "i_IW1_VH_SA1"])
        runtime_mod.step_terrain_correction(
            ctx,
            source_bands=runtime_mod.ALL_SOURCE_BANDS,
            include_subap_source_bands=include_subap_source_bands,
        )
        return op.calls[0]

    assert run(False)["source_bands"] == ["Sigma0_VH"]
    assert run(True)["source_bands"] == ["Sigma0_VH", "i_IW1_VH_SA1"]


def test_do_subaps_cleans_only_obsolete_current_swath_intermediates(tmp_path: Path) -> None:
    output_dir = tmp_path / "IW3"
    output_dir.mkdir()
    keep = _dimap(output_dir / "deb.dim")
    old = _dimap(output_dir / "cal.dim")
    outside = _dimap(tmp_path / "other" / "external.dim")

    class FakeOp:
        prod_path = keep

        def do_subaps(self, **_kwargs):
            return keep

    ctx = runtime_mod.PipelineContext(
        tmp_path / "input.SAFE",
        keep,
        output_dir,
        FakeOp(),
        {"cal": old, "deb": keep, "external": outside},
        {"cleanup_before_subaps": True, "keep_intermediate": False},
        {},
    )

    runtime_mod.step_do_subaps(ctx)

    assert keep.exists()
    assert keep.with_suffix(".data").exists()
    assert not old.exists()
    assert not old.with_suffix(".data").exists()
    assert outside.exists()
    assert outside.with_suffix(".data").exists()


def test_cleanup_intermediates_runs_after_tiling_and_keeps_final_product(tmp_path: Path) -> None:
    from sarpyx.snapflow.preprocessing import _cleanup_intermediates, _tiling_created_final_tiles

    output_dir = tmp_path / "IW3"
    output_dir.mkdir()
    final_tc = _dimap(output_dir / "tc.dim")
    old = _dimap(output_dir / "cal.dim")

    tiling = {
        "tiling_result": {"actual_tiles": ["001U_001R"]},
        "validation_group": {"results": [{"tile": "001U_001R", "status": "success"}]},
    }
    assert _tiling_created_final_tiles(tiling) is True

    if _tiling_created_final_tiles(tiling):
        _cleanup_intermediates(output_dir, final_tc, keep_intermediate=False)

    assert final_tc.exists()
    assert final_tc.with_suffix(".data").exists()
    assert not old.exists()
    assert not old.with_suffix(".data").exists()


def test_sentinel_tops_preprocessing_uses_output_local_tmp_and_removes_after_tiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sarpyx.snapflow import preprocessing

    product = tmp_path / "input.SAFE"
    product.mkdir()
    output_dir = tmp_path / "worldsar_output"
    finalized: list[dict[str, object]] = []

    class FakeOp:
        def __init__(self, product_path, outdir):
            self.prod_path = Path(product_path)
            self.outdir = Path(outdir)

    def fake_create_operator(product_path, output_dir, *_args, **_kwargs):
        return FakeOp(product_path, output_dir)

    def fake_run_step(ctx, step):
        assert step.name == "TopsarSplit"
        assert ctx.output_dir.name == "IW3"
        assert ctx.output_dir.parent.parent == output_dir
        assert ctx.output_dir.parent.name.startswith(preprocessing.ISOLATED_PREPROCESSING_PREFIX)
        split = _dimap(ctx.output_dir / "split.dim")
        ctx.op.prod_path = split
        return split

    def fake_run_steps(ctx, _steps):
        assert not ctx.metadata.get("cleanup_before_subaps")
        _dimap(ctx.output_dir / "stale.dim")
        final = _dimap(ctx.output_dir / "tc.dim")
        ctx.op.prod_path = final
        ctx.saved["tiling"] = {
            "tiling_result": {"actual_tiles": ["001U_001R"]},
            "validation_group": {"results": [{"tile": "001U_001R", "status": "success"}]},
        }
        return ctx

    def fake_finalize(_product_wkt, _grid_path, _cuts_outdir, swath_products, *_args, **_kwargs):
        tmp_root = Path(swath_products["IW3"]).parents[1]
        assert (tmp_root / "IW3" / "stale.dim").exists()
        assert (tmp_root / "IW3" / "tc.dim").exists()
        finalized.append({"called": True})

    monkeypatch.setattr(preprocessing, "run_step", fake_run_step)
    monkeypatch.setattr(preprocessing, "run_steps", fake_run_steps)
    monkeypatch.setattr(preprocessing, "finalize_tops_tiling", fake_finalize)

    result = preprocessing.run_sentinel_tops_pipeline(
        product,
        output_dir,
        create_operator=fake_create_operator,
        product_wkt="POLYGON EMPTY",
        grid_path=tmp_path / "grid.geojson",
        cuts_outdir=output_dir / "tiles",
        sentinel_swath="IW3",
        keep_intermediate=False,
    )

    assert finalized == [{"called": True}]
    assert any(
        part.startswith(preprocessing.ISOLATED_PREPROCESSING_PREFIX)
        for part in result["IW3"].parts
    )
    assert not result["IW3"].exists()
    assert not list(output_dir.glob(f"{preprocessing.ISOLATED_PREPROCESSING_PREFIX}*"))
    assert not (output_dir / "IW3").exists()


def test_worldsar_runner_accepts_deleted_isolated_intermediate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sarpyx.snapflow import preprocessing, runner

    output_dir = tmp_path / "worldsar_output"
    tmp_root = preprocessing._create_isolated_preprocessing_root(output_dir)
    deleted_intermediate = tmp_root / "IW3" / "tc.dim"
    preprocessing._cleanup_isolated_preprocessing_root(deleted_intermediate, output_dir)

    def fake_pipeline(*_args, **_kwargs):
        return {"IW3": deleted_intermediate}

    monkeypatch.setitem(runner.ROUTER, "S1TOPS", fake_pipeline)

    result = runner._run_preprocessing(
        tmp_path / "input.SAFE",
        output_dir,
        "S1TOPS",
        orbit_type="Sentinel Precise (Auto Download)",
        orbit_continue_on_fail=False,
        gpt_memory="1G",
        gpt_parallelism=1,
        gpt_timeout=None,
        gpt_cache_size="1G",
        keep_intermediate=False,
    )

    assert result == {"IW3": deleted_intermediate}


def test_find_existing_intermediates_searches_isolated_tmp_roots(tmp_path: Path) -> None:
    from sarpyx.snapflow import preprocessing, runner

    output_dir = tmp_path / "worldsar_output"
    tmp_root = preprocessing._create_isolated_preprocessing_root(output_dir)
    intermediate = _dimap(tmp_root / "IW3" / "tc.dim")

    result = runner._find_existing_intermediates(output_dir, "S1TOPS", sentinel_swath="IW3")

    assert result == {"IW3": intermediate}


def test_final_cleanup_removes_nested_tiling_intermediates(tmp_path: Path) -> None:
    from sarpyx.snapflow.runner import _cleanup_final_intermediates

    output_dir = tmp_path / "out"
    final_tc = _dimap(output_dir / "IW1" / "tc.dim")
    nested_tc = _dimap(output_dir / "IW1" / "worldsar_tc_epsg" / "tc_EPSG32632.dim")
    nested_reprojected = _dimap(output_dir / "IW1" / "worldsar_reprojected" / "tc_EPSG32633.dim")
    nested_subap = _dimap(output_dir / "IW1" / "worldsar_subap_tc" / "tc_SA1_EPSG32632.dim")
    pdf = output_dir / "pdfs" / "product_zarr_validation_report.pdf"
    cut_report = output_dir / "pdfs" / "product" / "IW1" / "product_cuts_report_SUCCESS.txt"
    cut_manifest = cut_report.with_suffix(".json")
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    cut_report.parent.mkdir(parents=True)
    cut_report.write_text("cut report", encoding="utf-8")
    cut_manifest.write_text("{}", encoding="utf-8")

    _cleanup_final_intermediates({"IW1": final_tc}, output_dir)

    assert not final_tc.exists()
    assert not final_tc.with_suffix(".data").exists()
    assert not nested_tc.parent.exists()
    assert not nested_reprojected.parent.exists()
    assert not nested_subap.parent.exists()
    assert pdf.exists()
    assert not cut_report.exists()
    assert not cut_manifest.exists()
    assert not (output_dir / "pdfs" / "product").exists()


def test_named_isolated_preprocessing_root_replaces_previous_run(tmp_path: Path) -> None:
    from sarpyx.snapflow import preprocessing

    output_dir = tmp_path / "worldsar_output"
    product = "S1A_IW_SLC__1SDV_20180921T175241_20180921T175308_023800_0298AE_B95D"
    first_root = preprocessing._create_isolated_preprocessing_root(output_dir, product)
    stale_file = first_root / "stale.dim"
    stale_file.write_text("old run")

    second_root = preprocessing._create_isolated_preprocessing_root(output_dir, product)

    assert second_root == output_dir / f"worldsar_tmp_{product}"
    assert second_root.is_dir()
    assert not stale_file.exists()


@pytest.mark.parametrize("name", ["../escape", "nested/name", "", "bad name"])
def test_named_isolated_preprocessing_root_rejects_unsafe_names(
    tmp_path: Path,
    name: str,
) -> None:
    from sarpyx.snapflow import preprocessing

    with pytest.raises(ValueError):
        preprocessing._create_isolated_preprocessing_root(tmp_path, name)




@pytest.mark.parametrize(
    "tiling",
    [
        {},
        {"tiling_result": {"actual_tiles": []}, "validation_group": {"results": []}},
        {
            "tiling_result": {"actual_tiles": ["001U_001R"]},
            "validation_group": {"results": [{"tile": "001U_001R", "status": "failed"}]},
        },
        {
            "error": RuntimeError("cut failed"),
            "tiling_result": {"actual_tiles": ["001U_001R"]},
            "validation_group": {"results": [{"tile": "001U_001R", "status": "success"}]},
        },
    ],
)
def test_cleanup_gate_requires_successful_final_tiles(tiling: dict) -> None:
    from sarpyx.snapflow.preprocessing import _tiling_created_final_tiles

    assert _tiling_created_final_tiles(tiling) is False


def test_epsg_terrain_correction_merges_geocoded_subap_bands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from sarpyx.snapflow import tile_crs
    from sarpyx.snapflow.dimap import materialized_band_names

    source = _dimap_with_bands(
        tmp_path / "pdec.dim",
        ["i_IW1_VV", "q_IW1_VV", "i_IW1_VV_SA1", "q_IW1_VV_SA1"],
    )
    target = _dimap_with_bands(
        tmp_path / "worldsar_tc_epsg" / "pdec_TC_EPSG32633.dim",
        ["i_IW1_VV", "q_IW1_VV"],
    )
    calls: list[dict[str, object]] = []

    def fake_run_gpt_op(product_path, output_dir, _format, op_name, **kwargs):
        calls.append({"product_path": Path(product_path), "output_dir": Path(output_dir), "op_name": op_name, **kwargs})
        return _dimap_with_bands(Path(output_dir) / f"{kwargs['output_name']}.dim", ["i_IW1_VV", "q_IW1_VV"])

    monkeypatch.setattr(tile_crs, "run_gpt_op", fake_run_gpt_op)

    tile_crs._ensure_subap_bands_terrain_corrected(
        source,
        target,
        32633,
        {"map_projection": "EPSG:32633", "pixel_spacing_in_meter": 10.0},
        {"gpt_memory": "4G"},
    )

    assert calls[0]["op_name"] == "TerrainCorrection"
    assert calls[0]["source_bands"] == ["i_IW1_VV", "q_IW1_VV"]
    redirect_product = target.parent / "worldsar_subap_tc" / f"{source.stem}_SA1_source.dim"
    redirect_root = ET.parse(redirect_product).getroot()
    hrefs = [node.get("href") for node in redirect_root.findall(".//*[@href]")]
    assert hrefs
    assert all(not Path(href).is_absolute() for href in hrefs if href)
    assert all(".." not in Path(href).parts for href in hrefs if href)
    assert all((redirect_product.parent / href).resolve().exists() for href in hrefs if href)
    assert (redirect_product.with_suffix(".data") / "i_IW1_VV_SA1.img").exists()
    assert (redirect_product.with_suffix(".data") / "q_IW1_VV_SA1.img").exists()
    assert "i_IW1_VV_SA1" in materialized_band_names(target)
    assert "q_IW1_VV_SA1" in materialized_band_names(target)
    assert (target.with_suffix(".data") / "i_IW1_VV_SA1.img").exists()
    assert (target.with_suffix(".data") / "q_IW1_VV_SA1.img").exists()


def test_subap_terrain_correction_fails_before_gpt_when_payload_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sarpyx.snapflow import tile_crs

    source = _dimap_with_bands(
        tmp_path / "pdec.dim",
        ["i_IW1_VV", "q_IW1_VV", "i_IW1_VV_SA1", "q_IW1_VV_SA1"],
    )
    target = _dimap_with_bands(tmp_path / "tc.dim", ["i_IW1_VV", "q_IW1_VV"])
    (source.with_suffix(".data") / "i_IW1_VV_SA1.img").unlink()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_gpt_op should not be called for incomplete redirect payloads")

    monkeypatch.setattr(tile_crs, "run_gpt_op", fail_if_called)

    with pytest.raises(FileNotFoundError, match="i_IW1_VV_SA1.img"):
        tile_crs._ensure_subap_bands_terrain_corrected(
            source,
            target,
            32633,
            {"map_projection": "EPSG:32633", "pixel_spacing_in_meter": 10.0},
            {"gpt_memory": "4G"},
        )


def test_subap_terrain_correction_discards_partial_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sarpyx.snapflow import tile_crs
    from sarpyx.snapflow.dimap import materialized_band_names

    source = _dimap_with_bands(
        tmp_path / "pdec.dim",
        ["i_IW1_VV", "q_IW1_VV", "i_IW1_VV_SA1", "q_IW1_VV_SA1"],
    )
    target = _dimap_with_bands(
        tmp_path / "worldsar_tc_epsg" / "pdec_TC_EPSG32633.dim",
        ["i_IW1_VV", "q_IW1_VV"],
    )
    partial = _dimap_with_bands(
        target.parent / "worldsar_subap_tc" / "pdec_SA1_TC_EPSG32633.dim",
        ["i_IW1_VV", "q_IW1_VV"],
    )
    (partial.with_suffix(".data") / "i_IW1_VV.img").unlink()
    calls: list[Path] = []

    def fake_run_gpt_op(product_path, output_dir, _format, _op_name, **kwargs):
        calls.append(Path(product_path))
        return _dimap_with_bands(
            Path(output_dir) / f"{kwargs['output_name']}.dim",
            ["i_IW1_VV", "q_IW1_VV"],
        )

    monkeypatch.setattr(tile_crs, "run_gpt_op", fake_run_gpt_op)

    tile_crs._ensure_subap_bands_terrain_corrected(
        source,
        target,
        32633,
        {"map_projection": "EPSG:32633", "pixel_spacing_in_meter": 10.0},
        {"gpt_memory": "4G"},
    )

    assert calls == [target.parent / "worldsar_subap_tc" / "pdec_SA1_source.dim"]
    assert "i_IW1_VV_SA1" in materialized_band_names(target)
    assert "q_IW1_VV_SA1" in materialized_band_names(target)


def test_subap_terrain_correction_retries_with_rebuilt_redirect_after_gpt_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sarpyx.snapflow import tile_crs
    from sarpyx.snapflow.dimap import materialized_band_names

    source = _dimap_with_bands(
        tmp_path / "pdec.dim",
        ["i_IW1_VV", "q_IW1_VV", "i_IW1_VV_SA1", "q_IW1_VV_SA1"],
    )
    target = _dimap_with_bands(
        tmp_path / "worldsar_tc_epsg" / "pdec_TC_EPSG32633.dim",
        ["i_IW1_VV", "q_IW1_VV"],
    )
    calls: list[Path] = []

    def fake_run_gpt_op(product_path, output_dir, _format, _op_name, **kwargs):
        calls.append(Path(product_path))
        if len(calls) == 1:
            _dimap_with_bands(Path(output_dir) / f"{kwargs['output_name']}.dim", ["i_IW1_VV"])
            raise RuntimeError("transient SNAP failure")
        return _dimap_with_bands(
            Path(output_dir) / f"{kwargs['output_name']}.dim",
            ["i_IW1_VV", "q_IW1_VV"],
        )

    monkeypatch.setattr(tile_crs, "run_gpt_op", fake_run_gpt_op)

    tile_crs._ensure_subap_bands_terrain_corrected(
        source,
        target,
        32633,
        {"map_projection": "EPSG:32633", "pixel_spacing_in_meter": 10.0},
        {"gpt_memory": "4G"},
    )

    assert calls == [
        target.parent / "worldsar_subap_tc" / "pdec_SA1_source.dim",
        target.parent / "worldsar_subap_tc" / "pdec_SA1_source.dim",
    ]
    assert "i_IW1_VV_SA1" in materialized_band_names(target)
    assert "q_IW1_VV_SA1" in materialized_band_names(target)


def test_prepare_products_by_epsg_merges_subaps_when_source_epsg_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from sarpyx.snapflow import tile_crs

    intermediate = tmp_path / "tc.dim"
    source = tmp_path / "pdec.dim"
    calls: list[tuple[Path, Path, int, dict[str, object], dict[str, object]]] = []

    monkeypatch.setattr(tile_crs, "dim_epsg", lambda _path: 32632)
    monkeypatch.setattr(tile_crs, "_ensure_subap_bands_terrain_corrected", lambda *args: calls.append(args))

    products = tile_crs.prepare_products_by_epsg(
        intermediate,
        [32632],
        {"gpt_memory": "4G"},
        terrain_correction={"source_product": source, "params": {"map_projection": "AUTO:42001", "output_name": "tc"}},
    )

    assert products == {32632: intermediate}
    assert calls == [(source, intermediate, 32632, {"map_projection": "EPSG:32632"}, {"gpt_memory": "4G"})]


def _dimap(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dim", encoding="utf-8")
    data_dir = path.with_suffix(".data")
    data_dir.mkdir()
    (data_dir / "band.img").write_bytes(b"data")
    return path


def _dimap_with_bands(path: Path, band_names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = path.with_suffix(".data")
    data_dir.mkdir(parents=True, exist_ok=True)
    root = ET.Element("Dimap_Document")
    raster = ET.SubElement(root, "Raster_Dimensions")
    ET.SubElement(raster, "NCOLS").text = "4"
    ET.SubElement(raster, "NROWS").text = "4"
    ET.SubElement(raster, "NBANDS").text = str(len(band_names))
    image = ET.SubElement(root, "Image_Interpretation")
    access = ET.SubElement(root, "Data_Access")
    for index, band_name in enumerate(band_names):
        _write_band_files(data_dir, band_name)
        spectral = ET.SubElement(image, "Spectral_Band_Info")
        ET.SubElement(spectral, "BAND_INDEX").text = str(index)
        ET.SubElement(spectral, "BAND_NAME").text = band_name
        ET.SubElement(spectral, "BAND_DESCRIPTION").text = band_name
        ET.SubElement(spectral, "PHYSICAL_UNIT").text = "real" if band_name.startswith("i_") else "imaginary"
        ET.SubElement(spectral, "BAND_RASTER_WIDTH").text = "4"
        ET.SubElement(spectral, "BAND_RASTER_HEIGHT").text = "4"
        data_file = ET.SubElement(access, "Data_File")
        ET.SubElement(data_file, "BAND_INDEX").text = str(index)
        ET.SubElement(data_file, "DATA_FILE_PATH", href=f"{data_dir.name}/{band_name}.hdr")
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=False)
    return path


def _write_band_files(data_dir: Path, band_name: str) -> None:
    (data_dir / f"{band_name}.hdr").write_text(
        "\n".join(
            [
                "ENVI",
                "samples = 4",
                "lines = 4",
                "bands = 1",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bsq",
                "byte order = 1",
                f"band names = {{ {band_name} }}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / f"{band_name}.img").write_bytes(b"0" * 64)
