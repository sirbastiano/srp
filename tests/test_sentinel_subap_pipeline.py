from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from shapely import wkt as shapely_wkt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strip_annotations(node: ast.FunctionDef) -> ast.FunctionDef:
    fn = copy.deepcopy(node)
    fn.returns = None
    for arg in (
        fn.args.posonlyargs
        + fn.args.args
        + fn.args.kwonlyargs
    ):
        arg.annotation = None
    if fn.args.vararg is not None:
        fn.args.vararg.annotation = None
    if fn.args.kwarg is not None:
        fn.args.kwarg.annotation = None
    return fn


def _load_functions_from_file(path: Path, names: list[str], extra_globals: dict[str, object] | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected: list[ast.FunctionDef] = []
    for name in names:
        match = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        selected.append(_strip_annotations(match))
    module_ast = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(module_ast, str(path), "exec"), namespace)
    return [namespace[name] for name in names], namespace


ENGINE_MOD = _load_module_from_path("_engine_for_tests", REPO_ROOT / "sarpyx" / "snapflow" / "engine.py")
DIM_UPDATER_MOD = _load_module_from_path(
    "_dim_updater_for_tests",
    REPO_ROOT / "sarpyx" / "processor" / "core" / "dim_updater.py",
)
WKT_UTILS_MOD = _load_module_from_path("_wkt_utils_for_tests", REPO_ROOT / "sarpyx" / "utils" / "wkt_utils.py")
GPT = ENGINE_MOD.GPT
update_dim_add_bands_from_data_dir = DIM_UPDATER_MOD.update_dim_add_bands_from_data_dir
sentinel1_swath_wkt_extractor_safe = WKT_UTILS_MOD.sentinel1_swath_wkt_extractor_safe

[_sentinel_post_chain, pipeline_sentinel, _resolve_tiling_wkt, _run_tops_swath_tiling], _WORLDSAR_NS = _load_functions_from_file(
    REPO_ROOT / "pyscripts" / "worldsar.py",
    ["_sentinel_post_chain", "pipeline_sentinel", "_resolve_tiling_wkt", "_run_tops_swath_tiling"],
)
[merge_iq_into_pdec], _MERGE_NS = _load_functions_from_file(
    REPO_ROOT / "pyscripts" / "merge_iq_into_pdec.py",
    ["merge_iq_into_pdec"],
    extra_globals={"Path": Path},
)


def _touch(path: Path, text: str = "placeholder") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_envi_stub(data_dir: Path, band_name: str) -> None:
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
    (data_dir / f"{band_name}.img").write_bytes(b"")


def _write_safe_annotation(
    safe_dir: Path,
    filename: str,
    points: list[tuple[float, float]],
) -> Path:
    annotation_dir = safe_dir / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    xml_path = annotation_dir / filename
    body = "".join(
        f"""
        <geolocationGridPoint>
          <latitude>{lat}</latitude>
          <longitude>{lon}</longitude>
        </geolocationGridPoint>
        """
        for lon, lat in points
    )
    xml_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <geolocationGrid>
    <geolocationGridPointList>
      {body}
    </geolocationGridPointList>
  </geolocationGrid>
</product>
""",
        encoding="utf-8",
    )
    return xml_path


def _write_minimal_dim(dim_path: Path) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<Dimap_Document>
  <RasterDataNode>
    <Coordinate_Reference_System>
      <NAME>EPSG:4326</NAME>
    </Coordinate_Reference_System>
    <Geoposition>
      <ULXMAP>0.0</ULXMAP>
      <ULYMAP>0.0</ULYMAP>
      <XDIM>10.0</XDIM>
      <YDIM>-10.0</YDIM>
    </Geoposition>
    <Raster_Dimensions>
      <NCOLS>4</NCOLS>
      <NROWS>4</NROWS>
      <NBANDS>4</NBANDS>
    </Raster_Dimensions>
  </RasterDataNode>
  <Image_Interpretation>
    <Spectral_Band_Info>
      <BAND_INDEX>0</BAND_INDEX>
      <BAND_NAME>i_IW1_VV</BAND_NAME>
      <BAND_DESCRIPTION>Real part (VV)</BAND_DESCRIPTION>
      <BAND_UNIT>real</BAND_UNIT>
    </Spectral_Band_Info>
    <Spectral_Band_Info>
      <BAND_INDEX>1</BAND_INDEX>
      <BAND_NAME>q_IW1_VV</BAND_NAME>
      <BAND_DESCRIPTION>Imag part (VV)</BAND_DESCRIPTION>
      <BAND_UNIT>imag</BAND_UNIT>
    </Spectral_Band_Info>
    <Spectral_Band_Info>
      <BAND_INDEX>2</BAND_INDEX>
      <BAND_NAME>Intensity_IW1_VV</BAND_NAME>
      <BAND_DESCRIPTION>Intensity (VV)</BAND_DESCRIPTION>
      <BAND_UNIT>intensity</BAND_UNIT>
      <EXPRESSION>i_IW1_VV == 0.0 ? 0.0 : i_IW1_VV * i_IW1_VV + q_IW1_VV * q_IW1_VV</EXPRESSION>
    </Spectral_Band_Info>
    <Spectral_Band_Info>
      <BAND_INDEX>3</BAND_INDEX>
      <BAND_NAME>derampDemodPhase</BAND_NAME>
      <BAND_DESCRIPTION>Auxiliary</BAND_DESCRIPTION>
      <BAND_UNIT>radian</BAND_UNIT>
    </Spectral_Band_Info>
  </Image_Interpretation>
  <Data_Access>
    <Data_File>
      <BAND_INDEX>0</BAND_INDEX>
      <DATA_FILE_PATH href="./case.data/i_IW1_VV.hdr" />
    </Data_File>
  </Data_Access>
  <MDElem name="Abstracted_Metadata">
    <MDElem name="Band_IW1_VV">
      <MDATTR name="polarization">VV</MDATTR>
      <MDATTR name="band_names">i_IW1_VV,q_IW1_VV</MDATTR>
    </MDElem>
  </MDElem>
</Dimap_Document>
"""
    dim_path.write_text(xml, encoding="utf-8")


def _spectral_band_map(root: ET.Element) -> dict[str, ET.Element]:
    return {
        (sbi.findtext("BAND_NAME") or "").strip(): sbi
        for sbi in root.findall(".//Image_Interpretation/Spectral_Band_Info")
    }


def test_gpt_do_subaps_forwards_update_dim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    product = _touch(tmp_path / "input.dim")
    safe = _touch(tmp_path / "input.SAFE")
    captured: dict[str, object] = {}

    def fake_do_subaps(**kwargs):
        captured.update(kwargs)

    fake_sarpyx = types.ModuleType("sarpyx")
    fake_processor = types.ModuleType("sarpyx.processor")
    fake_core = types.ModuleType("sarpyx.processor.core")
    fake_subap = types.ModuleType("sarpyx.processor.core.subaperture_full_img")
    fake_subap.do_subaps = fake_do_subaps
    fake_core.subaperture_full_img = fake_subap
    fake_processor.core = fake_core
    fake_sarpyx.processor = fake_processor

    monkeypatch.setitem(sys.modules, "sarpyx", fake_sarpyx)
    monkeypatch.setitem(sys.modules, "sarpyx.processor", fake_processor)
    monkeypatch.setitem(sys.modules, "sarpyx.processor.core", fake_core)
    monkeypatch.setitem(sys.modules, "sarpyx.processor.core.subaperture_full_img", fake_subap)

    gpt = GPT(product=product, outdir=tmp_path / "out", gpt_path="/tmp/fake-gpt")
    result = gpt.do_subaps(dim_path=product, safe_path=safe, update_dim=False, chunk_cols=128)

    assert result == product
    assert captured["dim_path"] == str(product)
    assert captured["safe_path"] == str(safe)
    assert captured["update_dim"] is False
    assert captured["chunk_cols"] == 128


def test_sentinel_post_chain_fails_fast_before_merge_and_tc(tmp_path: Path):
    merge_called = False

    class FakeOp:
        def __init__(self):
            self.prod_path = tmp_path / "initial.dim"
            self.do_subaps_kwargs: dict[str, object] | None = None
            self.tc_called = False

        def ApplyOrbitFile(self):
            return str(tmp_path / "orb.dim")

        def Calibration(self, **_kwargs):
            return str(tmp_path / "cal.dim")

        def TopsarDerampDemod(self):
            return str(tmp_path / "deramp.dim")

        def Deburst(self):
            self.prod_path = tmp_path / "deb.dim"
            return str(self.prod_path)

        def do_subaps(self, **kwargs):
            self.do_subaps_kwargs = kwargs

        def polarimetric_decomposition(self, **_kwargs):
            return None

        def TerrainCorrection(self, **_kwargs):
            self.tc_called = True
            return str(tmp_path / "tc.dim")

        def last_error_summary(self):
            return "forced pdec failure"

    def fake_merge(**_kwargs):
        nonlocal merge_called
        merge_called = True

    _sentinel_post_chain.__globals__["merge_iq_into_pdec"] = fake_merge

    op = FakeOp()
    with pytest.raises(RuntimeError, match="Polarimetric decomposition failed: forced pdec failure"):
        _sentinel_post_chain(op=op, product_path=str(tmp_path / "input.SAFE"))

    assert op.do_subaps_kwargs is not None
    assert op.do_subaps_kwargs["update_dim"] is False
    assert merge_called is False
    assert op.tc_called is False


def test_sentinel_post_chain_fails_fast_on_deburst_before_subaps(tmp_path: Path):
    merge_called = False

    class FakeOp:
        def __init__(self):
            self.prod_path = tmp_path / "initial.dim"
            self.do_subaps_called = False
            self.tc_called = False

        def ApplyOrbitFile(self):
            self.prod_path = tmp_path / "orb.dim"
            return str(self.prod_path)

        def Calibration(self, **_kwargs):
            self.prod_path = tmp_path / "cal.dim"
            return str(self.prod_path)

        def TopsarDerampDemod(self):
            self.prod_path = tmp_path / "deramp.dim"
            return str(self.prod_path)

        def Deburst(self):
            return None

        def do_subaps(self, **_kwargs):
            self.do_subaps_called = True

        def polarimetric_decomposition(self, **_kwargs):
            return str(tmp_path / "pdec.dim")

        def TerrainCorrection(self, **_kwargs):
            self.tc_called = True
            return str(tmp_path / "tc.dim")

        def last_error_summary(self):
            return "forced deburst failure"

    def fake_merge(**_kwargs):
        nonlocal merge_called
        merge_called = True

    _sentinel_post_chain.__globals__["merge_iq_into_pdec"] = fake_merge

    op = FakeOp()
    with pytest.raises(RuntimeError, match="TOPSAR-Deburst failed: forced deburst failure"):
        _sentinel_post_chain(op=op, product_path=str(tmp_path / "input.SAFE"))

    assert op.do_subaps_called is False
    assert merge_called is False
    assert op.tc_called is False


def test_pipeline_sentinel_strip_uses_update_dim_false_and_real_pdec_path(
    tmp_path: Path,
):
    merge_calls: list[dict[str, object]] = []

    class FakeOp:
        def __init__(self):
            self.prod_path = tmp_path / "input.dim"
            self.do_subaps_kwargs: dict[str, object] | None = None

        def ApplyOrbitFile(self):
            self.prod_path = tmp_path / "orb.dim"
            return str(self.prod_path)

        def Calibration(self, **_kwargs):
            self.prod_path = tmp_path / "cal.dim"
            return str(self.prod_path)

        def do_subaps(self, **kwargs):
            self.do_subaps_kwargs = kwargs

        def polarimetric_decomposition(self, **_kwargs):
            self.prod_path = tmp_path / "pdec.dim"
            return str(self.prod_path)

        def TerrainCorrection(self, **_kwargs):
            self.prod_path = tmp_path / "tc.dim"
            return str(self.prod_path)

        def last_error_summary(self):
            return "unused"

    fake_op = FakeOp()

    def fake_create_gpt_operator(*_args, **_kwargs):
        return fake_op

    def fake_merge(**kwargs):
        merge_calls.append(kwargs)

    pipeline_sentinel.__globals__["_create_gpt_operator"] = fake_create_gpt_operator
    pipeline_sentinel.__globals__["merge_iq_into_pdec"] = fake_merge
    pipeline_sentinel.__globals__["_sentinel_post_chain"] = _sentinel_post_chain

    result = pipeline_sentinel(
        product_path=str(tmp_path / "input.SAFE"),
        output_dir=tmp_path / "out",
        is_TOPS=False,
    )

    assert Path(result) == tmp_path / "tc.dim"
    assert fake_op.do_subaps_kwargs is not None
    assert fake_op.do_subaps_kwargs["update_dim"] is False
    assert len(merge_calls) == 1
    assert Path(merge_calls[0]["src_dim"]) == tmp_path / "cal.dim"
    assert Path(merge_calls[0]["pdec_dim"]) == tmp_path / "pdec.dim"
    assert merge_calls[0]["is_tops"] is False
    assert merge_calls[0]["overwrite_copied_files"] is False
    assert merge_calls[0]["backup"] is False


def test_merge_iq_into_pdec_rejects_same_dim_product(tmp_path: Path):
    dim_path = _touch(tmp_path / "same.dim")

    with pytest.raises(ValueError, match="resolve to the same DIM product"):
        merge_iq_into_pdec(
            src_dim=str(dim_path),
            pdec_dim=str(dim_path),
            is_tops=True,
        )


def test_update_dim_add_bands_from_data_dir_backfills_geocoding_and_sa_expressions(tmp_path: Path):
    dim_path = tmp_path / "case.dim"
    data_dir = tmp_path / "case.data"
    data_dir.mkdir()

    _write_minimal_dim(dim_path)
    for band_name in (
        "i_IW1_VV",
        "q_IW1_VV",
        "i_IW1_VV_SA1",
        "q_IW1_VV_SA1",
        "i_IW1_VV_SA2",
        "q_IW1_VV_SA2",
    ):
        _write_envi_stub(data_dir, band_name)

    update_dim_add_bands_from_data_dir(str(dim_path), verbose=False)

    root = ET.parse(dim_path).getroot()
    sbis = _spectral_band_map(root)
    geo_band_indices = {
        int(geo.findtext("BAND_INDEX"))
        for geo in root.findall(".//Geoposition")
        if geo.findtext("BAND_INDEX") is not None
    }
    sbi_band_indices = {
        int(sbi.findtext("BAND_INDEX"))
        for sbi in root.findall(".//Image_Interpretation/Spectral_Band_Info")
    }

    assert "Intensity_IW1_VV_SA1" in sbis
    assert "Intensity_IW1_VV_SA2" in sbis
    assert sbis["Intensity_IW1_VV_SA1"].findtext("EXPRESSION") == (
        "i_IW1_VV_SA1 == 0.0 ? 0.0 : i_IW1_VV_SA1 * i_IW1_VV_SA1 + q_IW1_VV_SA1 * q_IW1_VV_SA1"
    )
    assert sbis["Intensity_IW1_VV_SA2"].findtext("EXPRESSION") == (
        "i_IW1_VV_SA2 == 0.0 ? 0.0 : i_IW1_VV_SA2 * i_IW1_VV_SA2 + q_IW1_VV_SA2 * q_IW1_VV_SA2"
    )
    assert sbi_band_indices.issubset(geo_band_indices)


def test_sentinel1_swath_wkt_extractor_safe_reads_annotation_geolocation_points(tmp_path: Path):
    safe_dir = tmp_path / "product.SAFE"
    _write_safe_annotation(
        safe_dir,
        "s1-test-iw1-vv-001.xml",
        [
            (12.0, 46.0),
            (12.4, 46.0),
            (12.4, 45.6),
            (12.0, 45.6),
        ],
    )

    polygon = shapely_wkt.loads(
        sentinel1_swath_wkt_extractor_safe(safe_dir, "IW1")
    )

    assert polygon.bounds == pytest.approx((12.0, 45.6, 12.4, 46.0))


def test_resolve_tiling_wkt_prefers_swath_dim_footprint_and_falls_back(tmp_path: Path):
    safe_dir = tmp_path / "product.SAFE"
    swath_product = tmp_path / "iw1.dim"
    swath_product.write_text("placeholder", encoding="utf-8")

    _resolve_tiling_wkt.__globals__["sentinel1_swath_wkt_extractor_safe"] = lambda *_args, **_kwargs: "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))"
    assert _resolve_tiling_wkt("FULL_WKT", safe_dir, swath_product, "S1TOPS", swath="IW1") == "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))"

    _resolve_tiling_wkt.__globals__["sentinel1_swath_wkt_extractor_safe"] = lambda *_args, **_kwargs: None
    assert _resolve_tiling_wkt("FULL_WKT", safe_dir, swath_product, "S1TOPS", swath="IW1") == "FULL_WKT"
    assert _resolve_tiling_wkt("FULL_WKT", safe_dir, swath_product, "S1TOPS") == "FULL_WKT"
    assert _resolve_tiling_wkt("FULL_WKT", safe_dir, swath_product, "S1STRIP", swath="IW1") == "FULL_WKT"


def test_run_tops_swath_tiling_uses_swath_specific_wkt(tmp_path: Path):
    calls: list[tuple[str, Path]] = []
    db_calls: list[tuple[Path, str]] = []
    verify_calls: list[dict[str, object]] = []
    swath_products = {
        "IW1": _touch(tmp_path / "IW1.dim"),
        "IW2": _touch(tmp_path / "IW2.dim"),
    }

    def fake_resolve(full_wkt, source_product, swath_product, product_mode, swath=None):
        assert Path(source_product) == tmp_path / "source.SAFE"
        assert product_mode == "S1TOPS"
        assert swath in {"IW1", "IW2"}
        return f"WKT::{Path(swath_product).stem}"

    def fake_run_tiling(product_wkt, _grid_path, _source_product, intermediate_product, _cuts_outdir, _product_mode, **_kwargs):
        calls.append((product_wkt, Path(intermediate_product)))
        return Path(intermediate_product).stem

    def fake_run_db_indexing(cuts_outdir, name):
        db_calls.append((Path(cuts_outdir), name))

    def fake_verify(product_wkt, _grid_path, cuts_outdir, intermediate, swath_wkts=None):
        verify_calls.append(
            {
                "product_wkt": product_wkt,
                "cuts_outdir": Path(cuts_outdir),
                "intermediate_keys": sorted(intermediate),
                "swath_wkts": dict(swath_wkts or {}),
            }
        )

    _run_tops_swath_tiling.__globals__["tiling"] = True
    _run_tops_swath_tiling.__globals__["extract_product_id"] = lambda path: Path(path).stem
    _run_tops_swath_tiling.__globals__["_resolve_tiling_wkt"] = fake_resolve
    _run_tops_swath_tiling.__globals__["_run_tiling"] = fake_run_tiling
    _run_tops_swath_tiling.__globals__["_run_db_indexing"] = fake_run_db_indexing
    _run_tops_swath_tiling.__globals__["_verify_tops_tile_coverage"] = fake_verify

    _run_tops_swath_tiling(
        product_wkt="FULL_WKT",
        grid_geoj_path=tmp_path / "grid.geojson",
        product_path=tmp_path / "source.SAFE",
        intermediate=swath_products,
        cuts_outdir=tmp_path / "cuts",
        product_mode="S1TOPS",
        gpt_kwargs={"gpt_memory": None, "gpt_parallelism": None, "gpt_timeout": None},
    )

    assert calls == [
        ("WKT::IW1", swath_products["IW1"]),
        ("WKT::IW2", swath_products["IW2"]),
    ]
    assert db_calls == [
        (tmp_path / "cuts" / "IW1", "IW1"),
        (tmp_path / "cuts" / "IW2", "IW2"),
    ]
    assert verify_calls == []
