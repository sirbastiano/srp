import importlib.util
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from sarpyx.cli import worldsar
from sarpyx.utils.wkt_utils import terrasar_wkt_extractor


def _write_tsx_xml(path: Path, lon_offset: float = 0.0) -> Path:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <sceneCornerCoord><lon>{13.0 + lon_offset}</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>{13.1 + lon_offset}</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>{13.1 + lon_offset}</lon><lat>52.1</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>{13.0 + lon_offset}</lon><lat>52.1</lat></sceneCornerCoord>
</product>
""",
        encoding="utf-8",
    )
    return path


def _write_tsx_eec_xml(path: Path) -> Path:
    return _write_tsx_product_xml(
        path,
        product_type="EEC_SE_HS_S",
        product_variant="EEC",
        image_data_type="DETECTED",
        projection="MAP",
        map_projection="UTM",
        imaging_mode="HS",
    )


def _write_tsx_product_xml(
    path: Path,
    *,
    product_type: str,
    product_variant: str | None,
    image_data_type: str,
    projection: str = "SLANTRANGE",
    map_projection: str | None = None,
    imaging_mode: str = "SM",
) -> Path:
    product_variant_xml = (
        f"      <productVariant>{product_variant}</productVariant>\n"
        if product_variant is not None
        else ""
    )
    map_projection_xml = f"      <mapProjection>{map_projection}</mapProjection>\n" if map_projection else ""
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<level1Product>
  <productInfo>
    <acquisitionInfo>
      <imagingMode>{imaging_mode}</imagingMode>
    </acquisitionInfo>
    <productVariantInfo>
      <productType>{product_type}</productType>
{product_variant_xml}      <projection>{projection}</projection>
{map_projection_xml}      <radiometricCorrection>CALIBRATED</radiometricCorrection>
    </productVariantInfo>
    <imageDataInfo>
      <imageDataType>{image_data_type}</imageDataType>
    </imageDataInfo>
  </productInfo>
  <sceneCornerCoord><lon>13.0</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>13.1</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>13.1</lon><lat>52.1</lat></sceneCornerCoord>
</level1Product>
""",
        encoding="utf-8",
    )
    return path


def test_resolve_terrasar_directory_to_scene_corner_xml(tmp_path: Path) -> None:
    product_dir = tmp_path / "TDX1_SAR__MGD_RE___SL_S_SRA_TEST"
    product_dir.mkdir()
    (product_dir / "notes.xml").write_text("<product />", encoding="utf-8")
    metadata_xml = _write_tsx_xml(product_dir / "metadata.xml")

    assert worldsar._resolve_terrasar_product_xml(product_dir) == metadata_xml
    assert worldsar._resolve_terrasar_product_xml(metadata_xml) == metadata_xml


def test_resolve_terrasar_directory_rejects_ambiguous_xml(tmp_path: Path) -> None:
    product_dir = tmp_path / "TSX_PRODUCT"
    product_dir.mkdir()
    _write_tsx_xml(product_dir / "metadata-a.xml")
    _write_tsx_xml(product_dir / "metadata-b.xml", lon_offset=1.0)

    with pytest.raises(ValueError, match="Multiple TerraSAR-X/TanDEM-X metadata XML"):
        worldsar._resolve_terrasar_product_xml(product_dir)


def test_resolve_terrasar_zip_extracts_and_resolves_xml(tmp_path: Path) -> None:
    archive_path = tmp_path / "TSX_PRODUCT.ZIP"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("product/notes.xml", "<product />")
        archive.writestr(
            "product/metadata.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<product>
  <sceneCornerCoord><lon>13.0</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>13.1</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>13.1</lon><lat>52.1</lat></sceneCornerCoord>
</product>
""",
        )

    resolved = worldsar._resolve_terrasar_product_xml(archive_path)

    assert resolved.name == "metadata.xml"
    assert resolved.exists()
    assert ".sarpyx_extracted" in resolved.parts


def test_resolve_terrasar_sip_zip_extracts_nested_product_zip(tmp_path: Path) -> None:
    inner_zip_path = tmp_path / "TSX_INNER.ZIP"
    with zipfile.ZipFile(inner_zip_path, "w") as archive:
        archive.writestr("product/notes.xml", "<product />")
        archive.writestr(
            "product/TSX1_SAR_PRODUCT.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<product>
  <sceneCornerCoord><lon>13.0</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>13.1</lon><lat>52.0</lat></sceneCornerCoord>
  <sceneCornerCoord><lon>13.1</lon><lat>52.1</lat></sceneCornerCoord>
</product>
""",
        )

    sip_zip_path = tmp_path / "TSX_OPER_SAR_TEST.SIP.ZIP"
    with zipfile.ZipFile(sip_zip_path, "w") as archive:
        archive.writestr("TSX_OPER_SAR_TEST.MD.XML", "<metadata />")
        archive.write(inner_zip_path, arcname="TSX_OPER_SAR_TEST.ZIP")

    resolved = worldsar._resolve_terrasar_product_xml(sip_zip_path)

    assert resolved.name == "TSX1_SAR_PRODUCT.xml"
    assert resolved.exists()
    assert "TSX_OPER_SAR_TEST" in resolved.as_posix()


def test_terrasar_wkt_extractor_reads_scene_corners(tmp_path: Path) -> None:
    metadata_xml = _write_tsx_xml(tmp_path / "metadata.xml")

    assert terrasar_wkt_extractor(metadata_xml) == (
        "POLYGON((13.0 52.0, 13.1 52.0, 13.1 52.1, 13.0 52.1, 13.0 52.0))"
    )


def test_resolve_product_wkt_uses_product_wkt_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PRODUCT_WKT", " POLYGON((1 1, 2 1, 2 2, 1 1)) ")
    args = Namespace(product_wkt=None)

    assert worldsar._resolve_product_wkt(args, tmp_path / "CSG_PRODUCT", "CSG") == (
        "POLYGON((1 1, 2 1, 2 2, 1 1))"
    )


def test_resolve_product_wkt_auto_extracts_tsx_directory(tmp_path: Path) -> None:
    product_dir = tmp_path / "TDX1_SAR__MGD_RE___SL_S_SRA_TEST"
    product_dir.mkdir()
    _write_tsx_xml(product_dir / "metadata.xml")
    args = Namespace(product_wkt=None)

    assert worldsar._resolve_product_wkt(args, product_dir, "TSX") == (
        "POLYGON((13.0 52.0, 13.1 52.0, 13.1 52.1, 13.0 52.1, 13.0 52.0))"
    )


def test_infer_product_mode_uses_terrasar_metadata_for_neutral_directory(tmp_path: Path) -> None:
    product_dir = tmp_path / "neutral_product_name"
    product_dir.mkdir()
    _write_tsx_product_xml(
        product_dir / "metadata.xml",
        product_type="SSC_RE_SM_S",
        product_variant="SSC",
        image_data_type="COMPLEX",
        imaging_mode="SM",
    )

    assert worldsar.infer_product_mode(product_dir) == "TSX"


def test_pipeline_tsx_csg_passes_resolved_xml_to_gpt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    product_dir = tmp_path / "TDX1_SAR__MGD_RE___SL_S_SRA_TEST"
    product_dir.mkdir()
    metadata_xml = _write_tsx_xml(product_dir / "metadata.xml")
    captured = {}

    class FakeOp:
        prod_path = tmp_path / "TDX1_CAL_TC.dim"

        def Calibration(self, output_complex: bool = False):
            captured["calibration_output_complex"] = output_complex
            return tmp_path / "TDX1_CAL.dim"

        def TerrainCorrection(self, **kwargs):
            captured["terrain_correction_kwargs"] = kwargs
            return self.prod_path

    def fake_create_gpt_operator(product_path, output_dir, output_format, *args, **kwargs):
        captured["product_path"] = Path(product_path)
        captured["output_dir"] = Path(output_dir)
        captured["output_format"] = output_format
        return FakeOp()

    monkeypatch.setattr(worldsar, "_create_gpt_operator", fake_create_gpt_operator)

    assert worldsar.pipeline_tsx_csg(product_dir, tmp_path / "out") == FakeOp.prod_path
    assert captured["product_path"] == metadata_xml
    assert captured["output_format"] == "BEAM-DIMAP"
    assert captured["calibration_output_complex"] is True
    assert captured["terrain_correction_kwargs"]["pixel_spacing_in_meter"] == 5.0


def test_pipeline_tsx_eec_uses_write_without_calibration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    product_dir = tmp_path / "TSX_PRODUCT"
    product_dir.mkdir()
    metadata_xml = _write_tsx_eec_xml(product_dir / "metadata.xml")
    captured = {}

    class FakeOp:
        prod_path = tmp_path / "TSX_WRITE.dim"

        def write(self, **kwargs):
            captured["write_kwargs"] = kwargs
            return self.prod_path

        def Calibration(self, output_complex: bool = False):
            raise AssertionError("EEC products should not run Calibration")

        def TerrainCorrection(self, **kwargs):
            raise AssertionError("EEC products should not run TerrainCorrection")

    def fake_create_gpt_operator(product_path, output_dir, output_format, *args, **kwargs):
        captured["product_path"] = Path(product_path)
        captured["output_dir"] = Path(output_dir)
        captured["output_format"] = output_format
        return FakeOp()

    monkeypatch.setattr(worldsar, "_create_gpt_operator", fake_create_gpt_operator)

    assert worldsar.pipeline_tsx_csg(product_dir, tmp_path / "out") == FakeOp.prod_path
    assert captured["product_path"] == metadata_xml
    assert captured["output_format"] == "BEAM-DIMAP"
    assert captured["write_kwargs"]["format_name"] == "BEAM-DIMAP"
    assert captured["write_kwargs"]["output_file"] == tmp_path / "out" / "metadata_WRITE.dim"


def test_pipeline_tsx_geocoded_variant_from_product_type_uses_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    product_dir = tmp_path / "TSX_PRODUCT"
    product_dir.mkdir()
    metadata_xml = _write_tsx_product_xml(
        product_dir / "metadata.xml",
        product_type="GEC_RE_SM_D",
        product_variant=None,
        image_data_type="DETECTED",
        projection="MAP",
        map_projection="UTM",
        imaging_mode="SM",
    )
    captured = {}

    class FakeOp:
        prod_path = tmp_path / "TSX_WRITE.dim"

        def write(self, **kwargs):
            captured["write_kwargs"] = kwargs
            return self.prod_path

        def Calibration(self, output_complex: bool = False):
            raise AssertionError("geocoded products should not run Calibration")

        def TerrainCorrection(self, **kwargs):
            raise AssertionError("geocoded products should not run TerrainCorrection")

    def fake_create_gpt_operator(product_path, output_dir, output_format, *args, **kwargs):
        captured["product_path"] = Path(product_path)
        return FakeOp()

    monkeypatch.setattr(worldsar, "_create_gpt_operator", fake_create_gpt_operator)

    assert worldsar._terrasar_product_variant(metadata_xml) == "GEC"
    assert worldsar.pipeline_tsx_csg(product_dir, tmp_path / "out") == FakeOp.prod_path
    assert captured["product_path"] == metadata_xml
    assert captured["write_kwargs"]["format_name"] == "BEAM-DIMAP"


@pytest.mark.parametrize(
    ("variant", "product_type", "image_data_type", "imaging_mode", "expected_complex"),
    [
        ("SSC", "SSC_RE_ST_S", "COMPLEX", "ST", True),
        ("MGD", "MGD_SE_SL_D", "DETECTED", "SL", False),
    ],
)
def test_pipeline_tsx_variant_controls_calibration_complexity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variant: str,
    product_type: str,
    image_data_type: str,
    imaging_mode: str,
    expected_complex: bool,
) -> None:
    product_dir = tmp_path / "TSX_PRODUCT"
    product_dir.mkdir()
    metadata_xml = _write_tsx_product_xml(
        product_dir / "metadata.xml",
        product_type=product_type,
        product_variant=variant,
        image_data_type=image_data_type,
        imaging_mode=imaging_mode,
    )
    captured = {}

    class FakeOp:
        prod_path = tmp_path / "TSX_TC.dim"

        def Calibration(self, output_complex: bool = False):
            captured["calibration_output_complex"] = output_complex
            return tmp_path / "TSX_CAL.dim"

        def TerrainCorrection(self, **kwargs):
            captured["terrain_correction_kwargs"] = kwargs
            return self.prod_path

    def fake_create_gpt_operator(product_path, output_dir, output_format, *args, **kwargs):
        captured["product_path"] = Path(product_path)
        return FakeOp()

    monkeypatch.setattr(worldsar, "_create_gpt_operator", fake_create_gpt_operator)

    assert worldsar.pipeline_tsx_csg(product_dir, tmp_path / "out") == FakeOp.prod_path
    assert captured["product_path"] == metadata_xml
    assert captured["calibration_output_complex"] is expected_complex
    assert captured["terrain_correction_kwargs"]["output_complex"] is expected_complex
    assert captured["terrain_correction_kwargs"]["pixel_spacing_in_meter"] == 5.0


def test_pyscript_worldsar_parser_uses_shared_none_defaults() -> None:
    module_path = Path(__file__).resolve().parents[1] / "pyscripts" / "worldsar.py"
    spec = importlib.util.spec_from_file_location("worldsar_pyscript_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.create_parser().parse_args(["--input", "/tmp/product"])

    assert args.product_path == "/tmp/product"
    assert args.output_dir is None
    assert args.cuts_outdir is None
    assert args.gpt_path is None
    assert args.grid_path is None
    assert args.db_dir is None
    assert args.gpt_memory is None
    assert args.gpt_parallelism is None
    assert args.gpt_timeout is None
