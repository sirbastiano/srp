from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from sarpyx.cli.merge_iq_into_pdec import merge_iq_into_pdec


def _write_minimal_dim(path: Path, ncols: int = 4, nrows: int = 3) -> None:
    path.write_text(
        f"""<Dimap_Document>
    <Raster_Dimensions>
        <NCOLS>{ncols}</NCOLS>
        <NROWS>{nrows}</NROWS>
        <NBANDS>1</NBANDS>
    </Raster_Dimensions>
    <Data_Access />
    <Coordinate_Reference_System>
        <WKT>LOCAL_CS["test"]</WKT>
    </Coordinate_Reference_System>
    <Geoposition>
        <LATITUDE_BAND>lat</LATITUDE_BAND>
        <LONGITUDE_BAND>lon</LONGITUDE_BAND>
    </Geoposition>
    <Image_Interpretation>
        <Spectral_Band_Info>
            <BAND_INDEX>0</BAND_INDEX>
            <BAND_NAME>Entropy</BAND_NAME>
        </Spectral_Band_Info>
    </Image_Interpretation>
</Dimap_Document>
""",
        encoding="utf-8",
    )


def _write_iq_pair(data_dir: Path, suffix: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ("i", "q"):
        (data_dir / f"{prefix}_{suffix}.hdr").write_text("ENVI\n", encoding="ascii")
        (data_dir / f"{prefix}_{suffix}.img").write_bytes(b"\0\0\0\0")


def test_merge_iq_into_pdec_rejects_same_dim_product(tmp_path: Path) -> None:
    dim_path = tmp_path / "same.dim"
    _write_minimal_dim(dim_path)

    with pytest.raises(ValueError, match="resolve to the same DIM product"):
        merge_iq_into_pdec(dim_path, dim_path)


def test_merge_iq_into_pdec_copies_iq_files_and_updates_pdec_dim(tmp_path: Path) -> None:
    src_dim = tmp_path / "src.dim"
    pdec_dim = tmp_path / "pdec.dim"
    src_data = tmp_path / "src.data"
    pdec_data = tmp_path / "pdec.data"

    _write_minimal_dim(src_dim)
    _write_minimal_dim(pdec_dim)
    _write_iq_pair(src_data, "VV")
    pdec_data.mkdir()

    merge_iq_into_pdec(src_dim, pdec_dim, backup=False)

    assert (pdec_data / "i_VV.hdr").exists()
    assert (pdec_data / "i_VV.img").exists()
    assert (pdec_data / "q_VV.hdr").exists()
    assert (pdec_data / "q_VV.img").exists()

    root = ET.parse(pdec_dim).getroot()
    raster_dimensions = root.find("Raster_Dimensions")
    assert raster_dimensions is not None
    assert raster_dimensions.findtext("NBANDS") == "4"

    band_names = [
        band.findtext("BAND_NAME")
        for band in root.findall("./Image_Interpretation/Spectral_Band_Info")
    ]
    assert band_names == ["Entropy", "i_VV", "q_VV", "Intensity_VV"]

    data_file_hrefs = [
        data_file.find("DATA_FILE_PATH").get("href")
        for data_file in root.findall("./Data_Access/Data_File")
    ]
    assert data_file_hrefs == ["pdec.data/i_VV.hdr", "pdec.data/q_VV.hdr"]

    intensity_band = root.findall("./Image_Interpretation/Spectral_Band_Info")[-1]
    assert intensity_band.findtext("VIRTUAL_BAND") == "true"
    assert intensity_band.findtext("EXPRESSION") == "i_VV == 0.0 ? 0.0 : i_VV * i_VV + q_VV * q_VV"
