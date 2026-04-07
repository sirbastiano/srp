#!/usr/bin/env python3


from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def find_required(parent: ET.Element, tag: str) -> ET.Element:
    element = parent.find(tag)
    if element is None:
        raise RuntimeError(f"Missing <{tag}> inside <{parent.tag}>.")
    return element


def text_required(parent: ET.Element, tag: str) -> str:
    element = find_required(parent, tag)
    if element.text is None:
        raise RuntimeError(f"Tag <{tag}> inside <{parent.tag}> has no text.")
    return element.text.strip()


def get_data_dir_from_dim(dim_path: Path) -> Path:
    return dim_path.with_suffix("").with_name(dim_path.stem + ".data")


def already_has_band(image_interpretation: ET.Element, band_name: str) -> bool:
    for spectral_band in image_interpretation.findall("Spectral_Band_Info"):
        existing_name = spectral_band.findtext("BAND_NAME", default="").strip()
        if existing_name == band_name:
            return True
    return False


def detect_suffixes_from_src_data(src_data_dir: Path) -> list[str]:
    i_suffixes = set()
    q_suffixes = set()

    for hdr_path in src_data_dir.glob("*.hdr"):
        stem = hdr_path.stem

        if stem.startswith("i_"):
            suffix = stem[2:]
            if (src_data_dir / f"i_{suffix}.img").exists():
                i_suffixes.add(suffix)

        elif stem.startswith("q_"):
            suffix = stem[2:]
            if (src_data_dir / f"q_{suffix}.img").exists():
                q_suffixes.add(suffix)

    return sorted(i_suffixes & q_suffixes)


def build_band_plan(suffixes: list[str], start_index: int) -> list[dict]:
    bands: list[dict] = []
    current_index = start_index

    for suffix in suffixes:
        i_name = f"i_{suffix}"
        q_name = f"q_{suffix}"
        intensity_name = f"Intensity_{suffix}"

        bands.append(
            {
                "band_index": current_index,
                "band_name": i_name,
                "physical_unit": "real",
                "virtual": False,
                "expr": None,
                "file_name": f"{i_name}.hdr",
            }
        )
        current_index += 1

        bands.append(
            {
                "band_index": current_index,
                "band_name": q_name,
                "physical_unit": "imaginary",
                "virtual": False,
                "expr": None,
                "file_name": f"{q_name}.hdr",
            }
        )
        current_index += 1

        bands.append(
            {
                "band_index": current_index,
                "band_name": intensity_name,
                "physical_unit": "intensity",
                "virtual": True,
                "expr": f"{i_name} == 0.0 ? 0.0 : {i_name} * {i_name} + {q_name} * {q_name}",
                "file_name": None,
            }
        )
        current_index += 1

    return bands


def build_data_file(href: str, band_index: int) -> ET.Element:
    data_file = ET.Element("Data_File")
    data_file_path = ET.SubElement(data_file, "DATA_FILE_PATH")
    data_file_path.set("href", href)
    band_index_el = ET.SubElement(data_file, "BAND_INDEX")
    band_index_el.text = str(band_index)
    return data_file


def build_spectral_band_info(
    band_index: int,
    band_name: str,
    width: str,
    height: str,
    physical_unit: str,
    virtual: bool = False,
    expr: str | None = None,
) -> ET.Element:
    spectral_band = ET.Element("Spectral_Band_Info")

    def add(tag: str, text: str | None = "") -> ET.Element:
        child = ET.SubElement(spectral_band, tag)
        if text is not None:
            child.text = text
        return child

    add("BAND_INDEX", str(band_index))
    add("BAND_DESCRIPTION", "Intensity from complex data" if band_name.startswith("Intensity_") else None)
    add("BAND_NAME", band_name)
    add("BAND_RASTER_WIDTH", width)
    add("BAND_RASTER_HEIGHT", height)
    add("DATA_TYPE", "float32")
    add("PHYSICAL_UNIT", physical_unit)
    add("SOLAR_FLUX", "0.0")
    add("BAND_WAVELEN", "0.0")
    add("BAND_ANGULAR_VALUE", "-999.0")
    add("BANDWIDTH", "0.0")
    add("SCALING_FACTOR", "1.0")
    add("SCALING_OFFSET", "0.0")
    add("LOG10_SCALED", "false")
    add("NO_DATA_VALUE_USED", "true")
    add("NO_DATA_VALUE", "0.0")

    if virtual:
        add("VIRTUAL_BAND", "true")
        add("EXPRESSION", expr if expr is not None else "")

    return spectral_band


def clone_crs_geoposition_pair(
    template_crs: ET.Element,
    template_geoposition: ET.Element,
    band_index: int,
) -> list[ET.Element]:
    crs_copy = copy.deepcopy(template_crs)
    geoposition_copy = copy.deepcopy(template_geoposition)

    band_index_el = geoposition_copy.find("BAND_INDEX")
    if band_index_el is None:
        band_index_el = ET.Element("BAND_INDEX")
        geoposition_copy.insert(0, band_index_el)

    band_index_el.text = str(band_index)
    return [crs_copy, geoposition_copy]


def copy_src_files(
    src_data_dir: Path,
    pdec_data_dir: Path,
    suffixes: list[str],
    overwrite: bool = False,
) -> None:
    pdec_data_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy: list[str] = []
    for suffix in suffixes:
        files_to_copy.extend(
            [
                f"i_{suffix}.hdr",
                f"i_{suffix}.img",
                f"q_{suffix}.hdr",
                f"q_{suffix}.img",
            ]
        )

    for file_name in files_to_copy:
        src = src_data_dir / file_name
        dst = pdec_data_dir / file_name

        if not src.exists():
            raise FileNotFoundError(f"Required source file does not exist: {src}")

        if dst.exists() and not overwrite:
            print(f"[skip] File already exists: {dst.name}")
            continue

        shutil.copy2(src, dst)
        print(f"[copy] {src.name} -> {dst}")


def validate_same_dimensions(src_dim: Path, pdec_dim: Path) -> None:
    src_root = ET.parse(src_dim).getroot()
    pdec_root = ET.parse(pdec_dim).getroot()

    src_raster_dimensions = find_required(src_root, "Raster_Dimensions")
    pdec_raster_dimensions = find_required(pdec_root, "Raster_Dimensions")

    src_ncols = text_required(src_raster_dimensions, "NCOLS")
    src_nrows = text_required(src_raster_dimensions, "NROWS")
    pdec_ncols = text_required(pdec_raster_dimensions, "NCOLS")
    pdec_nrows = text_required(pdec_raster_dimensions, "NROWS")

    if src_ncols != pdec_ncols or src_nrows != pdec_nrows:
        raise RuntimeError(
            "Source and PDEC dimensions do not match: "
            f"SRC=({src_ncols}, {src_nrows}) vs PDEC=({pdec_ncols}, {pdec_nrows})"
        )


def analyze_geoposition_mode(root: ET.Element) -> tuple[bool, ET.Element | None, ET.Element | None]:
    """
    Returns:
        should_clone_geoposition,
        template_crs,
        template_geoposition

    Rules:
        - If there is no Geoposition block, do not touch geocoding.
        - If any Geoposition block contains BAND_INDEX, clone CRS + Geoposition for new bands.
        - If Geoposition is global and has no BAND_INDEX, do not touch geocoding.
    """
    geopositions = root.findall("Geoposition")
    first_crs = root.find("Coordinate_Reference_System")

    if first_crs is None or not geopositions:
        return False, None, None

    uses_band_geoposition = any(geoposition.find("BAND_INDEX") is not None for geoposition in geopositions)

    if uses_band_geoposition:
        return True, first_crs, geopositions[0]

    return False, first_crs, geopositions[0]


def insert_geoposition_nodes_if_needed(root: ET.Element, bands_to_add: list[dict]) -> None:
    should_clone, template_crs, template_geoposition = analyze_geoposition_mode(root)

    if not should_clone:
        print("[edit] Global Geoposition detected, no CRS/Geoposition blocks will be added")
        return

    if template_crs is None or template_geoposition is None:
        raise RuntimeError("Could not determine a valid CRS/Geoposition template")

    root_children = list(root)
    insert_position = 0
    for index, child in enumerate(root_children):
        if child.tag == "Geoposition":
            insert_position = index + 1

    new_geocoding_nodes: list[ET.Element] = []
    for band in bands_to_add:
        new_geocoding_nodes.extend(
            clone_crs_geoposition_pair(
                template_crs=template_crs,
                template_geoposition=template_geoposition,
                band_index=band["band_index"],
            )
        )

    for offset, node in enumerate(new_geocoding_nodes):
        root.insert(insert_position + offset, node)

    print(f"[edit] Added CRS + Geoposition for {len(bands_to_add)} new bands")


def edit_pdec_dim(
    pdec_dim: Path,
    suffixes: list[str],
    is_tops: bool,
    backup: bool = True,
) -> None:
    if backup:
        backup_path = pdec_dim.with_suffix(pdec_dim.suffix + ".bak")
        shutil.copy2(pdec_dim, backup_path)
        print(f"[backup] {backup_path}")

    tree = ET.parse(pdec_dim)
    root = tree.getroot()

    raster_dimensions = find_required(root, "Raster_Dimensions")
    data_access = find_required(root, "Data_Access")
    image_interpretation = find_required(root, "Image_Interpretation")

    ncols = text_required(raster_dimensions, "NCOLS")
    nrows = text_required(raster_dimensions, "NROWS")

    existing_band_indices: list[int] = []
    for spectral_band in image_interpretation.findall("Spectral_Band_Info"):
        band_index_text = spectral_band.findtext("BAND_INDEX")
        if band_index_text is not None:
            existing_band_indices.append(int(band_index_text))

    start_index = max(existing_band_indices) + 1 if existing_band_indices else 0
    planned_new_bands = build_band_plan(suffixes, start_index=start_index)

    bands_to_add: list[dict] = []
    for band in planned_new_bands:
        if already_has_band(image_interpretation, band["band_name"]):
            print(f"[skip] Band already exists in PDEC.dim: {band['band_name']}")
        else:
            bands_to_add.append(band)

    if not bands_to_add:
        print("[info] No new bands need to be added")
        return

    old_nbands = int(text_required(raster_dimensions, "NBANDS"))
    find_required(raster_dimensions, "NBANDS").text = str(old_nbands + len(bands_to_add))
    print(f"[edit] NBANDS: {old_nbands} -> {old_nbands + len(bands_to_add)}")

    # Geoposition handling is decided from the actual PDEC structure.
    # The is_tops flag is kept for compatibility and logging only.
    print(f"[edit] is_TOPS={is_tops}")
    insert_geoposition_nodes_if_needed(root, bands_to_add)

    data_access_children = list(data_access)
    tie_point_insert_position = None
    for index, child in enumerate(data_access_children):
        if child.tag == "Tie_Point_Grid_File":
            tie_point_insert_position = index
            break
    if tie_point_insert_position is None:
        tie_point_insert_position = len(data_access_children)

    pdec_data_dir_name = pdec_dim.with_suffix("").name + ".data"

    data_files_to_add: list[ET.Element] = []
    for band in bands_to_add:
        if not band["virtual"]:
            href = f"{pdec_data_dir_name}/{band['file_name']}"
            data_files_to_add.append(build_data_file(href, band["band_index"]))

    for offset, node in enumerate(data_files_to_add):
        data_access.insert(tie_point_insert_position + offset, node)

    print(f"[edit] Added {len(data_files_to_add)} Data_File entries")

    for band in bands_to_add:
        spectral_band = build_spectral_band_info(
            band_index=band["band_index"],
            band_name=band["band_name"],
            width=ncols,
            height=nrows,
            physical_unit=band["physical_unit"],
            virtual=band["virtual"],
            expr=band["expr"],
        )
        image_interpretation.append(spectral_band)

    print(f"[edit] Added {len(bands_to_add)} Spectral_Band_Info entries")

    indent_xml(root)
    tree.write(pdec_dim, encoding="UTF-8", xml_declaration=False)
    print(f"[write] {pdec_dim}")


def merge_iq_into_pdec(
    src_dim: str | Path,
    pdec_dim: str | Path,
    is_tops: bool = False,
    overwrite_copied_files: bool = False,
    backup: bool = True,
) -> None:
    """
    Merge i/q bands from a source DIMAP product into a PDEC DIMAP product.

    This function:
        - autodetects all valid i_/q_ suffixes from SRC.data
        - copies i_/q_ .hdr/.img files into PDEC.data
        - updates PDEC.dim by adding:
            - Data_File entries for i_/q_
            - Spectral_Band_Info entries for i_/q_ and Intensity_*
            - Geoposition/CRS entries only when the original PDEC structure requires it
    """
    src_dim = Path(src_dim).resolve()
    pdec_dim = Path(pdec_dim).resolve()

    if not src_dim.exists():
        raise FileNotFoundError(f"Source DIM file does not exist: {src_dim}")
    if not pdec_dim.exists():
        raise FileNotFoundError(f"PDEC DIM file does not exist: {pdec_dim}")
    if src_dim == pdec_dim:
        raise ValueError(
            "src_dim and pdec_dim resolve to the same DIM product. "
            "This usually means Polarimetric-Decomposition failed and the pipeline "
            "kept the pre-PDEC product path."
        )

    src_data_dir = get_data_dir_from_dim(src_dim)
    pdec_data_dir = get_data_dir_from_dim(pdec_dim)

    if not src_data_dir.exists():
        raise FileNotFoundError(f"Source data directory does not exist: {src_data_dir}")
    if not pdec_data_dir.exists():
        raise FileNotFoundError(f"PDEC data directory does not exist: {pdec_data_dir}")

    suffixes = detect_suffixes_from_src_data(src_data_dir)
    if not suffixes:
        raise RuntimeError(
            f"No valid suffixes were detected in {src_data_dir}. "
            "Expected paired i_*.hdr/.img and q_*.hdr/.img files."
        )

    validate_same_dimensions(src_dim, pdec_dim)

    print(f"[info] SRC.dim :  {src_dim}")
    print(f"[info] PDEC.dim:  {pdec_dim}")
    print(f"[info] SRC.data : {src_data_dir}")
    print(f"[info] PDEC.data: {pdec_data_dir}")
    print(f"[info] is_TOPS : {is_tops}")
    print(f"[info] Autodetected suffixes ({len(suffixes)}): {suffixes}")

    copy_src_files(
        src_data_dir=src_data_dir,
        pdec_data_dir=pdec_data_dir,
        suffixes=suffixes,
        overwrite=overwrite_copied_files,
    )

    edit_pdec_dim(
        pdec_dim=pdec_dim,
        suffixes=suffixes,
        is_tops=is_tops,
        backup=backup,
    )

    print("\nDone.")
    print("You can now test the product with a SNAP Read operation or directly run Terrain Correction.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inject i/q bands from a source DIMAP product into a PDEC DIMAP product. "
            "Geoposition handling is decided from the actual structure of the original PDEC.dim."
        )
    )
    parser.add_argument("--src-dim", required=True, help="Path to the source .dim file, for example CAL or DEB")
    parser.add_argument("--pdec-dim", required=True, help="Path to the Polarimetric Decomposition .dim file")
    parser.add_argument(
        "--is_TOPS",
        action="store_true",
        help="Compatibility flag. It is logged, but Geoposition logic is now inferred from the PDEC structure.",
    )
    parser.add_argument(
        "--overwrite-copied-files",
        action="store_true",
        help="Overwrite i/q files if they already exist inside PDEC.data",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak backup of the PDEC.dim file",
    )
    args = parser.parse_args()

    merge_iq_into_pdec(
        src_dim=args.src_dim,
        pdec_dim=args.pdec_dim,
        is_tops=args.is_TOPS,
        overwrite_copied_files=args.overwrite_copied_files,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
