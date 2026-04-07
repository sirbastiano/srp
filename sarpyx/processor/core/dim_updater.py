import os
import re
import copy
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

# Supports:
#   i_VV.hdr
#   q_VV.hdr
#   i_IW1_VV.hdr
#   q_IW1_VV.hdr
#   L2_i_IW1_VV_SA1.hdr
#   L3_q_VH_SA2.hdr
#
# Where:
#   - optional L-prefix is restricted to "L<digits>_" (used to namespace multiple decompositions)
#   - optional swath token is "<2 letters><digit>" (IW1/EW3/SM?); if absent -> "SM-like"
HDR_RE = re.compile(
    r"^(?:(L\d+)_)?(i|q)_(?:(?P<swath>[A-Z]{2}\d)_)?(?P<pol>[A-Z]{2})(?:_(?P<sa>SA\d+))?$",
    re.IGNORECASE
)

def _rel_href(dim_path: str, filename: str) -> str:
    data_dir_name = os.path.basename(dim_path[:-4] + ".data")
    return f"./{data_dir_name}/{filename}"

def _safe_int(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None

def _scan_data_dir_for_groups(data_dir: str) -> List[Tuple[Optional[str], Optional[str], str, Optional[str]]]:
    """
    Return tuples: (Lpref, swath, pol, sa) for which i/q .hdr pairs exist.

    Examples
    --------
      (None, None, 'VV', None)            -> i_VV.hdr / q_VV.hdr
      (None, 'IW1', 'VV', None)           -> i_IW1_VV.hdr / q_IW1_VV.hdr
      ('L2', 'IW1', 'VV', 'SA1')          -> L2_i_IW1_VV_SA1.hdr / L2_q_IW1_VV_SA1.hdr
    """
    hdrs = [f for f in os.listdir(data_dir) if f.lower().endswith(".hdr")]
    found_i = set()
    found_q = set()

    for h in hdrs:
        stem = os.path.splitext(h)[0]
        m = HDR_RE.match(stem)
        if not m:
            continue

        Lpref = (m.group(1) or None)
        iq = (m.group(2) or "").lower()
        swath = m.group("swath")
        pol = m.group("pol")
        sa = m.group("sa")

        Lpref = Lpref.upper() if Lpref else None
        swath = swath.upper() if swath else None
        pol = pol.upper() if pol else None
        sa = sa.upper() if sa else None

        if not pol:
            continue

        key = (Lpref, swath, pol, sa)
        if iq == "i":
            found_i.add(key)
        else:
            found_q.add(key)

    return list(found_i.intersection(found_q))

def _sort_groups(groups: List[Tuple[Optional[str], Optional[str], str, Optional[str]]]) -> List[Tuple[Optional[str], Optional[str], str, Optional[str]]]:
    def parse_swath(sw: Optional[str]):
        if not sw:
            return ("", -1)
        m = re.match(r"^([A-Z]+)(\d+)$", sw.upper())
        if not m:
            return (sw.upper(), 9999)
        return (m.group(1), int(m.group(2)))

    def sort_key(t):
        Lpref, swath, pol, sa = t
        Lnum = int(Lpref[1:]) if Lpref else -1

        sw_alpha, sw_num = parse_swath(swath)
        sw_missing = 0 if swath is None else 1  # None first

        has_sa = 1 if sa else 0
        sanum = int(sa[2:]) if sa else -1

        return (Lnum, sw_missing, sw_alpha, sw_num, has_sa, sanum, pol)

    return sorted(groups, key=sort_key)

def _find_single(root: ET.Element, path: str) -> ET.Element:
    el = root.find(path)
    if el is None:
        raise RuntimeError(f"No encontré el nodo requerido: {path}")
    return el

def _find_parent(root: ET.Element, child: ET.Element) -> Optional[ET.Element]:
    for parent in root.iter():
        for c in list(parent):
            if c is child:
                return parent
    return None

def _build_existing_band_maps(root: ET.Element) -> Tuple[Dict[str, int], int]:
    """Map BAND_NAME -> BAND_INDEX for existing SBI entries (skip broken ones)."""
    name_to_idx: Dict[str, int] = {}
    max_idx = -1
    for sbi in root.findall(".//Image_Interpretation/Spectral_Band_Info"):
        name = (sbi.findtext("BAND_NAME") or "").strip()
        idx = _safe_int(sbi.findtext("BAND_INDEX"))
        if not name or idx is None:
            continue
        name_to_idx[name] = idx
        max_idx = max(max_idx, idx)
    return name_to_idx, max_idx

def _ensure_sbi(
    template_sbi: ET.Element,
    band_name: str,
    band_desc: str,
    unit: str,
    idx: int,
    expression: Optional[str] = None,
) -> ET.Element:
    sbi = copy.deepcopy(template_sbi)

    bi = sbi.find("BAND_INDEX")
    if bi is None:
        bi = ET.SubElement(sbi, "BAND_INDEX")
    bi.text = str(idx)

    bn = sbi.find("BAND_NAME")
    if bn is None:
        bn = ET.SubElement(sbi, "BAND_NAME")
    bn.text = band_name

    bd = sbi.find("BAND_DESCRIPTION")
    if bd is not None:
        bd.text = band_desc

    bu = sbi.find("BAND_UNIT")
    if bu is not None:
        bu.text = unit

    expr = sbi.find("EXPRESSION")
    if expression is not None:
        if expr is None:
            expr = ET.SubElement(sbi, "EXPRESSION")
        expr.text = expression

    return sbi

def _insert_data_file_in_order(data_access_el: ET.Element, df_el: ET.Element):
    children = list(data_access_el)
    first_tpg = next((i for i, ch in enumerate(children) if ch.tag == "Tie_Point_Grid_File"), None)
    if first_tpg is None:
        data_access_el.append(df_el)
    else:
        data_access_el.insert(first_tpg, df_el)

def _ensure_data_file(data_access_el: ET.Element, template_df: ET.Element, band_index: int, hdr_filename: str, dim_path: str):
    df = copy.deepcopy(template_df)

    bi = df.find("BAND_INDEX")
    if bi is None:
        bi = ET.SubElement(df, "BAND_INDEX")
    bi.text = str(band_index)

    df_path = df.find("DATA_FILE_PATH")
    if df_path is None:
        df_path = ET.SubElement(df, "DATA_FILE_PATH")
    df_path.set("href", _rel_href(dim_path, hdr_filename))

    _insert_data_file_in_order(data_access_el, df)

def _get_georef_templates(root: ET.Element):
    template_geo = root.find(".//Geoposition")
    if template_geo is None:
        raise RuntimeError("<Geoposition> node did not find to be used as template.")

    parent = _find_parent(root, template_geo)
    if parent is None:
        raise RuntimeError("Not possible to determine father of <Geoposition>.")

    children = list(parent)
    geo_idx = next((i for i, el in enumerate(children) if el is template_geo), None)
    if geo_idx is None:
        raise RuntimeError("Not founded the template of <Geoposition>.")

    template_crs = None
    if geo_idx > 0 and children[geo_idx - 1].tag == "Coordinate_Reference_System":
        template_crs = children[geo_idx - 1]
    else:
        template_crs = next((el for el in children if el.tag == "Coordinate_Reference_System"), None)

    if template_crs is None:
        raise RuntimeError("Not founded any <Coordinate_Reference_System> to be cloned.")

    return parent, template_crs, template_geo

def _insert_georef_pair_before_rasterdims(parent: ET.Element, crs_el: ET.Element, geo_el: ET.Element):
    children = list(parent)
    raster_idx = next((i for i, ch in enumerate(children) if ch.tag == "Raster_Dimensions"), None)
    if raster_idx is None:
        parent.append(crs_el)
        parent.append(geo_el)
        return
    parent.insert(raster_idx, crs_el)
    parent.insert(raster_idx + 1, geo_el)

def _append_georef_pair(parent: ET.Element, template_crs: ET.Element, template_geo: ET.Element, band_index: int):
    crs = copy.deepcopy(template_crs)
    geo = copy.deepcopy(template_geo)

    bi = geo.find("BAND_INDEX")
    if bi is None:
        bi = ET.SubElement(geo, "BAND_INDEX")
    bi.text = str(band_index)

    _insert_georef_pair_before_rasterdims(parent, crs, geo)

def _ensure_band_mdelem(abstract_md: ET.Element, template_band_md: ET.Element, swath: str, token: str, band_i_name: str, band_q_name: str):
    """Best-effort: keep Abstracted_Metadata coherent (SNAP is usually ok without this)."""
    new_name = f"Band_{swath}_{token}"
    for ch in abstract_md.findall("MDElem"):
        if ch.get("name") == new_name:
            return

    md = copy.deepcopy(template_band_md)
    md.set("name", new_name)

    pol_attr = md.find("./MDATTR[@name='polarization']")
    if pol_attr is not None:
        pol_attr.text = token

    bn_attr = md.find("./MDATTR[@name='band_names']")
    if bn_attr is not None:
        bn_attr.text = f"{band_i_name},{band_q_name}"

    abstract_md.append(md)

def update_dim_add_bands_from_data_dir(dim_in: str, dim_out: str = None, verbose: bool = True) -> str:
    """
    Add to a DIMAP .dim all (i/q/intensity) bands found in its .data folder.

    Supports IW/EW swath token in the band stems, and optional decomposition prefix L<k>_.

    Notes
    -----
    - The function expects ENVI .hdr files for i/q (and may add intensity SBI+geoposition even if the .hdr is absent),
      following the convention used by your subaperture generator.
    - Updates:
        * Raster_Dimensions/NBANDS
        * Image_Interpretation/Spectral_Band_Info
        * Data_Access/Data_File (for i/q .hdr)
        * Geocoding (Coordinate_Reference_System + Geoposition per band)
      plus best-effort Abstracted_Metadata Band_* entries when templates exist.
    """
    if not dim_in.lower().endswith(".dim"):
        raise ValueError(f"It was expected a .dim: {dim_in}")

    data_dir = dim_in[:-4] + ".data"
    groups = _sort_groups(_scan_data_dir_for_groups(data_dir))
    if not groups:
        raise RuntimeError(f"Not founded i/q .hdr valid pairs in {data_dir}")

    tree = ET.parse(dim_in)
    root = tree.getroot()

    raster_dims = _find_single(root, ".//Raster_Dimensions")
    nbands_el = _find_single(raster_dims, "NBANDS")

    img_interp = _find_single(root, ".//Image_Interpretation")
    data_access = _find_single(root, ".//Data_Access")

    georef_parent, template_crs, template_geo = _get_georef_templates(root)

    # SNAP readers can choke if the template Geoposition misses BAND_INDEX
    bi0 = template_geo.find("BAND_INDEX")
    if bi0 is None:
        bi0 = ET.SubElement(template_geo, "BAND_INDEX")
    if not (bi0.text or "").strip():
        bi0.text = "0"

    existing_dfs = root.findall(".//Data_File")
    if not existing_dfs:
        raise RuntimeError("The .dim does not have Data_File (template).")
    template_df = existing_dfs[0]

    sbis = root.findall(".//Image_Interpretation/Spectral_Band_Info")
    if len(sbis) < 3:
        raise RuntimeError("Do not found enough Spectral_Band_Info for templates.")
    template_sbi_i = next((x for x in sbis if (x.findtext("BAND_NAME") or "").startswith("i_")), sbis[0])
    template_sbi_q = next((x for x in sbis if (x.findtext("BAND_NAME") or "").startswith("q_")), sbis[1])
    template_sbi_int = next((x for x in sbis if (x.findtext("BAND_NAME") or "").lower().startswith("intensity")), sbis[2])

    # Best-effort Abstracted_Metadata templates (optional)
    abstract_md = None
    for md in root.findall(".//MDElem"):
        if md.get("name") == "Abstracted_Metadata":
            abstract_md = md
            break

    template_band_by_key = {}  # (swath, pol) -> template MDElem
    if abstract_md is not None:
        band_mds = [md for md in abstract_md.findall("MDElem") if (md.get("name") or "").startswith("Band_")]
        # try to collect Band_<swath>_<pol> templates
        for md in band_mds:
            name = md.get("name") or ""
            m = re.match(r"^Band_(?P<swath>[A-Z]{2}\d)_([A-Z]{2})$", name, re.IGNORECASE)
            if m:
                sw = m.group("swath").upper()
                pol = name.split("_")[-1].upper()
                template_band_by_key[(sw, pol)] = md

    name_to_idx, max_idx = _build_existing_band_maps(root)

    def _existing_df_indices() -> set:
        idxs = set()
        for df in root.findall(".//Data_File"):
            v = _safe_int(df.findtext("BAND_INDEX"))
            if v is not None:
                idxs.add(v)
        return idxs

    def _existing_geo_indices() -> set:
        idxs = set()
        for g in root.findall(".//Geoposition"):
            v = _safe_int(g.findtext("BAND_INDEX"))
            if v is not None:
                idxs.add(v)
        return idxs

    def ensure_triplet(Lpref: Optional[str], swath: Optional[str], pol: str, sa: Optional[str]):
        nonlocal max_idx

        token = pol if not sa else f"{pol}_{sa}"
        iq_prefix = f"{Lpref}_" if Lpref else ""
        sw_prefix = f"{swath}_" if swath else ""
        sa_suffix = f"_{sa}" if sa else ""

        b_i = f"{iq_prefix}i_{sw_prefix}{pol}{sa_suffix}"
        b_q = f"{iq_prefix}q_{sw_prefix}{pol}{sa_suffix}"
        b_int = f"Intensity_{iq_prefix}{sw_prefix}{pol}{sa_suffix}"

        def get_or_add(name: str) -> int:
            nonlocal max_idx
            if name in name_to_idx:
                return name_to_idx[name]
            max_idx += 1
            name_to_idx[name] = max_idx
            return max_idx

        idx_i = get_or_add(b_i)
        idx_q = get_or_add(b_q)
        idx_int = get_or_add(b_int)
        int_expr = f"{b_i} == 0.0 ? 0.0 : {b_i} * {b_i} + {b_q} * {b_q}"

        xml_band_names = set(
            (x.findtext("BAND_NAME") or "").strip()
            for x in root.findall(".//Image_Interpretation/Spectral_Band_Info")
        )

        if b_i not in xml_band_names:
            img_interp.append(_ensure_sbi(template_sbi_i, b_i, f"Real part ({token})", "real", idx_i))
        if b_q not in xml_band_names:
            img_interp.append(_ensure_sbi(template_sbi_q, b_q, f"Imag part ({token})", "imag", idx_q))
        if b_int not in xml_band_names:
            img_interp.append(
                _ensure_sbi(
                    template_sbi_int,
                    b_int,
                    f"Intensity ({token})",
                    "intensity",
                    idx_int,
                    expression=int_expr,
                )
            )

        df_idxs = _existing_df_indices()
        if idx_i not in df_idxs:
            _ensure_data_file(data_access, template_df, idx_i, f"{b_i}.hdr", dim_in)
        if idx_q not in df_idxs:
            _ensure_data_file(data_access, template_df, idx_q, f"{b_q}.hdr", dim_in)

        geo_idxs = _existing_geo_indices()
        for bi in (idx_i, idx_q, idx_int):
            if bi not in geo_idxs:
                _append_georef_pair(georef_parent, template_crs, template_geo, bi)
                geo_idxs.add(bi)

        # Optional Abstracted_Metadata update
        if abstract_md is not None and swath is not None:
            tmpl = template_band_by_key.get((swath, pol))
            if tmpl is not None:
                _ensure_band_mdelem(abstract_md, tmpl, swath, token, b_i, b_q)

    for (Lpref, swath, pol, sa) in groups:
        ensure_triplet(Lpref, swath, pol, sa)

    # If the product is switched to band-indexed geocoding, every indexed band
    # must have a matching Geoposition/CRS pair. Auxiliary bands such as
    # derampDemodPhase are not part of the i/q group scan and must be backfilled.
    geo_idxs = _existing_geo_indices()
    for sbi in root.findall(".//Image_Interpretation/Spectral_Band_Info"):
        band_index = _safe_int(sbi.findtext("BAND_INDEX"))
        if band_index is None or band_index in geo_idxs:
            continue
        _append_georef_pair(georef_parent, template_crs, template_geo, band_index)
        geo_idxs.add(band_index)

    nbands_el.text = str(len(root.findall(".//Image_Interpretation/Spectral_Band_Info")))

    if dim_out is None:
        dim_out = dim_in

    tree.write(dim_out, encoding="UTF-8", xml_declaration=True)

    if verbose:
        print(f"OK: write {dim_out}")
        print(f"  data_dir: {data_dir}")
        print(f"  grupos detected: {groups}")
        print(f"  NBANDS: {nbands_el.text}")

    return dim_out
