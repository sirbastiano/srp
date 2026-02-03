import os
import re
import copy
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

# Accepts:
#   i_VV.hdr
#   q_VV.hdr
#   L2_i_VV_SA1.hdr
#   L3_q_VH_SA2.hdr
HDR_RE = re.compile(
    r"^(?:(L\d+)_)?(i|q)_([A-Z]{2})(?:_(SA\d+))?$",
    re.IGNORECASE
)

def _rel_href(dim_path: str, filename: str) -> str:
    data_dir_name = os.path.basename(dim_path[:-4] + ".data")
    return f"./{data_dir_name}/{filename}"

def _scan_data_dir_for_groups(data_dir: str) -> List[Tuple[Optional[str], str, Optional[str]]]:
    """
    Devuelve tuplas (Lprefix, POL, SA):
      (None, 'VV', None)          -> i_VV/q_VV
      ('L2','VV','SA1')           -> L2_i_VV_SA1 / L2_q_VV_SA1
    Requiere pares i y q.
    """
    hdrs = [f for f in os.listdir(data_dir) if f.lower().endswith(".hdr")]
    found_i = set()
    found_q = set()

    for h in hdrs:
        stem = os.path.splitext(h)[0]
        m = HDR_RE.match(stem)
        if not m:
            continue

        Lpref = m.group(1)
        iq = m.group(2).lower()
        pol = m.group(3).upper()
        sa = m.group(4)
        sa = sa.upper() if sa else None
        Lpref = Lpref.upper() if Lpref else None

        key = (Lpref, pol, sa)
        if iq == "i":
            found_i.add(key)
        else:
            found_q.add(key)

    groups = list(found_i.intersection(found_q))
    return groups

def _sort_groups(groups: List[Tuple[Optional[str], str, Optional[str]]]) -> List[Tuple[Optional[str], str, Optional[str]]]:
    """
    Orden:
      - base sin Lpref primero (None)
      - luego L2, L3, ...
    Dentro:
      - sin SA primero, luego SA1..SAn
      - y por pol.
    """
    def sort_key(t):
        Lpref, pol, sa = t
        Lnum = int(Lpref[1:]) if Lpref else -1
        has_sa = 1 if sa else 0
        sanum = int(sa[2:]) if sa else -1
        return (Lnum, has_sa, sanum, pol)

    return sorted(groups, key=sort_key)

def _find_single(root: ET.Element, path: str) -> ET.Element:
    el = root.find(path)
    if el is None:
        raise RuntimeError(f"No encontré el nodo requerido: {path}")
    return el

def _build_existing_band_maps(root: ET.Element) -> Tuple[Dict[str, int], int]:
    name_to_idx: Dict[str, int] = {}
    max_idx = -1
    for sbi in root.findall(".//Image_Interpretation/Spectral_Band_Info"):
        name = (sbi.findtext("BAND_NAME") or "").strip()
        idx = int(sbi.findtext("BAND_INDEX"))
        name_to_idx[name] = idx
        max_idx = max(max_idx, idx)
    return name_to_idx, max_idx

def _ensure_sbi(template_sbi: ET.Element, band_name: str, band_desc: str, unit: str, idx: int) -> ET.Element:
    sbi = copy.deepcopy(template_sbi)
    sbi.find("BAND_INDEX").text = str(idx)
    sbi.find("BAND_NAME").text = band_name

    bd = sbi.find("BAND_DESCRIPTION")
    if bd is not None:
        bd.text = band_desc

    bu = sbi.find("BAND_UNIT")
    if bu is not None:
        bu.text = unit

    return sbi

def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in root.iter():
        for c in list(parent):
            if c is child:
                return parent
    return None

def _insert_data_file_in_order(data_access_el: ET.Element, df_el: ET.Element):
    children = list(data_access_el)
    first_tpg = next((i for i, ch in enumerate(children) if ch.tag == "Tie_Point_Grid_File"), None)
    if first_tpg is None:
        data_access_el.append(df_el)
    else:
        data_access_el.insert(first_tpg, df_el)

def _ensure_data_file(data_access_el: ET.Element, template_df: ET.Element, band_index: int, hdr_filename: str, dim_path: str):
    df = copy.deepcopy(template_df)
    df.find("BAND_INDEX").text = str(band_index)
    df_path = df.find("DATA_FILE_PATH")
    if df_path is None:
        raise RuntimeError("DATA_FILE_PATH no existe en el template de Data_File")
    df_path.set("href", _rel_href(dim_path, hdr_filename))
    _insert_data_file_in_order(data_access_el, df)

def _get_georef_templates(root: ET.Element):
    template_geo = root.find(".//Geoposition")
    if template_geo is None:
        raise RuntimeError("No encontré ningún nodo <Geoposition> para usar como template.")

    parent = _find_parent(root, template_geo)
    if parent is None:
        raise RuntimeError("No pude determinar el padre de <Geoposition>.")

    children = list(parent)
    geo_idx = next((i for i, el in enumerate(children) if el is template_geo), None)
    if geo_idx is None:
        raise RuntimeError("No pude ubicar el template <Geoposition>.")

    template_crs = None
    if geo_idx > 0 and children[geo_idx - 1].tag == "Coordinate_Reference_System":
        template_crs = children[geo_idx - 1]
    else:
        template_crs = next((el for el in children if el.tag == "Coordinate_Reference_System"), None)

    if template_crs is None:
        raise RuntimeError("No encontré ningún <Coordinate_Reference_System> para clonar.")

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
        raise RuntimeError("El template <Geoposition> no tiene <BAND_INDEX>.")
    bi.text = str(band_index)
    _insert_georef_pair_before_rasterdims(parent, crs, geo)

def _ensure_band_mdelem(abstract_md: ET.Element, template_band_md: ET.Element, swath: str, token: str, band_i_name: str, band_q_name: str):
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
    Agrega al .dim todas las bandas (i/q/intensity) encontradas en .data
    soportando prefijos tipo L2_ (p.ej. L2_i_VV_SA1.hdr).
    """
    if not dim_in.lower().endswith(".dim"):
        raise ValueError(f"Se esperaba un .dim: {dim_in}")

    data_dir = dim_in[:-4] + ".data"
    groups = _sort_groups(_scan_data_dir_for_groups(data_dir))
    if not groups:
        raise RuntimeError(f"No encontré pares i/q .hdr válidos en {data_dir}")

    tree = ET.parse(dim_in)
    root = tree.getroot()

    raster_dims = _find_single(root, ".//Raster_Dimensions")
    nbands_el = _find_single(raster_dims, "NBANDS")

    img_interp = _find_single(root, ".//Image_Interpretation")
    data_access = _find_single(root, ".//Data_Access")

    georef_parent, template_crs, template_geo = _get_georef_templates(root)

    abstract_md = None
    for md in root.findall(".//MDElem"):
        if md.get("name") == "Abstracted_Metadata":
            abstract_md = md
            break
    if abstract_md is None:
        raise RuntimeError("No encontré MDElem name='Abstracted_Metadata'")

    existing_dfs = root.findall(".//Data_File")
    if not existing_dfs:
        raise RuntimeError("El .dim no tiene Data_File (template).")
    template_df = existing_dfs[0]

    sbis = root.findall(".//Image_Interpretation/Spectral_Band_Info")
    if len(sbis) < 3:
        raise RuntimeError("No encuentro suficientes Spectral_Band_Info para templates.")

    template_sbi_i = next((x for x in sbis if (x.findtext("BAND_NAME") or "").startswith("i_")), sbis[0])
    template_sbi_q = next((x for x in sbis if (x.findtext("BAND_NAME") or "").startswith("q_")), sbis[1])
    template_sbi_int = next((x for x in sbis if (x.findtext("BAND_NAME") or "").lower().startswith("intensity")), sbis[2])

    band_mds = [md for md in abstract_md.findall("MDElem") if (md.get("name") or "").startswith("Band_")]
    if not band_mds:
        raise RuntimeError("No encontré Band_* dentro de Abstracted_Metadata.")
    m0 = re.match(r"Band_(S\d+)_", band_mds[0].get("name") or "")
    if not m0:
        raise RuntimeError(f"No pude inferir swath desde {band_mds[0].get('name')}")
    swath = m0.group(1)

    template_band_by_pol = {}
    for md in band_mds:
        name = md.get("name") or ""
        m = re.match(rf"Band_{swath}_([A-Z]{{2}})$", name)
        if m:
            template_band_by_pol[m.group(1)] = md

    name_to_idx, max_idx = _build_existing_band_maps(root)

    def ensure_triplet(Lpref: Optional[str], pol: str, sa: Optional[str]):
        nonlocal max_idx

        # Construir el “token” que SNAP usa en Band_* (polarization)
        # ejemplo: VV, VH_SA1, etc.
        token = pol if not sa else f"{pol}_{sa}"

        # Band names EXACTOS como stems de los .hdr:
        #   i/q: (L2_)i_VV(_SA1)
        #   intensity: Intensity_(L2_)VV(_SA1)
        # Nota: intensity NO tiene L2_ en el original SNAP, pero vos lo querés coherente con lo generado.
        # Si preferís otra convención, lo ajustamos.
        iq_prefix = f"{Lpref}_" if Lpref else ""
        sa_suffix = f"_{sa}" if sa else ""

        b_i = f"{iq_prefix}i_{pol}{sa_suffix}"
        b_q = f"{iq_prefix}q_{pol}{sa_suffix}"
        b_int = f"Intensity_{iq_prefix}{pol}{sa_suffix}"

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

        xml_band_names = set(
            (x.findtext("BAND_NAME") or "").strip()
            for x in root.findall(".//Image_Interpretation/Spectral_Band_Info")
        )

        if b_i not in xml_band_names:
            img_interp.append(_ensure_sbi(template_sbi_i, b_i, f"Real part ({token})", "real", idx_i))
        if b_q not in xml_band_names:
            img_interp.append(_ensure_sbi(template_sbi_q, b_q, f"Imag part ({token})", "imag", idx_q))
        if b_int not in xml_band_names:
            img_interp.append(_ensure_sbi(template_sbi_int, b_int, f"Intensity ({token})", "intensity", idx_int))

        existing_df_indices = set(int(df.findtext("BAND_INDEX")) for df in root.findall(".//Data_File"))
        if idx_i not in existing_df_indices:
            _ensure_data_file(data_access, template_df, idx_i, f"{b_i}.hdr", dim_in)
        if idx_q not in existing_df_indices:
            _ensure_data_file(data_access, template_df, idx_q, f"{b_q}.hdr", dim_in)

        existing_geo_indices = set(int(g.findtext("BAND_INDEX")) for g in root.findall(".//Geoposition"))
        for bi in (idx_i, idx_q, idx_int):
            if bi not in existing_geo_indices:
                _append_georef_pair(georef_parent, template_crs, template_geo, bi)
                existing_geo_indices.add(bi)

        tmpl = template_band_by_pol.get(pol)
        if tmpl is not None:
            _ensure_band_mdelem(
                abstract_md,
                tmpl,
                swath,
                token,      # Band_S3_<token>
                b_i,
                b_q
            )

    for (Lpref, pol, sa) in groups:
        ensure_triplet(Lpref, pol, sa)

    nbands_el.text = str(len(root.findall(".//Image_Interpretation/Spectral_Band_Info")))

    if dim_out is None:
        dim_out = dim_in #[:-4] + "_subaps.dim"

    tree.write(dim_out, encoding="UTF-8", xml_declaration=True)

    if verbose:
        print(f"OK: escrito {dim_out}")
        print(f"  data_dir: {data_dir}")
        print(f"  grupos detectados: {groups}")
        print(f"  NBANDS: {nbands_el.text}")

    return dim_out
