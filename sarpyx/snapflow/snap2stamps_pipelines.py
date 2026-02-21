"""snap2stamps-compatible workflows built on top of :mod:`sarpyx.snapflow.engine`.

This module mirrors SNAP2StaMPS v2 functionality by executing the original SNAP
XML graph templates through :class:`~sarpyx.snapflow.engine.GPT.run_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import configparser
import os
import re
import shutil
import subprocess
import tarfile

from lxml import etree

from sarpyx.snapflow.engine import GPT
from sarpyx.utils.io import delProd


# Graph filename stem (normalized) -> template filename
GRAPH_STEM_TO_TEMPLATE: dict[str, str] = {
    "plot_amplitude": "plot_amplitude.xml",
    "plot_phase": "plot_phase.xml",
    "stripmap_dem_assisted_coregistration": "stripmap_DEM_Assisted_Coregistration.xml",
    "stripmap_dem_assisted_coregistration_extdem": "stripmap_DEM_Assisted_Coregistration_extDEM.xml",
    "stripmap_export": "stripmap_Export.xml",
    "stripmap_interferogram_topophase": "stripmap_Interferogram_TopoPhase.xml",
    "stripmap_interferogram_topophase_extdem": "stripmap_Interferogram_TopoPhase_extDEM.xml",
    "stripmap_tsx_subset": "stripmap_TSX_Subset.xml",
    "stripmap_plot_coreg": "stripmap_plot_coreg.xml",
    "stripmap_plot_ifg": "stripmap_plot_ifg.xml",
    "stripmap_plot_split": "stripmap_plot_split.xml",
    "topsar_coreg_ifg_computation": "topsar_coreg_ifg_computation.xml",
    "topsar_coreg_ifg_computation_extdem": "topsar_coreg_ifg_computation_extDEM.xml",
    "topsar_coreg_ifg_computation_noesd": "topsar_coreg_ifg_computation_noESD.xml",
    "topsar_coreg_ifg_computation_noesd_extdem": "topsar_coreg_ifg_computation_noESD_extDEM.xml",
    "topsar_coreg_ifg_computation_subset": "topsar_coreg_ifg_computation_subset.xml",
    "topsar_coreg_ifg_computation_subset_extdem": "topsar_coreg_ifg_computation_subset_extDEM.xml",
    "topsar_coreg_ifg_computation_subset_noesd": "topsar_coreg_ifg_computation_subset_noESD.xml",
    "topsar_coreg_ifg_computation_subset_noesd_extdem": "topsar_coreg_ifg_computation_subset_noESD_extDEM.xml",
    "topsar_export": "topsar_export.xml",
    "topsar_export_mergeiw_subset": "topsar_export_mergeIW_subset.xml",
    "topsar_export_mergeiw_subset_extdem": "topsar_export_mergeIW_subset_extDEM.xml",
    "topsar_master_assemble_split_applyorbit": "topsar_master_assemble_split_applyorbit.xml",
    "topsar_master_split_applyorbit": "topsar_master_split_applyorbit.xml",
    "topsar_plot_amplitude_deburst": "topsar_plot_amplitude_deburst.xml",
    "topsar_secondaries_assemble_split_applyorbit": "topsar_secondaries_assemble_split_applyorbit.xml",
    "topsar_secondaries_split_applyorbit": "topsar_secondaries_split_applyorbit.xml",
}


SNAP2STAMPS_WORKFLOWS: dict[str, tuple[str, ...]] = {
    "topsar_split": (
        "topsar_master_split_applyorbit",
        "topsar_secondaries_split_applyorbit",
    ),
    "topsar_coreg_ifg": ("topsar_coreg_ifg_computation",),
    "topsar_export": ("topsar_export",),
    "stripmap_subset_coreg_ifg_export": (
        "stripmap_tsx_subset",
        "stripmap_dem_assisted_coregistration",
        "stripmap_interferogram_topophase",
        "stripmap_export",
    ),
}


def _normalize_stem(graph_file: Path) -> str:
    stem = graph_file.stem.lower()
    stem = re.sub(r"^\d+[_-]?", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def operators_from_graph_xml(graph_file: str | Path) -> tuple[str, ...]:
    root = etree.parse(str(graph_file)).getroot()
    ops: list[str] = []
    for op in root.xpath("//node/operator/text()"):
        op_name = str(op).strip()
        if op_name in {"Read", "Write"}:
            continue
        ops.append(op_name)
    return tuple(ops)


def verify_graph_coverage(graph_dir: str | Path) -> dict[str, Any]:
    """Check that all XML graphs in ``graph_dir`` are mapped for execution."""
    graph_dir = Path(graph_dir)
    xml_files = sorted(graph_dir.glob("*.xml"))

    report: dict[str, Any] = {
        "graph_dir": graph_dir.as_posix(),
        "graph_count": len(xml_files),
        "unmapped_graphs": [],
        "verified": [],
        "missing_template_files": [],
    }

    for xml in xml_files:
        norm_stem = _normalize_stem(xml)
        template_name = GRAPH_STEM_TO_TEMPLATE.get(norm_stem)
        if template_name is None:
            report["unmapped_graphs"].append(xml.name)
            continue

        template_path = graph_dir / template_name
        if not template_path.exists():
            report["missing_template_files"].append(template_name)
            continue

        report["verified"].append({"graph": xml.name, "template": template_name})

    report["ok"] = not report["unmapped_graphs"] and not report["missing_template_files"]
    return report


@dataclass
class Snap2StampsProject:
    """Parsed project configuration for a snap2stamps run."""

    path: Path
    config: configparser.ConfigParser

    @property
    def project_folder(self) -> Path:
        return Path(self.config["PROJECT_DEFINITION"].get("PROJECTFOLDER", "")).expanduser().resolve()

    @property
    def graphs_folder(self) -> Path:
        graph_dir = self.config["PROJECT_DEFINITION"].get("GRAPHSFOLDER", "")
        if graph_dir:
            return Path(graph_dir).expanduser().resolve()
        return (self.path.parent / "../graphs").resolve()

    @property
    def sensor(self) -> str:
        return self.config["PROC_PARAMETERS"].get("SENSOR", "").upper()

    @property
    def polarisation(self) -> str:
        return self.config["PROC_PARAMETERS"].get("POLARISATION", "VV")

    @property
    def ext_dem(self) -> str:
        return self.config["PROC_PARAMETERS"].get("EXTDEM", "")

    @property
    def overwrite(self) -> str:
        return self.config["PROC_OPTIONS"].get("OVERWRITE", "N").upper()

    @property
    def smart_hdd(self) -> str:
        return self.config["PROC_OPTIONS"].get("SMARTHDD", "N").upper()

    @property
    def plotting(self) -> str:
        return self.config["PROC_OPTIONS"].get("PLOTTING", "N").upper()

    @property
    def cache(self) -> str:
        return self.config["COMPUTING_RESOURCES"].get("CACHE", "16G")

    @property
    def cpu(self) -> str:
        return self.config["COMPUTING_RESOURCES"].get("CPU", "8")

    @property
    def master_selection(self) -> str:
        return self.config["PROC_PARAMETERS"].get("MASTERSEL", "MANUAL").upper()

    @property
    def auto_download(self) -> str:
        if "SEARCH_PARAMS" not in self.config:
            return "N"
        return self.config["SEARCH_PARAMS"].get("autoDownload", "N").upper()

    @property
    def master_path(self) -> Path:
        raw = self.config["PROC_PARAMETERS"].get("MASTER", "") or self.config["PROC_PARAMETERS"].get("MASTERFOLDER", "")
        return Path(raw).expanduser().resolve() if raw else (self.project_folder / "master")


@dataclass
class Snap2StampsRunner:
    """End-to-end runner compatible with SNAP2StaMPS v2 step scripts."""

    project: Snap2StampsProject
    gpt_path: str = "/usr/local/snap/bin/gpt"
    memory: Optional[str] = None
    parallelism: Optional[int] = None
    timeout: Optional[int] = 7200
    snap_userdir: Optional[str | Path] = None

    @classmethod
    def from_project_file(
        cls,
        project_file: str | Path,
        *,
        gpt_path: str = "/usr/local/snap/bin/gpt",
        memory: Optional[str] = None,
        parallelism: Optional[int] = None,
        timeout: Optional[int] = 7200,
        snap_userdir: Optional[str | Path] = None,
    ) -> "Snap2StampsRunner":
        cfg = configparser.ConfigParser()
        cfg.read(str(project_file))
        proj = Snap2StampsProject(path=Path(project_file).resolve(), config=cfg)
        return cls(
            project=proj,
            gpt_path=gpt_path,
            memory=memory,
            parallelism=parallelism,
            timeout=timeout,
            snap_userdir=snap_userdir,
        )

    def _new_gpt(self, anchor_product: Path, outdir: Path, fmt: str = "BEAM-DIMAP") -> GPT:
        return GPT(
            product=anchor_product,
            outdir=outdir,
            format=fmt,
            gpt_path=self.gpt_path,
            memory=self.memory,
            parallelism=self.parallelism,
            timeout=self.timeout,
            snap_userdir=self.snap_userdir,
        )

    def _resolve_aoi_wkt(self) -> str:
        aoi_mode = self.project.config["AOI_DEFINITION"].get("AOI_MODE", "WKT")
        aoi_mode_u = aoi_mode.upper()

        if aoi_mode_u == "WKT":
            return self.project.config["AOI_DEFINITION"].get("WKT", "")

        if aoi_mode_u == "BBOX":
            lonmin = self.project.config["AOI_DEFINITION"].get("LONMIN")
            latmin = self.project.config["AOI_DEFINITION"].get("LATMIN")
            lonmax = self.project.config["AOI_DEFINITION"].get("LONMAX")
            latmax = self.project.config["AOI_DEFINITION"].get("LATMAX")
            return f"POLYGON (({lonmin} {latmin},{lonmax} {latmin},{lonmax} {latmax},{lonmin} {latmax},{lonmin} {latmin}))"

        if aoi_mode_u in {"SHP", "GEOJSON", "KML"}:
            import geopandas as gpd

            aoi_file = self.project.config["AOI_DEFINITION"].get("AOI_FILE", "")
            if aoi_mode_u == "KML":
                import fiona

                fiona.drvsupport.supported_drivers["KML"] = "rw"
                gdf = gpd.read_file(aoi_file, driver="KML")
            else:
                gdf = gpd.read_file(aoi_file)
            return str(gdf.envelope.iloc[0])

        raise ValueError(f"Unsupported AOI_MODE: {aoi_mode}")

    def _render_graph(
        self,
        graph_name: str,
        replacements: dict[str, Any],
        generated_graph_path: Path,
    ) -> Path:
        template_file = GRAPH_STEM_TO_TEMPLATE.get(graph_name, graph_name)
        template_path = self.project.graphs_folder / template_file
        if not template_path.exists():
            raise FileNotFoundError(f"Graph template not found: {template_path}")

        content = template_path.read_text(encoding="utf-8")
        for key in sorted(replacements, key=len, reverse=True):
            content = content.replace(key, str(replacements[key]))

        generated_graph_path.parent.mkdir(parents=True, exist_ok=True)
        generated_graph_path.write_text(content, encoding="utf-8")
        return generated_graph_path

    def run_graph_pipeline(
        self,
        graph_name: str,
        replacements: dict[str, Any],
        *,
        anchor_product: Path,
        output_path: Path,
        outdir: Optional[Path] = None,
    ) -> Path:
        graph_outdir = outdir or (self.project.project_folder / "graphs")
        gpt = self._new_gpt(anchor_product=anchor_product, outdir=output_path.parent)
        graph_2run = graph_outdir / f"{graph_name}_2run.xml"
        graph_path = self._render_graph(graph_name, replacements, graph_2run)

        result = gpt.run_graph(graph_path=graph_path, output_path=output_path)
        if result is None:
            raise RuntimeError(f"Graph {graph_name} failed: {gpt.last_error_summary()}")
        return Path(result)

    def prepare_topsar_secondaries(self) -> None:
        secondaries = self.project.project_folder / "secondaries"
        secondaries.mkdir(parents=True, exist_ok=True)

        for manifest in sorted(secondaries.glob("*SAFE/manifest.safe")):
            safe_name = manifest.parent.name
            date_dir = secondaries / safe_name[17:25]
            date_dir.mkdir(parents=True, exist_ok=True)
            dst = date_dir / manifest.name
            if manifest != dst:
                shutil.move(manifest.as_posix(), dst.as_posix())

        for archive in sorted(secondaries.glob("*.zip")):
            if not archive.name.startswith("S1"):
                continue
            date_dir = secondaries / archive.name[17:25]
            date_dir.mkdir(parents=True, exist_ok=True)
            dst = date_dir / archive.name
            if archive != dst:
                shutil.move(archive.as_posix(), dst.as_posix())

    def select_topsar_master(self, mode: Optional[str] = None) -> Optional[str]:
        secondaries = self.project.project_folder / "secondaries"
        master_folder = self.project.master_path
        master_folder.mkdir(parents=True, exist_ok=True)
        dirs = sorted([d.name for d in secondaries.iterdir() if d.is_dir()])
        if not dirs:
            return None

        mode_val = (mode or self.project.master_selection).upper()
        options = list(range(1, len(dirs) + 1))

        if mode_val == "AUTO":
            idx = max(0, int(len(dirs) // 2) - 1)
        elif mode_val == "FIRST":
            idx = 0
        elif mode_val == "LAST":
            idx = len(dirs) - 1
        elif mode_val == "MANUAL":
            return None
        elif mode_val.isdigit() and int(mode_val) in options:
            idx = int(mode_val) - 1
        else:
            idx = max(0, int(len(dirs) // 2) - 1)

        selected = dirs[idx]
        src = secondaries / selected
        for item in sorted(src.iterdir()):
            shutil.move(item.as_posix(), (master_folder / item.name).as_posix())
        src.rmdir()
        return selected

    def _master_scene_files(self) -> list[Path]:
        master_folder = self.project.master_path
        files = sorted(master_folder.glob("*.zip"))
        if files:
            return files
        return sorted(master_folder.glob("*SAFE/manifest.safe"))

    def run_topsar_split_master(self) -> list[Path]:
        files = self._master_scene_files()
        if not files:
            raise FileNotFoundError(f"No master scenes found in {self.project.master_path}")

        polygon = self._resolve_aoi_wkt()
        split_master = self.project.project_folder / "MasterSplit"
        split_master.mkdir(parents=True, exist_ok=True)

        base_name = files[0].parent.name if files[0].name == "manifest.safe" else files[0].name
        acq_date = base_name[17:25] if len(base_name) >= 25 else files[0].stem[:8]
        outputs: list[Path] = []

        for iw in ("IW1", "IW2", "IW3"):
            output = split_master / f"{acq_date}_{iw}.dim"
            graph_name = (
                "topsar_master_split_applyorbit"
                if len(files) == 1
                else "topsar_master_assemble_split_applyorbit"
            )
            repl = {
                "IWs": iw,
                "POLARISATION": self.project.polarisation,
                "POLYGON": polygon,
                "OUTPUTFILE": output.as_posix(),
            }
            if len(files) == 1:
                repl["INPUTFILE"] = files[0].as_posix()
            else:
                repl["INPUTFILE1"] = files[0].as_posix()
                repl["INPUTFILE2"] = files[1].as_posix()

            self.run_graph_pipeline(
                graph_name=graph_name,
                replacements=repl,
                anchor_product=files[0],
                output_path=output,
            )
            outputs.append(output)

        return outputs

    def run_topsar_split_secondaries(self) -> list[Path]:
        secondaries = self.project.project_folder / "secondaries"
        masters = sorted((self.project.project_folder / "MasterSplit").glob("*.dim"))
        if not masters:
            raise FileNotFoundError("No master split products found; run run_topsar_split_master() first.")

        polygon = self._resolve_aoi_wkt()
        split_dir = self.project.project_folder / "split"
        split_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []

        for date_dir in sorted([d for d in secondaries.iterdir() if d.is_dir()]):
            files = sorted(date_dir.glob("*.zip"))
            if not files:
                files = sorted(date_dir.glob("*SAFE/manifest.safe"))
            if not files:
                continue

            for master in masters:
                iw = master.stem[-3:]
                output = split_dir / f"{date_dir.name}_{iw}.dim"
                if output.exists() and self.project.overwrite == "N":
                    continue

                graph_name = (
                    "topsar_secondaries_split_applyorbit"
                    if len(files) == 1
                    else "topsar_secondaries_assemble_split_applyorbit"
                )
                repl = {
                    "IWs": iw,
                    "POLARISATION": self.project.polarisation,
                    "POLYGON": polygon,
                    "OUTPUTFILE": output.as_posix(),
                }
                if len(files) == 1:
                    repl["INPUTFILE"] = files[0].as_posix()
                else:
                    repl["INPUTFILE1"] = files[0].as_posix()
                    repl["INPUTFILE2"] = files[1].as_posix()

                self.run_graph_pipeline(
                    graph_name=graph_name,
                    replacements=repl,
                    anchor_product=files[0],
                    output_path=output,
                )
                outputs.append(output)

        return outputs

    @staticmethod
    def _get_nbursts(dim_file: Path) -> int:
        try:
            lines = dim_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return 1

        for i, line in enumerate(lines):
            if "burstList" in line and i + 1 < len(lines):
                try:
                    return int(lines[i + 1].split(">", 1)[1].split("<", 1)[0])
                except Exception:
                    return 1
        return 1

    def _smart_hdd_coreg_cleanup(self, output_name: str, index: int) -> None:
        if self.project.smart_hdd != "Y":
            return

        coreg_dir = self.project.project_folder / "coreg"
        ifg_dir = self.project.project_folder / "ifg"

        try:
            coreg_data = sorted(coreg_dir.glob("*.data"))[0]
            ifg_data = sorted(ifg_dir.glob("*.data"))[0]
        except IndexError:
            return

        ifile = next(iter(coreg_data.glob("i*mst*img")), None)
        qfile = next(iter(coreg_data.glob("q*mst*img")), None)
        dem = next(iter(ifg_data.glob("elevation.img")), None)
        lat = next(iter(ifg_data.glob("orthorectifiedLat.img")), None)
        lon = next(iter(ifg_data.glob("orthorectifiedLon.img")), None)

        if index <= 1:
            return

        target_data = coreg_dir / f"{output_name}.data"
        for src_ref, patt in ((ifile, "*mst*.img"), (qfile, "*mst*.img")):
            if src_ref is None:
                continue
            files = sorted(target_data.glob(patt))
            for f in files:
                try:
                    f.unlink(missing_ok=True)
                    os.symlink(src_ref.as_posix(), f.as_posix())
                except OSError:
                    pass

        ifg_target = ifg_dir / f"{output_name}.data"
        for src_ref, patt in ((lat, "orthorectifiedLat.img"), (lon, "orthorectifiedLon.img"), (dem, "elevation.img")):
            if src_ref is None:
                continue
            for f in ifg_target.glob(patt):
                try:
                    f.unlink(missing_ok=True)
                    os.symlink(src_ref.as_posix(), f.as_posix())
                except OSError:
                    pass

    def run_topsar_coreg_ifg(self) -> list[Path]:
        master_dir = self.project.project_folder / "MasterSplit"
        split_dir = self.project.project_folder / "split"
        coreg_dir = self.project.project_folder / "coreg"
        ifg_dir = self.project.project_folder / "ifg"
        coreg_dir.mkdir(parents=True, exist_ok=True)
        ifg_dir.mkdir(parents=True, exist_ok=True)

        masters = sorted(master_dir.glob("*.dim"))
        if not masters:
            raise FileNotFoundError("No split master products found.")

        polygon = self._resolve_aoi_wkt()
        outputs: list[Path] = []

        for master in masters:
            nmasters = len(masters)
            graph_name = "topsar_coreg_ifg_computation"
            if nmasters == 1:
                graph_name += "_subset"
            if self._get_nbursts(master) == 1:
                graph_name += "_noesd"
            if self.project.ext_dem:
                graph_name += "_extdem"

            iw = master.stem[-3:]
            k = 0
            for secondary in sorted(split_dir.glob(f"*{iw}.dim")):
                k += 1
                output_name = f"{master.stem[0:8]}_{secondary.stem[0:8]}_{iw}"
                output_coreg = coreg_dir / f"{output_name}.dim"
                if output_coreg.exists() and self.project.overwrite == "N":
                    continue

                repl = {
                    "MASTER": master.as_posix(),
                    "SECONDARY": secondary.as_posix(),
                    "EXTERNALDEM": self.project.ext_dem,
                    "OUTPUTCOREGFOLDER": coreg_dir.as_posix(),
                    "OUTPUTIFGFOLDER": ifg_dir.as_posix(),
                    "OUTPUTFILE": output_name,
                    "POLYGON": polygon,
                }
                self.run_graph_pipeline(
                    graph_name=graph_name,
                    replacements=repl,
                    anchor_product=master,
                    output_path=output_coreg,
                )
                outputs.append(output_coreg)
                self._smart_hdd_coreg_cleanup(output_name=output_name, index=k)

        return outputs

    @staticmethod
    def _discover_complex_bands(data_dir: Path, i_pattern: str, q_pattern: str) -> tuple[str, str]:
        ibands = sorted(data_dir.glob(i_pattern))
        qbands = sorted(data_dir.glob(q_pattern))
        if not ibands or not qbands:
            raise FileNotFoundError(f"Missing i/q bands in {data_dir}")
        return ibands[0].stem, qbands[0].stem

    def _run_pconvert(self, input_dim: Path, output_folder: Path, mode: str) -> None:
        pconvert = Path(self.gpt_path).with_name("pconvert")
        if not pconvert.exists():
            return

        cmd = [pconvert.as_posix(), "-b", "1", "-f", "png", "-o", output_folder.as_posix(), input_dim.as_posix()]
        if mode == "ifg":
            palette = (
                Path(self.snap_userdir).expanduser()
                if self.snap_userdir
                else (Path.home() / ".snap")
            ) / "auxdata" / "color_palettes" / "spectrum_cycle.cpd"
            if palette.exists():
                cmd[1:1] = ["-c", palette.as_posix()]

        subprocess.run(cmd, check=False)

    def run_topsar_plotting(self, mode: str) -> list[Path]:
        mode = mode.lower()
        if mode not in {"split", "coreg", "ifg"}:
            raise ValueError("mode must be split/coreg/ifg")

        project = self.project.project_folder
        master_dir = project / "MasterSplit"
        search_folder = {
            "split": project / "split",
            "coreg": project / "coreg",
            "ifg": project / "ifg",
        }[mode]
        plot_folder = project / "plot" / mode
        plot_folder.mkdir(parents=True, exist_ok=True)

        if mode == "split":
            graph_name = "topsar_plot_amplitude_deburst"
            i_pattern = f"i_*{self.project.polarisation}*.img"
            q_pattern = f"q_*{self.project.polarisation}*.img"
        elif mode == "coreg":
            graph_name = "plot_amplitude"
            i_pattern = f"i_*{self.project.polarisation}_slv1*.img"
            q_pattern = f"q_*{self.project.polarisation}_slv1*.img"
        else:
            graph_name = "plot_phase"
            i_pattern = f"i_ifg_*{self.project.polarisation}*.img"
            q_pattern = f"q_ifg_*{self.project.polarisation}*.img"

        outputs: list[Path] = []

        for master in sorted(master_dir.glob("*.dim")):
            iw = master.stem[-3:]
            for dim in sorted(search_folder.glob(f"*{iw}.dim")):
                output_name = f"{dim.stem}_{mode}"
                output_dim = plot_folder / f"{output_name}.dim"
                if (plot_folder / f"{output_name}.png").exists() and self.project.overwrite == "N":
                    continue

                data_dir = search_folder / f"{dim.stem}.data"
                iband, qband = self._discover_complex_bands(data_dir, i_pattern, q_pattern)
                repl = {
                    "INPUTFILE": dim.as_posix(),
                    "OUTPUTFOLDER": plot_folder.as_posix(),
                    "OUTPUTNAME": output_name,
                    "IBAND": iband,
                    "QBAND": qband,
                }
                self.run_graph_pipeline(
                    graph_name=graph_name,
                    replacements=repl,
                    anchor_product=dim,
                    output_path=output_dim,
                )
                outputs.append(output_dim)
                self._run_pconvert(input_dim=output_dim, output_folder=plot_folder, mode=mode)

        return outputs

    def run_topsar_export(self) -> list[Path]:
        project = self.project.project_folder
        master_dir = project / "MasterSplit"
        coreg_dir = project / "coreg"
        ifg_dir = project / "ifg"

        masters = sorted(master_dir.glob("*.dim"))
        if not masters:
            raise FileNotFoundError("No split master products found for export.")

        out_export = project / f"INSAR_{masters[0].stem[0:8]}"
        out_export.mkdir(parents=True, exist_ok=True)

        graph_name = "topsar_export" if len(masters) == 1 else "topsar_export_mergeiw_subset"
        if len(masters) > 1 and self.project.ext_dem:
            graph_name = "topsar_export_mergeiw_subset_extdem"

        pairs = sorted({f.stem[:17] for f in coreg_dir.glob("*.dim")})
        outputs: list[Path] = []

        for idx, pair in enumerate(pairs, start=1):
            dimfiles = sorted(coreg_dir.glob(f"{pair}*.dim"))
            ifgfiles = sorted(ifg_dir.glob(f"{pair}*.dim"))
            if not dimfiles or not ifgfiles:
                continue

            output_marker = out_export / f"{pair}.dim"
            repl = {
                "COREGFILE": ",".join(f.as_posix() for f in dimfiles),
                "IFGFILE": ",".join(f.as_posix() for f in ifgfiles),
                "OUTPUTFOLDER": out_export.as_posix(),
                "EXTERNALDEM": self.project.ext_dem,
                "POLYGON": self._resolve_aoi_wkt(),
            }
            self.run_graph_pipeline(
                graph_name=graph_name,
                replacements=repl,
                anchor_product=dimfiles[0],
                output_path=output_marker,
            )
            outputs.append(output_marker)

            if self.project.smart_hdd == "Y" and idx > 1:
                for f in dimfiles + ifgfiles:
                    try:
                        delProd(f)
                    except Exception:
                        pass

        return outputs

    def download_asf_s1(self) -> int:
        """Optional Sentinel-1 auto-download equivalent to asf_s1_downloader.py."""
        try:
            import asf_search as asf
        except ImportError as exc:
            raise ImportError("asf_search is required for autoDownload=Y") from exc

        search = self.project.config["SEARCH_PARAMS"]
        track = int(search.get("TRACK"))
        beam_mode = search.get("beamMode", "SLC")
        start = search.get("START")
        end = search.get("END")
        sat = search.get("SAT", "S1")
        user = search.get("ASF_USER")
        pwd = search.get("ASF_PASS")
        npd = int(self.project.config.get("SEARCH_PDOWNLOADS", "NPD", fallback="4"))

        platform = {
            "S1": asf.SENTINEL1,
            "S1A": asf.SENTINEL1A,
            "S1B": asf.SENTINEL1B,
        }.get(sat, asf.SENTINEL1)

        processing = [asf.PRODUCT_TYPE.SLC] if beam_mode == "SLC" else [asf.PRODUCT_TYPE.GRD_HD, asf.PRODUCT_TYPE.GRD_HS]
        secondaries = self.project.project_folder / "secondaries"
        secondaries.mkdir(parents=True, exist_ok=True)

        session = asf.ASFSession()
        session.auth_with_creds(user, pwd)

        results = asf.geo_search(
            platform=[platform],
            beamMode=[asf.IW],
            processingLevel=processing,
            relativeOrbit=track,
            intersectsWith=self._resolve_aoi_wkt(),
            start=start,
            end=end,
            maxResults=1000,
        )
        results.download(path=secondaries.as_posix(), session=session, processes=npd)
        return len(results)

    def run_topsar_auto(self, *, run_plots: bool = True, master_mode: Optional[str] = None) -> None:
        if self.project.auto_download == "Y":
            self.download_asf_s1()

        self.prepare_topsar_secondaries()
        if (master_mode or self.project.master_selection).upper() != "MANUAL":
            self.select_topsar_master(mode=master_mode)

        self.run_topsar_split_master()
        self.run_topsar_split_secondaries()
        self.run_topsar_coreg_ifg()
        if run_plots and self.project.plotting == "Y":
            self.run_topsar_plotting("ifg")
            self.run_topsar_plotting("coreg")
        self.run_topsar_export()

    def run_stripmap_unpack(self) -> None:
        project = self.project.project_folder
        zipfolder = project / "zipfiles"
        secondaries = project / "secondaries"
        zipfolder.mkdir(parents=True, exist_ok=True)
        secondaries.mkdir(parents=True, exist_ok=True)

        for filename in sorted(zipfolder.iterdir()):
            if not filename.is_file() or not (filename.name.endswith(".tar") or filename.name.endswith(".tar.gz")):
                continue
            mode = "r:gz" if filename.name.endswith(".tar.gz") else "r:"
            with tarfile.open(filename, mode) as tar:
                tar.extractall(path=zipfolder)

        for folder in sorted(zipfolder.iterdir()):
            if not folder.is_dir():
                continue
            for tsx_main in folder.glob("*.L1B"):
                for sub in tsx_main.iterdir():
                    shutil.move(sub.as_posix(), (secondaries / sub.name).as_posix())

    def run_stripmap_prepare_secondaries(self) -> None:
        secondaries = self.project.project_folder / "secondaries"
        for item in sorted(secondaries.iterdir()):
            n = item.name
            if n.startswith("TDX1_SAR__SSC______SM_S_SRA") or n.startswith("TSX1_SAR__SSC______SM_S_SRA"):
                date = n[28:36]
                out = secondaries / date
                out.mkdir(parents=True, exist_ok=True)
                shutil.move(item.as_posix(), (out / n).as_posix())

    def run_stripmap_subset(self) -> list[Path]:
        project = self.project.project_folder
        secondaries = project / "secondaries"
        subset = project / "subset"
        subset.mkdir(parents=True, exist_ok=True)

        polygon = self._resolve_aoi_wkt()
        outputs: list[Path] = []

        xmlfiles = sorted(secondaries.glob("*/*/*.xml"))
        if not xmlfiles:
            return outputs

        anchor = xmlfiles[0]
        for xml in xmlfiles:
            out_name = f"{xml.name[28:36]}_sub.dim"
            out_path = subset / out_name
            repl = {
                "INPUTXML": xml.as_posix(),
                "OUTPUTSUBSETFOLDER": subset.as_posix(),
                "OUTPUTFILE": out_name,
                "POLYGON": polygon,
            }
            self.run_graph_pipeline(
                graph_name="stripmap_tsx_subset",
                replacements=repl,
                anchor_product=anchor,
                output_path=out_path,
            )
            outputs.append(out_path)

        return outputs

    def run_stripmap_manual_master_selection(self, date_yyyymmdd: str) -> int:
        subset = self.project.project_folder / "subset"
        master = self.project.project_folder / "master"
        master.mkdir(parents=True, exist_ok=True)

        moved = 0
        for item in sorted(subset.iterdir()):
            if date_yyyymmdd in item.name:
                shutil.move(item.as_posix(), (master / item.name).as_posix())
                moved += 1
        return moved

    def _resolve_stripmap_master_dim(self) -> Path:
        m = self.project.master_path
        if m.is_file() and m.suffix == ".dim":
            return m
        candidates = sorted(m.glob("*.dim"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No stripmap master .dim found at {m}")

    def run_stripmap_coreg(self) -> list[Path]:
        subset = self.project.project_folder / "subset"
        coreg = self.project.project_folder / "coreg"
        coreg.mkdir(parents=True, exist_ok=True)
        master_dim = self._resolve_stripmap_master_dim()

        graph = "stripmap_dem_assisted_coregistration"
        if self.project.ext_dem:
            graph = "stripmap_dem_assisted_coregistration_extdem"

        outputs: list[Path] = []
        for dim in sorted(subset.glob("*.dim")):
            out_name = f"{master_dim.stem[0:8]}_{dim.stem[0:8]}.dim"
            out_path = coreg / out_name
            repl = {
                "MASTER": master_dim.as_posix(),
                "SECONDARY": dim.as_posix(),
                "EXTERNALDEM": self.project.ext_dem,
                "OUTPUTCOREGFOLDER": coreg.as_posix(),
                "OUTPUTFILE": out_name,
            }
            self.run_graph_pipeline(
                graph_name=graph,
                replacements=repl,
                anchor_product=master_dim,
                output_path=out_path,
            )
            outputs.append(out_path)
        return outputs

    def run_stripmap_ifg(self) -> list[Path]:
        coreg = self.project.project_folder / "coreg"
        ifg = self.project.project_folder / "ifg"
        ifg.mkdir(parents=True, exist_ok=True)

        graph = "stripmap_interferogram_topophase"
        if self.project.ext_dem:
            graph = "stripmap_interferogram_topophase_extdem"

        outputs: list[Path] = []
        for dim in sorted(coreg.glob("*.dim")):
            out_name = f"{dim.stem[0:17]}.dim"
            out_path = ifg / out_name
            repl = {
                "COREGFILE": dim.as_posix(),
                "EXTERNALDEM": self.project.ext_dem,
                "OUTPUTIFGFOLDER": ifg.as_posix(),
                "OUTPUTFILE": out_name,
            }
            self.run_graph_pipeline(
                graph_name=graph,
                replacements=repl,
                anchor_product=dim,
                output_path=out_path,
            )
            outputs.append(out_path)

        return outputs

    def run_stripmap_plotting(self, mode: str) -> list[Path]:
        mode = mode.lower()
        if mode not in {"split", "coreg", "ifg"}:
            raise ValueError("mode must be split/coreg/ifg")

        project = self.project.project_folder
        plot_folder = project / "plot" / mode
        plot_folder.mkdir(parents=True, exist_ok=True)

        if mode == "split":
            graph_name = "stripmap_plot_split"
            search = project / "subset"
            i_pattern = f"i_*{self.project.polarisation}*.img"
            q_pattern = f"q_*{self.project.polarisation}*.img"
        elif mode == "coreg":
            graph_name = "stripmap_plot_coreg"
            search = project / "coreg"
            i_pattern = f"i_*{self.project.polarisation}_slv*.img"
            q_pattern = f"q_*{self.project.polarisation}_slv*.img"
        else:
            graph_name = "stripmap_plot_ifg"
            search = project / "ifg"
            i_pattern = f"i_ifg_*{self.project.polarisation}*.img"
            q_pattern = f"q_ifg_*{self.project.polarisation}*.img"

        outputs: list[Path] = []
        dims = sorted(search.glob("*.dim"))
        if not dims:
            return outputs

        for dim in dims:
            out_name = f"{dim.stem}_{mode}"
            out_dim = plot_folder / f"{out_name}.dim"
            data_dir = search / f"{dim.stem}.data"
            iband, qband = self._discover_complex_bands(data_dir, i_pattern, q_pattern)

            repl = {
                "INPUTFILE": dim.as_posix(),
                "OUTPUTFOLDER": plot_folder.as_posix(),
                "OUTPUTNAME": out_name,
                "IBAND": iband,
                "QBAND": qband,
            }
            self.run_graph_pipeline(
                graph_name=graph_name,
                replacements=repl,
                anchor_product=dim,
                output_path=out_dim,
            )
            outputs.append(out_dim)
            self._run_pconvert(input_dim=out_dim, output_folder=plot_folder, mode=mode)

        return outputs

    def run_stripmap_export(self) -> list[Path]:
        project = self.project.project_folder
        coreg = project / "coreg"
        ifg = project / "ifg"
        master = self._resolve_stripmap_master_dim()
        export_dir = project / f"INSAR_{master.stem[0:8]}"
        export_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []
        for dim in sorted(coreg.glob("*.dim")):
            ifg_dim = ifg / f"{dim.stem}.dim"
            if not ifg_dim.exists():
                ifg_dim = ifg / f"{dim.stem}"
            repl = {
                "COREGFILE": dim.as_posix(),
                "IFGFILE": ifg_dim.as_posix(),
                "OUTPUTFOLDER": export_dir.as_posix(),
            }
            out_marker = export_dir / f"{dim.stem}.dim"
            self.run_graph_pipeline(
                graph_name="stripmap_export",
                replacements=repl,
                anchor_product=dim,
                output_path=out_marker,
            )
            outputs.append(out_marker)

        return outputs

    def run_stripmap_auto(self, *, run_plots: bool = True, master_date: Optional[str] = None) -> None:
        self.run_stripmap_unpack()
        self.run_stripmap_prepare_secondaries()
        self.run_stripmap_subset()
        if master_date:
            self.run_stripmap_manual_master_selection(master_date)
        self.run_stripmap_coreg()
        self.run_stripmap_ifg()
        if run_plots and self.project.plotting == "Y":
            self.run_stripmap_plotting("split")
            self.run_stripmap_plotting("ifg")
            self.run_stripmap_plotting("coreg")
        self.run_stripmap_export()

    def run_auto(self, *, master_mode: Optional[str] = None, stripmap_master_date: Optional[str] = None) -> None:
        sensor = self.project.sensor
        if sensor == "S1":
            self.run_topsar_auto(master_mode=master_mode)
            return
        if sensor in {"TSX", "TDX"}:
            self.run_stripmap_auto(master_date=stripmap_master_date)
            return
        raise ValueError(f"Unsupported sensor for auto mode: {sensor}")


def run_graph_pipeline(
    gpt: GPT,
    graph_name: str,
    *,
    graph_dir: str | Path,
    replacements: dict[str, Any],
    output_path: str | Path,
    generated_graph_path: Optional[str | Path] = None,
) -> str:
    """Execute one snap2stamps graph template with placeholder replacements."""
    graph_dir = Path(graph_dir)
    template_name = GRAPH_STEM_TO_TEMPLATE.get(graph_name, graph_name)
    template_path = graph_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Graph template not found: {template_path}")

    content = template_path.read_text(encoding="utf-8")
    for key in sorted(replacements, key=len, reverse=True):
        content = content.replace(key, str(replacements[key]))

    if generated_graph_path is None:
        generated_graph_path = gpt.outdir / "graphs" / f"{Path(template_name).stem}_2run.xml"
    generated_graph_path = Path(generated_graph_path)
    generated_graph_path.parent.mkdir(parents=True, exist_ok=True)
    generated_graph_path.write_text(content, encoding="utf-8")

    result = gpt.run_graph(graph_path=generated_graph_path, output_path=output_path)
    if result is None:
        raise RuntimeError(f"Graph '{graph_name}' failed: {gpt.last_error_summary()}")
    return result


def run_workflow(
    gpt: GPT,
    workflow: str,
    *,
    graph_dir: str | Path,
    replacements_by_graph: Optional[dict[str, dict[str, Any]]] = None,
    output_path_by_graph: Optional[dict[str, str | Path]] = None,
) -> str:
    """Run a simple composed workflow of graph templates.

    For full parity automation, prefer :class:`Snap2StampsRunner`.
    """
    try:
        graph_names = SNAP2STAMPS_WORKFLOWS[workflow]
    except KeyError as exc:
        available = ", ".join(sorted(SNAP2STAMPS_WORKFLOWS))
        raise KeyError(f"Unknown workflow '{workflow}'. Available: {available}") from exc

    last_output = gpt.prod_path.as_posix()
    for graph_name in graph_names:
        replacements = (replacements_by_graph or {}).get(graph_name, {})
        output_path = (output_path_by_graph or {}).get(graph_name, gpt.outdir / f"{graph_name}.dim")
        last_output = run_graph_pipeline(
            gpt=gpt,
            graph_name=graph_name,
            graph_dir=graph_dir,
            replacements=replacements,
            output_path=output_path,
        )
    return last_output


def build_gpt(
    product: str | Path,
    outdir: str | Path,
    *,
    format: str = "BEAM-DIMAP",
    gpt_path: str = "/usr/local/snap/bin/gpt",
    memory: str | None = None,
    parallelism: int | None = 14,
    timeout: int | None = 7200,
    snap_userdir: str | Path | None = None,
) -> GPT:
    """Convenience constructor aligned with snap2stamps-like batch processing."""
    return GPT(
        product=product,
        outdir=outdir,
        format=format,
        gpt_path=gpt_path,
        memory=memory,
        parallelism=parallelism,
        timeout=timeout,
        snap_userdir=snap_userdir,
    )
