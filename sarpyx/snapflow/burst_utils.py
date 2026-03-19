"""Helpers for burst-based SNAPFlow notebook workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


REQUIRED_PAIR_COLUMNS = (
    "Id",
    "BurstId",
    "SwathIdentifier",
    "RelativeOrbitNumber",
    "OrbitDirection",
    "PolarisationChannels",
)


@dataclass(frozen=True)
class BurstAcquisition:
    """Single burst acquisition selected for notebook download/processing."""

    id: str
    burst_id: int | None
    swath_identifier: str | None
    relative_orbit_number: int | None
    orbit_direction: str | None
    polarisation_channels: str | None
    parent_product_name: str | None
    content_start: datetime
    coverage: float | None


@dataclass(frozen=True)
class BurstPairSelection:
    """Resolved master/slave burst pair."""

    master: BurstAcquisition
    slave: BurstAcquisition


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _content_start(row: pd.Series) -> datetime:
    raw = row.get("ContentDate")
    if isinstance(raw, dict):
        raw = raw.get("Start")
    if raw is None:
        raw = row.get("OriginDate")
    if raw is None:
        raise ValueError("Rows must include ContentDate/Start or OriginDate")
    return pd.to_datetime(raw, utc=True).to_pydatetime()


def _coverage_value(row: pd.Series) -> float | None:
    value = pd.to_numeric(row.get("coverage"), errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def _normalized_key(row: pd.Series) -> tuple[object, ...]:
    return tuple(row.get(column) for column in REQUIRED_PAIR_COLUMNS[1:])


def _row_to_acquisition(row: pd.Series) -> BurstAcquisition:
    return BurstAcquisition(
        id=str(row["Id"]),
        burst_id=int(row["BurstId"]) if pd.notna(row["BurstId"]) else None,
        swath_identifier=str(row["SwathIdentifier"]) if pd.notna(row["SwathIdentifier"]) else None,
        relative_orbit_number=int(row["RelativeOrbitNumber"]) if pd.notna(row["RelativeOrbitNumber"]) else None,
        orbit_direction=str(row["OrbitDirection"]) if pd.notna(row["OrbitDirection"]) else None,
        polarisation_channels=(
            str(row["PolarisationChannels"]) if pd.notna(row["PolarisationChannels"]) else None
        ),
        parent_product_name=(
            str(row["ParentProductName"]) if pd.notna(row.get("ParentProductName")) else None
        ),
        content_start=_content_start(row),
        coverage=_coverage_value(row),
    )


def _validate_pair_rows(master_row: pd.Series, slave_row: pd.Series) -> None:
    if _normalized_key(master_row) != _normalized_key(slave_row):
        raise ValueError(
            "Selected master/slave rows do not share BurstId, SwathIdentifier, "
            "RelativeOrbitNumber, OrbitDirection, and PolarisationChannels"
        )


def select_burst_pair(
    df: pd.DataFrame,
    *,
    master_id: str | None = None,
    slave_id: str | None = None,
) -> BurstPairSelection:
    """Resolve a burst pair from burst-search results."""

    _require_columns(df, REQUIRED_PAIR_COLUMNS)
    if "ContentDate" not in df.columns and "OriginDate" not in df.columns:
        raise ValueError("Rows must include ContentDate or OriginDate")
    if df.empty:
        raise ValueError("Cannot select a burst pair from an empty DataFrame")

    work = df.copy()

    if master_id or slave_id:
        if not master_id or not slave_id:
            raise ValueError("master_id and slave_id must be provided together")
        try:
            master_row = work.loc[work["Id"] == master_id].iloc[0]
            slave_row = work.loc[work["Id"] == slave_id].iloc[0]
        except IndexError as exc:
            raise ValueError("Explicit master/slave Ids must exist in the DataFrame") from exc
        _validate_pair_rows(master_row, slave_row)
        return BurstPairSelection(
            master=_row_to_acquisition(master_row),
            slave=_row_to_acquisition(slave_row),
        )

    work["_content_start"] = work.apply(_content_start, axis=1)
    work["_coverage_num"] = pd.to_numeric(work.get("coverage"), errors="coerce")
    group_columns = list(REQUIRED_PAIR_COLUMNS[1:])

    ranked_groups: list[tuple[float, datetime, pd.DataFrame]] = []
    for _, group in work.groupby(group_columns, dropna=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values("_content_start", ascending=False).reset_index(drop=True)
        max_coverage = ordered["_coverage_num"].dropna().max()
        ranked_groups.append(
            (
                float(max_coverage) if pd.notna(max_coverage) else float("-inf"),
                ordered.loc[0, "_content_start"],
                ordered,
            )
        )

    if not ranked_groups:
        raise ValueError("Could not resolve a burst pair with at least two compatible acquisitions")

    ranked_groups.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = ranked_groups[0][2]
    master_row = selected.iloc[0]
    slave_row = selected.iloc[1]
    return BurstPairSelection(
        master=_row_to_acquisition(master_row),
        slave=_row_to_acquisition(slave_row),
    )


def extract_burst_archive(archive_path: str | Path, extract_root: str | Path) -> Path:
    """Extract a burst archive and return the single SAFE product root."""

    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive path does not exist: {archive}")
    target_dir = Path(extract_root) / archive.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_dirs = sorted(path for path in target_dir.rglob("*.SAFE") if path.is_dir())
    if not safe_dirs:
        try:
            with ZipFile(archive) as zf:
                for member in zf.namelist():
                    destination = (target_dir / member).resolve(strict=False)
                    root = target_dir.resolve(strict=False)
                    if destination != root and root not in destination.parents:
                        raise ValueError(f"Unsafe archive member path: {member}")
                    zf.extract(member, target_dir)
        except BadZipFile as exc:
            raise ValueError(f"Archive is not a valid ZIP file: {archive}") from exc
        safe_dirs = sorted(path for path in target_dir.rglob("*.SAFE") if path.is_dir())

    if len(safe_dirs) != 1:
        raise ValueError(
            f"Expected exactly one extracted .SAFE root in {target_dir}, found {len(safe_dirs)}"
        )
    return safe_dirs[0]
