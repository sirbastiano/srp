#!/usr/bin/env python3
"""Download Sentinel-1 POEORB orbit files into ./PEORB.

Mirrors the ESA STEP directory structure under the output directory:
  PEORB/<SATELLITE>/<YYYY>/<MM>/*.EOF.zip
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable
from urllib import request

BASE_URL = "https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB"
DEFAULT_SATS = ["S1A", "S1B", "S1C", "S1D"]


def _fetch_html(url: str, timeout: int) -> str:
    with request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _list_years(sat_url: str, timeout: int) -> list[str]:
    html = _fetch_html(sat_url, timeout)
    years = sorted({y for y in re.findall(r'href="(\d{4})/"', html)})
    return years


def _list_months(year_url: str, timeout: int) -> list[str]:
    html = _fetch_html(year_url, timeout)
    months = sorted({m for m in re.findall(r'href="(\d{2})/"', html)})
    return [m for m in months if 1 <= int(m) <= 12]


def _list_files(month_url: str, timeout: int) -> list[str]:
    html = _fetch_html(month_url, timeout)
    files = re.findall(r'href="([^\"]+\.EOF\.zip)"', html, flags=re.IGNORECASE)
    return sorted(set(files))


def _parse_years(years_str: str | None) -> list[str]:
    if not years_str:
        return []
    years: set[int] = set()
    for part in re.split(r"[,\s]+", years_str.strip()):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            for year in range(min(start, end), max(start, end) + 1):
                years.add(year)
        else:
            years.add(int(part))
    return [f"{year:04d}" for year in sorted(years)]


def _parse_months(months_str: str | None) -> list[str]:
    if not months_str:
        return []
    months: set[int] = set()
    for part in re.split(r"[,\s]+", months_str.strip()):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            for month in range(min(start, end), max(start, end) + 1):
                if 1 <= month <= 12:
                    months.add(month)
        else:
            month = int(part)
            if 1 <= month <= 12:
                months.add(month)
    return [f"{month:02d}" for month in sorted(months)]


def _parse_sats(sats_str: str | None) -> list[str]:
    if not sats_str:
        return DEFAULT_SATS
    return [item.strip().upper() for item in sats_str.split(",") if item.strip()]


def _download(url: str, dest_path: Path, timeout: int) -> None:
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with request.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as fh:
        fh.write(response.read())
    tmp_path.replace(dest_path)


def _iter_targets(
    base_url: str,
    satellites: Iterable[str],
    years_filter: list[str],
    months_filter: list[str],
    timeout: int,
) -> Iterable[tuple[str, str, str, str]]:
    for sat in satellites:
        sat_url = f"{base_url}/{sat}/"
        try:
            years = _list_years(sat_url, timeout)
        except Exception as exc:
            print(f"Warning: failed to list years for {sat_url}: {exc}")
            continue
        if years_filter:
            years = [y for y in years if y in years_filter]
        for year in years:
            year_url = f"{sat_url}{year}/"
            try:
                months = _list_months(year_url, timeout)
            except Exception as exc:
                print(f"Warning: failed to list months for {year_url}: {exc}")
                continue
            if months_filter:
                months = [m for m in months if m in months_filter]
            for month in months:
                month_url = f"{year_url}{month}/"
                try:
                    files = _list_files(month_url, timeout)
                except Exception as exc:
                    print(f"Warning: failed to list files for {month_url}: {exc}")
                    continue
                for fname in files:
                    yield sat, year, month, f"{month_url}{fname}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 POEORB orbit files into ./PEORB",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent / "PEORB"),
        help="Output directory (default: ./PEORB)",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"Base URL (default: {BASE_URL})",
    )
    parser.add_argument(
        "--satellites",
        default=",".join(DEFAULT_SATS),
        help="Comma-separated satellites (default: S1A,S1B,S1C,S1D)",
    )
    parser.add_argument(
        "--years",
        default="",
        help='Years to include, e.g. "2024,2025" or "2020-2026" (default: all found)',
    )
    parser.add_argument(
        "--months",
        default="",
        help='Months to include, e.g. "1,2,12" or "01-06" (default: all found)',
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without downloading",
    )
    parser.add_argument(
        "--report-missing",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write missing file list to a report. If no path is provided, "
            "defaults to <outdir>/missing_orbits.txt."
        ),
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    satellites = _parse_sats(args.satellites)
    years_filter = _parse_years(args.years)
    months_filter = _parse_months(args.months)

    outdir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    missing: list[str] = []

    for sat, year, month, url in _iter_targets(
        args.base_url,
        satellites,
        years_filter,
        months_filter,
        args.timeout,
    ):
        fname = url.rsplit("/", 1)[-1]
        dest_dir = outdir / sat / year / month
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fname
        if dest_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            print(f"[DRY-RUN] {url} -> {dest_path}")
            missing.append(f"{url} -> {dest_path}")
            continue
        if args.report_missing is not None:
            missing.append(f"{url} -> {dest_path}")
        try:
            print(f"Downloading {fname} -> {dest_path}")
            _download(url, dest_path, args.timeout)
            downloaded += 1
        except Exception as exc:
            print(f"Warning: failed to download {url}: {exc}")

    if args.report_missing is not None:
        report_path = (
            Path(args.report_missing)
            if args.report_missing
            else outdir / "missing_orbits.txt"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(missing) + ("\n" if missing else ""))
        print(f"Wrote missing report: {report_path}")

    print(f"Done. Downloaded: {downloaded}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
