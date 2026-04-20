from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from sarpyx.snapflow.burst_utils import extract_burst_archive, select_burst_pair


def _content_date(start: str) -> dict[str, str]:
    return {"Start": start}


def test_select_burst_pair_chooses_best_group_by_coverage_then_recency():
    df = pd.DataFrame(
        [
            {
                "Id": "group-a-1",
                "BurstId": 100,
                "SwathIdentifier": "IW1",
                "RelativeOrbitNumber": 8,
                "OrbitDirection": "DESCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "A1.SAFE",
                "ContentDate": _content_date("2025-02-01T00:00:00Z"),
                "coverage": 55.0,
            },
            {
                "Id": "group-a-2",
                "BurstId": 100,
                "SwathIdentifier": "IW1",
                "RelativeOrbitNumber": 8,
                "OrbitDirection": "DESCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "A2.SAFE",
                "ContentDate": _content_date("2025-02-13T00:00:00Z"),
                "coverage": 65.0,
            },
            {
                "Id": "group-b-1",
                "BurstId": 200,
                "SwathIdentifier": "IW2",
                "RelativeOrbitNumber": 9,
                "OrbitDirection": "ASCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "B1.SAFE",
                "ContentDate": _content_date("2025-02-02T00:00:00Z"),
                "coverage": 80.0,
            },
            {
                "Id": "group-b-2",
                "BurstId": 200,
                "SwathIdentifier": "IW2",
                "RelativeOrbitNumber": 9,
                "OrbitDirection": "ASCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "B2.SAFE",
                "ContentDate": _content_date("2025-02-20T00:00:00Z"),
                "coverage": 90.0,
            },
            {
                "Id": "group-b-3",
                "BurstId": 200,
                "SwathIdentifier": "IW2",
                "RelativeOrbitNumber": 9,
                "OrbitDirection": "ASCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "B3.SAFE",
                "ContentDate": _content_date("2025-03-01T00:00:00Z"),
                "coverage": 85.0,
            },
        ]
    )

    pair = select_burst_pair(df)

    assert pair.master.id == "group-b-3"
    assert pair.slave.id == "group-b-2"
    assert pair.master.burst_id == 200


def test_select_burst_pair_respects_explicit_override_order():
    df = pd.DataFrame(
        [
            {
                "Id": "master-uuid",
                "BurstId": 100,
                "SwathIdentifier": "IW1",
                "RelativeOrbitNumber": 8,
                "OrbitDirection": "DESCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "A1.SAFE",
                "ContentDate": _content_date("2025-02-01T00:00:00Z"),
                "coverage": 55.0,
            },
            {
                "Id": "slave-uuid",
                "BurstId": 100,
                "SwathIdentifier": "IW1",
                "RelativeOrbitNumber": 8,
                "OrbitDirection": "DESCENDING",
                "PolarisationChannels": "VV",
                "ParentProductName": "A2.SAFE",
                "ContentDate": _content_date("2025-02-13T00:00:00Z"),
                "coverage": 65.0,
            },
        ]
    )

    pair = select_burst_pair(df, master_id="master-uuid", slave_id="slave-uuid")

    assert pair.master.id == "master-uuid"
    assert pair.slave.id == "slave-uuid"


def test_select_burst_pair_rejects_mismatched_override_rows():
    df = pd.DataFrame(
        [
            {
                "Id": "master-uuid",
                "BurstId": 100,
                "SwathIdentifier": "IW1",
                "RelativeOrbitNumber": 8,
                "OrbitDirection": "DESCENDING",
                "PolarisationChannels": "VV",
                "ContentDate": _content_date("2025-02-01T00:00:00Z"),
            },
            {
                "Id": "slave-uuid",
                "BurstId": 101,
                "SwathIdentifier": "IW2",
                "RelativeOrbitNumber": 8,
                "OrbitDirection": "DESCENDING",
                "PolarisationChannels": "VV",
                "ContentDate": _content_date("2025-02-13T00:00:00Z"),
            },
        ]
    )

    with pytest.raises(ValueError, match="do not share"):
        select_burst_pair(df, master_id="master-uuid", slave_id="slave-uuid")


def test_extract_burst_archive_returns_single_safe_root(tmp_path: Path):
    archive = tmp_path / "burst.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("product.SAFE/manifest.safe", "content")
        zf.writestr("product.SAFE/measurement/file.txt", "content")

    safe_root = extract_burst_archive(archive, tmp_path / "extract")

    assert safe_root.name == "product.SAFE"
    assert safe_root.is_dir()


def test_extract_burst_archive_rejects_missing_safe_root(tmp_path: Path):
    archive = tmp_path / "burst.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("not-a-safe/file.txt", "content")

    with pytest.raises(ValueError, match="exactly one extracted .SAFE root"):
        extract_burst_archive(archive, tmp_path / "extract")


def test_extract_burst_archive_rejects_multiple_safe_roots(tmp_path: Path):
    archive = tmp_path / "burst.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("first.SAFE/manifest.safe", "content")
        zf.writestr("second.SAFE/manifest.safe", "content")

    with pytest.raises(ValueError, match="exactly one extracted .SAFE root"):
        extract_burst_archive(archive, tmp_path / "extract")
