# =====================================================================
# Sentinel-1 core metadata extraction
# =====================================================================
import re
from datetime import datetime, timezone


def normalize_sar_timestamp(value) -> str | None:
    """Normalize mixed SAR timestamp representations to UTC ISO-8601.

    Output format: YYYY-MM-DDTHH:MM:SS.ffffffZ
    """
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode('utf-8', errors='replace')

    ts = str(value).strip()
    if not ts:
        return None

    # SNAP-style Sentinel-1 timestamp, e.g. 30-JAN-2026 15:26:10.271545
    for fmt in ('%d-%b-%Y %H:%M:%S.%f', '%d-%b-%Y %H:%M:%S'):
        try:
            dt = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            pass

    iso_candidate = ts
    if iso_candidate.endswith('Z'):
        iso_candidate = f'{iso_candidate[:-1]}+00:00'

    # Python datetime supports microseconds; trim longer fractional parts if present.
    match = re.match(r'^(.*T\d{2}:\d{2}:\d{2})\.(\d+)(.*)$', iso_candidate)
    if match:
        frac = match.group(2)[:6].ljust(6, '0')
        iso_candidate = f'{match.group(1)}.{frac}{match.group(3)}'

    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        # Preserve original value if parsing fails.
        return ts


def extract_core_metadata_sentinel(md: dict) -> dict:
    """
    Extract a minimal, cross-mission-relevant SAR metadata subset
    for geospatial foundation models.

    Args:
        md (dict): Metadata dictionary containing SAR metadata.

    Returns:
        dict: A dictionary containing the extracted metadata subset with the following keys:
            - MISSION
            - ACQUISITION_MODE
            - PRODUCT_TYPE
            - radar_frequency
            - pulse_repetition_frequency
            - range_spacing
            - azimuth_spacing
            - range_bandwidth
            - azimuth_bandwidth
            - PASS
            - avg_scene_height
    """
    def _decode(v):
        # SNAP often stores strings as bytes
        if isinstance(v, (bytes, bytearray)):
            return v.decode('utf-8')
        return v

    keys = [
        'MISSION',
        'ACQUISITION_MODE',
        'PRODUCT_TYPE',
        'radar_frequency',
        'pulse_repetition_frequency',
        'range_spacing',
        'azimuth_spacing',
        'range_bandwidth',
        'azimuth_bandwidth',
        'antenna_pointing',
        'PASS',
        'avg_scene_height',
        'PRODUCT',
        'mds1_tx_rx_polar',
        'mds2_tx_rx_polar',
        'first_line_time',
    ]

    out = {}
    for k in keys:
        if k not in md:
            continue
        value = _decode(md.get(k))
        if k == 'first_line_time':
            value = normalize_sar_timestamp(value)
        out[k] = value
    return out