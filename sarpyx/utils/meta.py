# =====================================================================
# Sentinel-1 core metadata extraction
# =====================================================================
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

    return {
        k: _decode(md.get(k))
        for k in keys
        if k in md
    }