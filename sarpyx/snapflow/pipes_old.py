import pandas as pd
from pathlib import Path

from sarpyx.snapflow.engine import GPT
from sarpyx.snapflow.utils import delProd, mode_identifier


def _process_product_cfar(product_path: Path, mask_shp_path: Path, gpt_mode: str | None, 
                         delete_intermediate: bool, pfa_thresholds: list[float]):
    """Helper function to process a single product through the CFAR chain."""
    out_dir = product_path.parent
    try:
        prod_type = mode_identifier(product_path.name)
    except Exception as e:
        print(f"Error determining product type for {product_path}: {e}")
        return None, None

    op = GPT(product=product_path.as_posix(), outdir=out_dir.as_posix(), mode=gpt_mode)
    op.prod_type = prod_type

    processed_products = []
    prod_start_cfar = product_path.as_posix()

    # Process based on product type
    if prod_type == "Sentinel-1":
        prod_deb = op.Deburst()
        if not prod_deb:
            return None, None
        processed_products.append(Path(prod_deb))
        
        prod_cal = op.Calibration(Pols=['VH'])
        if not prod_cal:
            return None, None
        processed_products.append(Path(prod_cal))
        
        prod_shp = op.ImportVector(vector_data=mask_shp_path)
        if not prod_shp:
            return None, None
        processed_products.append(Path(prod_shp))
        
        prod_lm = op.LandMask()
        if not prod_lm:
            return None, None
        processed_products.append(Path(prod_lm))
        prod_start_cfar = prod_lm
        
        if delete_intermediate:
            delProd(processed_products[1])
            delProd(processed_products[2])

    elif prod_type == "COSMO-SkyMed":
        prod_ml = op.Multilook(nRgLooks=2, nAzLooks=2)
        if not prod_ml:
            return None, None
        processed_products.append(Path(prod_ml))
        
        prod_cal = op.Calibration(Pols=['HH'])
        if not prod_cal:
            return None, None
        processed_products.append(Path(prod_cal))
        
        prod_shp = op.ImportVector(vector_data=mask_shp_path)
        if not prod_shp:
            return None, None
        processed_products.append(Path(prod_shp))
        
        prod_lm = op.LandMask()
        if not prod_lm:
            return None, None
        processed_products.append(Path(prod_lm))
        prod_start_cfar = prod_lm
        
        if delete_intermediate:
            delProd(processed_products[0])
            delProd(processed_products[1])
            delProd(processed_products[2])

    elif prod_type == "SAOCOM":
        prod_shp = op.ImportVector(vector_data=mask_shp_path)
        if not prod_shp:
            return None, None
        processed_products.append(Path(prod_shp))
        
        prod_lm = op.LandMask()
        if not prod_lm:
            return None, None
        processed_products.append(Path(prod_lm))
        prod_start_cfar = prod_lm
        
        if delete_intermediate:
            delProd(processed_products[0])

    # Process CFAR for each PFA threshold
    last_successful_excel = None
    for pfa in pfa_thresholds:
        op_cfar = GPT(product=prod_start_cfar, outdir=out_dir.as_posix(), mode=gpt_mode)

        at_params = {'pfa': pfa}
        if prod_type == "COSMO-SkyMed":
            at_params.update({'background_window_m': 650, 'guard_window_m': 400, 'target_window_m': 25})

        prod_at = op_cfar.AdaptiveThresholding(**at_params)
        if not prod_at:
            continue

        prod_od = op_cfar.ObjectDiscrimination(min_target_m=35, max_target_m=500)
        if not prod_od:
            if delete_intermediate:
                delProd(prod_at)
            continue

        prod_od_path = Path(prod_od)
        prod_od_data_dir = prod_od_path.with_suffix('.data')

        try:
            csv_files = list(prod_od_data_dir.glob('*.csv'))
            ship_csv_path = next((f for f in csv_files if f.stem.lower().startswith('ship')), None)

            if ship_csv_path:
                ship_detections_df = pd.read_csv(ship_csv_path, header=1, sep='\t')
                out_excel_path = out_dir / f"{product_path.stem}_pfa_{pfa}.xlsx"
                ship_detections_df.to_excel(out_excel_path, index=False)
                print(f'Saved ExcelFile to: {out_excel_path}')
                last_successful_excel = out_excel_path.as_posix()
            else:
                print(f"No Ship detection CSV found for PFA {pfa}")

        except Exception as e:
            print(f"Error processing detection results for PFA {pfa}: {e}")
        finally:
            if delete_intermediate:
                delProd(prod_at)
                delProd(prod_od)

    if delete_intermediate and Path(prod_start_cfar).exists() and prod_start_cfar != product_path.as_posix():
        delProd(prod_start_cfar)

    first_processed = processed_products[0].as_posix() if processed_products else product_path.as_posix()
    return first_processed, last_successful_excel


def CFAR(prod: str | Path, mask_shp_path: str | Path, mode: str | None = None, 
         Thresh: list[float] | float = 12.5, DELETE: bool = False):
    """
    Performs Constant False Alarm Rate (CFAR) ship detection processing chain.

    Args:
        prod: Path to the input SAR product file
        mask_shp_path: Path to the shapefile used for land masking
        mode: OS mode ('MacOS', 'Ubuntu', None) for GPT configuration
        Thresh: A single PFA threshold or a list of PFA thresholds to test
        DELETE: If True, delete intermediate processing files

    Returns:
        Tuple[str | None, str | None]: Path to the first major processed product 
                                       and path to the last generated Excel file
    """
    product_path = Path(prod)
    mask_path = Path(mask_shp_path)

    if isinstance(Thresh, (int, float)):
        pfa_thresholds = [float(Thresh)]
    elif isinstance(Thresh, list):
        pfa_thresholds = Thresh
    else:
        print("Warning: Invalid type for Thresh, using default [12.5]")
        pfa_thresholds = [12.5]

    return _process_product_cfar(
        product_path=product_path,
        mask_shp_path=mask_path,
        gpt_mode=mode,
        delete_intermediate=DELETE,
        pfa_thresholds=pfa_thresholds
    )
# ================================ EXAMPLE USAGE ==================================