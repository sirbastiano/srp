import subprocess
import os
from pathlib import Path

# Lista completa degli operatori SNAP
snap_operators = [
    "Aatsr.SST",
    "AATSR.Ungrid", 
    "AdaptiveThresholding",
    "AddElevation",
    "AddLandCover",
    "ALOS-Deskewing",
    "Apply-Orbit-File",
    "Arc.SST",
    "ArviOp",
    "Azimuth-Shift-Estimation-ESD",
    "AzimuthFilter",
    "Back-Geocoding",
    "BandMaths",
    "BandMerge",
    "BandPassFilter",
    "BandsDifferenceOp",
    "BandSelect",
    "BandsExtractorOp",
    "BatchSnaphuUnwrapOp",
    "Bi2Op",
    "Binning",
    "BiOp",
    "Biophysical10mOp",
    "BiophysicalLandsat8Op",
    "BiophysicalOp",
    "c2rcc.landsat8",
    "c2rcc.meris",
    "c2rcc.meris4",
    "c2rcc.modis",
    "c2rcc.msi",
    "c2rcc.olci",
    "c2rcc.seawifs",
    "c2rcc.viirs",
    "Calibration",
    "Change-Detection",
    "ChangeVectorAnalysisOp",
    "CiOp",
    "CloudProb",
    "Coherence",
    "Collocate",
    "Compactpol-Radar-Vegetation-Index",
    "Compute-Slope-Aspect",
    "Convert-Datatype",
    "CoregistrationOp",
    "CP-Decomposition",
    "CP-Simulation",
    "CP-Stokes-Parameters",
    "CreateStack",
    "Cross-Channel-SNR-Correction",
    "Cross-Correlation",
    "CrossResampling",
    "DarkObjectSubtraction",
    "DeburstWSS",
    "DecisionTree",
    "DEM-Assisted-Coregistration",
    "Demodulate",
    "Double-Difference-Interferogram",
    "DviOp",
    "EAP-Phase-Correction",
    "EcostressSwath2GridOp",
    "Ellipsoid-Correction-GG",
    "Ellipsoid-Correction-RD",
    "EMClusterAnalysis",
    "Enhanced-Spectral-Diversity",
    "Faraday-Rotation-Correction",
    "Fill-DEM-Hole",
    "FlhMci",
    "Flip",
    "ForestCoverChangeOp",
    "FUB.Water",
    "FuClassification",
    "GemiOp",
    "Generalized-Radar-Vegetation-Index",
    "GenericRegionMergingOp",
    "GLCM",
    "GndviOp",
    "GoldsteinPhaseFiltering",
    "GRD-Post",
    "HorizontalVerticalMotion",
    "IEM-Hybrid-Inversion",
    "IEM-Multi-Angle-Inversion",
    "IEM-Multi-Pol-Inversion",
    "Image-Filter",
    "Import-Vector",
    "InSAR-Overview",
    "IntegerInterferogram",
    "Interferogram",
    "IonosphericCorrection",
    "IpviOp",
    "IreciOp",
    "KDTree-KNN-Classifier",
    "KMeansClusterAnalysis",
    "KNN-Classifier",
    "Land-Cover-Mask",
    "Land-Sea-Mask",
    "LandWaterMask",
    "LinearToFromdB",
    "Maximum-Likelihood-Classifier",
    "McariOp",
    "Mci.s2",
    "Merge",
    "Meris.Adapt.4To3",
    "Meris.CorrectRadiometry",
    "Meris.N1Patcher",
    "Minimum-Distance-Classifier",
    "MndwiOp",
    "Mosaic",
    "MphChl",
    "Msavi2Op",
    "MsaviOp",
    "MtciOp",
    "Multi-size Mosaic",
    "Multi-Temporal-Speckle-Filter",
    "Multilook",
    "MultiMasterInSAR",
    "MultiMasterStackGenerator",
    "Multitemporal-Compositing",
    "Ndi45Op",
    "NdpiOp",
    "NdtiOp",
    "NdviOp",
    "Ndwi2Op",
    "NdwiOp",
    "Object-Discrimination",
    "Offset-Tracking",
    "Oil-Spill-Clustering",
    "Oil-Spill-Detection",
    "OlciAnomalyFlagging",
    "OlciO2aHarmonisation",
    "OlciSensorHarmonisation",
    "Orientation-Angle-Correction",
    "Oversample",
    "OWTClassification",
    "PCA",
    "PduStitching",
    "PhaseToDisplacement",
    "PhaseToElevation",
    "PhaseToHeight",
    "PixEx",
    "Polarimetric-Classification",
    "Polarimetric-Decomposition",
    "Polarimetric-Matrices",
    "Polarimetric-Parameters",
    "Polarimetric-Speckle-Filter",
    "PpeFiltering",
    "Principal-Components",
    "ProductSet-Reader",
    "PssraOp",
    "PviOp",
    "Quantization",
    "Rad2Refl",
    "Radar-Vegetation-Index",
    "Random-Forest-Classifier",
    "RangeFilter",
    "Raster-To-Vector",
    "RayleighCorrection",
    "REACTIV-Change-Detection",
    "Read",
    "ReflectanceToRadianceOp",
    "ReipOp",
    "Remodulate",
    "RemoteExecutionOp",
    "Remove-GRD-Border-Noise",
    "RemoveAntennaPattern",
    "ReplaceMetadata",
    "Reproject",
    "Resample",
    "RiOp",
    "RPCA-Change-Detection",
    "RviOp",
    "S1-ETAD-Correction",
    "S2repOp",
    "S2Resampling",
    "SAR-Mosaic",
    "SAR-Simulation",
    "SARSim-Terrain-Correction",
    "SaviOp",
    "SetNoDataValue",
    "SliceAssembly",
    "SM-Dielectric-Modeling",
    "SmacOp",
    "SnaphuExport",
    "SnaphuImport",
    "Speckle-Divergence",
    "Speckle-Filter",
    "SpectralAngleMapperOp",
    "SRGR",
    "Stack-Averaging",
    "Stack-Split",
    "StampsExport",
    "StatisticsOp",
    "SubGraph",
    "Subset",
    "Supervised-Wishart-Classification",
    "TemporalPercentile",
    "Terrain-Correction",
    "Terrain-Flattening",
    "Terrain-Mask",
    "ThermalNoiseRemoval",
    "Three-passDInSAR",
    "TileCache",
    "TileWriter",
    "TndviOp",
    "ToolAdapterOp",
    "TopoPhaseRemoval",
    "TOPSAR-Deburst",
    "TOPSAR-DerampDemod",
    "TOPSAR-Merge",
    "TOPSAR-Split",
    "TsaviOp",
    "Undersample",
    "Unmix",
    "Update-Geo-Reference",
    "Warp",
    "WdviOp",
    "Wind-Field-Estimation",
    "Write"
]

print(f"Totale operatori SNAP: {len(snap_operators)}")



# ...existing code...

def generate_operators_help():
    """
    Genera la documentazione help per tutti gli operatori SNAP
    """
    # Crea la directory tmp se non esiste
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    
    # File di output principale
    all_operators_file = tmp_dir / "all_operators_help.txt"
    
    print(f"Generazione documentazione per {len(snap_operators)} operatori...")
    print(f"Output salvato in: {tmp_dir.absolute()}")
    
    # File principale con tutti gli operatori
    with open(all_operators_file, 'w', encoding='utf-8') as f_all:
        f_all.write("SNAP OPERATORS DOCUMENTATION\n")
        f_all.write("=" * 50 + "\n\n")
        
        for i, operator in enumerate(snap_operators, 1):
            print(f"Processando {i}/{len(snap_operators)}: {operator}")
            
            # File individuale per ogni operatore
            safe_name = operator.replace(".", "_").replace("-", "_")
            individual_file = tmp_dir / f"{safe_name}_help.txt"
            
            try:
                # Esegui gpt -h per l'operatore specifico
                result = subprocess.run(
                    ['gpt', '-h', operator],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                help_content = result.stdout if result.stdout else result.stderr
                
                # Scrivi nel file principale
                f_all.write(f"OPERATOR: {operator}\n")
                f_all.write("-" * 40 + "\n")
                f_all.write(help_content)
                f_all.write("\n" + "=" * 50 + "\n\n")
                
                # Scrivi nel file individuale
                with open(individual_file, 'w', encoding='utf-8') as f_ind:
                    f_ind.write(f"OPERATOR: {operator}\n")
                    f_ind.write("-" * 40 + "\n")
                    f_ind.write(help_content)
                
                if result.returncode != 0:
                    print(f"  ⚠️ Warning per {operator}: return code {result.returncode}")
                else:
                    print(f"  ✅ Completato: {operator}")
                    
            except subprocess.TimeoutExpired:
                error_msg = f"TIMEOUT: L'operatore {operator} ha superato i 30 secondi\n"
                print(f"  ⏱️ Timeout: {operator}")
                
                f_all.write(f"OPERATOR: {operator}\n")
                f_all.write("-" * 40 + "\n")
                f_all.write(error_msg)
                f_all.write("\n" + "=" * 50 + "\n\n")
                
                with open(individual_file, 'w', encoding='utf-8') as f_ind:
                    f_ind.write(f"OPERATOR: {operator}\n")
                    f_ind.write("-" * 40 + "\n")
                    f_ind.write(error_msg)
                    
            except Exception as e:
                error_msg = f"ERRORE: {str(e)}\n"
                print(f"  ❌ Errore per {operator}: {e}")
                
                f_all.write(f"OPERATOR: {operator}\n")
                f_all.write("-" * 40 + "\n")
                f_all.write(error_msg)
                f_all.write("\n" + "=" * 50 + "\n\n")
                
                with open(individual_file, 'w', encoding='utf-8') as f_ind:
                    f_ind.write(f"OPERATOR: {operator}\n")
                    f_ind.write("-" * 40 + "\n")
                    f_ind.write(error_msg)
    
    print(f"\n✅ Documentazione completata!")
    print(f"📁 File principale: {all_operators_file}")
    print(f"📁 File individuali: {tmp_dir}/*_help.txt")
    
    # Genera anche un indice
    index_file = tmp_dir / "operators_index.txt"
    with open(index_file, 'w', encoding='utf-8') as f_index:
        f_index.write("INDICE OPERATORI SNAP\n")
        f_index.write("=" * 30 + "\n\n")
        for i, operator in enumerate(snap_operators, 1):
            safe_name = operator.replace(".", "_").replace("-", "_")
            f_index.write(f"{i:3d}. {operator:<40} -> {safe_name}_help.txt\n")
    
    print(f"📋 Indice: {index_file}")

# Esegui la funzione se il modulo viene eseguito direttamente
if __name__ == "__main__":
    generate_operators_help()