from phidown.search import CopernicusDataSearcher


dsearcher = CopernicusDataSearcher() 

def down(filename, output_dir):
    dsearcher.download_product(filename, 
                            output_dir=output_dir,
                            config_file='/Data_large/marine/PythonProjects/SAR/sarpyx/notebooks/.s5cfg')