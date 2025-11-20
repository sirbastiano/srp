from phidown.search import CopernicusDataSearcher
from phidown.s5cmd_utils import pull_down as download 
import argparse
import sys 


# Argument parser to handle command line arguments
parser = argparse.ArgumentParser(description='Download Sentinel-1 data from S3.')
parser.add_argument('--filename', type=str, required=True, help='S3 path of the product')
parser.add_argument('--output_dir', 
                    type=str, 
                    default='/Data_large/marine/PythonProjects/SAR/sarpyx/data/maya4ps', 
                    help='Output directory for the downloaded data')
args = parser.parse_args()


def main(args):
    """Main function to execute the download."""
    if not args.filename:
        print("Error: --filename argument is required.")
        sys.exit(1)
    if not args.output_dir:
        print("Error: --output_dir argument is required.")
        sys.exit(1)
    
    print(f"Downloading data from {args.filename} to {args.output_dir}...")
    searcher = CopernicusDataSearcher()
    df = searcher.query_by_name(
        args.filename
    )


    download(
        s3_path=df.iloc[0]['S3Path'],  # Get the S3 path of the product,
        output_dir=args.output_dir, 
        config_file='/workspace/.s5cfg',
    )
    sys.exit(0)
    
    
if __name__ == "__main__":
    main(args)