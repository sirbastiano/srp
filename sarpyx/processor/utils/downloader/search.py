import pandas as pd
import asf_search as asf
import argparse 
import os, sys

class RAWHandler:
    def __init__(self, args: dict = None):
        # Initialize the object and set the values of the attributes
        self.args = args
        self.wkt = args.wkt
        self.start_idx = args.start_idx
        
        
        if self.args.start is not None:        
            self.start = self.date_check(args.start)
        else: 
            print('Start date provided not useful.')
        if self.args.end is not None:
            self.end = self.date_check(args.end)
        else:
            print('End date provided not useful.')

        print('Start date:', self.start)
        print('End date:', self.end)
        self.results = None
    
    @staticmethod
    def date_check(date):
        if not date.endswith('Z'):
            if date.endswith('T00:00:00.000Z'):
                return date
            else:
                return date + 'T00:00:00.000Z'
        else:
            return date

    def search(self, max_res=10):
        # Set search parameters
        search_parameters = {
            "platform": "Sentinel-1",
            "processingLevel": "RAW",
            "instrument": "C-SAR",
            "intersectsWith": self.wkt,
            "maxResults": max_res,
            "beamSwath": ["S1", "S2", "S3", "S4", "S5", "S6"], # Stripmap mode | "beamSwath": ["IW"],
        }    
        search_parameters['start'] = self.start
        search_parameters['end'] = self.end

        try:
            # results = asf.geo_search(**search_parameters)
            results = asf.search(**search_parameters)
            self.results = results[int(self.start_idx):]
            print('Len results:', len(self.results))
            print('Search results:', results)
        except Exception:
            print("No results found.")
            sys.exit(1)

        with open("search_results.csv", "w") as f:
            f.writelines(results.csv())
        
        df = pd.read_csv("search_results.csv")
        os.remove("search_results.csv")
        return df
    
    def download(self, username, psw, output_dir):
        session = asf.ASFSession().auth_with_creds(username, psw)
        try:
            self.results.download(path=output_dir, session=session)
        except Exception as e:
            print('Error downloading results:', e)
            sys.exit(1)
    
def download_single(wkt, start, end, username, psw, output_dir, download=False, max_res=2):
        D = RAWHandler(wkt, start, end)
        df = D.search(max_res=max_res)
        print('='*50)
        print('Search results:', df)
        print('='*50)
        if download and len(df) > 0:
            print('Downloading...')
            os.makedirs(output_dir, exist_ok=True)
            D.download(username, psw, output_dir=output_dir)
            print('='*50)
            print('Download complete!')
        return df
    
if __name__ == "__main__":
    """
    Example usage:
        python -m utils.downloader.search --start "2020-01-01" --end "2022-12-31" --out "./DATA/" --download --max 1
    """    
    # Set search parameters with argparse
    parser = argparse.ArgumentParser()
    # create a wkt with: https://wktmap.com/
    parser.add_argument("--wkt", help="WKT polygon to search", default='POLYGON ((-103.302551 -57.705222, -103.302551 16.297126, -23.479304 16.297126, -23.479304 -57.705222, -103.302551 -57.705222))')
    parser.add_argument("--start", help="Start date", default=None)
    parser.add_argument("--end", help="End date", default=None)
    parser.add_argument("--out", help="Output folder of products", default='./DATA/')
    parser.add_argument("--max_res", help="Maximum number of results", default=1000)
    parser.add_argument("--start_idx", help="Maximum number of results", default=0)
    parser.add_argument('--download', dest='download', action='store_true', help='Set to all products', default=False)

    parser.add_argument("--username", help="ASF username", type=str, default=None)
    parser.add_argument("--psw", help="ASF password", type=str, default=None)
    
    args = parser.parse_args()
    # savedir
    os.makedirs(args.out, exist_ok=True)

    if args.username is not None:
        username = args.username
        assert args.psw is not None, "Password not provided."
        psw = args.psw
        
    else:
        username, psw = os.environ["USERNAME"], os.environ["PASSWORD"]
        print('Username:', username)
        print('Password:', psw)

    caller = RAWHandler(args=args)
    df = caller.search(max_res=args.max_res)
    df.to_excel(f'{args.out}/Searched_products.xlsx')
    
    if args.download:        
        caller.download(username, psw, args.out)
    else:
        sys.exit(0)