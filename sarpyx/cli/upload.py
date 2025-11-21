#!/usr/bin/env python3
"""
Upload CLI Tool for SARPyX.

This module provides a command-line interface for uploading data
to Hugging Face Hub.
"""

import argparse
import sys
import os
from pathlib import Path
import logging

from huggingface_hub import HfApi
from .utils import validate_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for upload command.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description='Upload data to Hugging Face Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a folder to a dataset repository
  sarpyx upload --folder /path/to/folder --repo username/dataset-name
  
  # Upload with explicit repository type
  sarpyx upload --folder /path/to/folder --repo username/dataset-name --repo-type dataset
  
  # Upload to a model repository
  sarpyx upload --folder /path/to/model --repo username/model-name --repo-type model

Note:
  You need to be authenticated with Hugging Face. Run 'huggingface-cli login' first
  or set the HF_TOKEN environment variable.
"""
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to the folder to upload'
    )
    
    parser.add_argument(
        '--repo',
        type=str,
        required=True,
        help='Repository ID in format username/repo-name'
    )
    
    parser.add_argument(
        '--repo-type',
        type=str,
        default='dataset',
        choices=['dataset', 'model', 'space'],
        help='Type of repository (default: dataset)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def upload_to_huggingface(folder_path: Path, repo_id: str, repo_type: str = 'dataset') -> None:
    """
    Upload a local folder to the Hugging Face Hub.

    Args:
        folder_path: Path to the local folder.
        repo_id: Repository ID (username/repo-name).
        repo_type: Type of repository ('dataset', 'model', or 'space').

    Raises:
        Exception: If an error occurs during upload.
    """
    api = HfApi()
    
    try:
        logger.info(f'📤 Uploading folder "{folder_path}" to Hugging Face repo "{repo_id}" ({repo_type})...')
        api.upload_large_folder(
            folder_path=str(folder_path), 
            repo_id=repo_id, 
            repo_type=repo_type
        )
        logger.info('✅ Upload completed successfully.')
    except Exception as e:
        logger.error(f'❌ Upload failed: {e}')
        raise


def main() -> None:
    """
    Main entry point for upload CLI.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Validate folder path
    folder_path = validate_path(args.folder, must_exist=True)
    
    if not folder_path.is_dir():
        logger.error(f'❌ Path is not a directory: {folder_path}')
        sys.exit(1)
    
    # Validate repo ID
    repo_id = args.repo.strip()
    if not repo_id or '/' not in repo_id:
        logger.error('❌ Repository ID must be in format username/repo-name')
        sys.exit(1)
    
    # Print summary
    if args.verbose:
        print('='*60)
        print('SARPyX Upload Tool')
        print('='*60)
        print(f'Folder: {folder_path}')
        print(f'Repository: {repo_id}')
        print(f'Repository Type: {args.repo_type}')
        print('='*60)
    
    # Check for authentication
    token = os.getenv('HF_TOKEN')
    if not token:
        logger.warning('⚠️  HF_TOKEN environment variable not set.')
        logger.warning('   Make sure you are authenticated with "huggingface-cli login"')
    
    # Upload to Hugging Face
    try:
        upload_to_huggingface(folder_path, repo_id, args.repo_type)
        print('✅ Upload completed successfully!')
        sys.exit(0)
    except Exception as e:
        logger.error(f'❌ Upload failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
