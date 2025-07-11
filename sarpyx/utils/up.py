
#!/usr/bin/env python3
"""
Script to upload models/datasets to Hugging Face Hub
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, upload_file


def upload_to_hf(
    local_path: str,
    repo_id: str,
    repo_type: str = "model",
    token: str = None,
    commit_message: str = "Upload files"
):
    """
    Upload files or folder to Hugging Face Hub
    
    Args:
        local_path: Path to local file or folder
        repo_id: Repository ID (username/repo-name)
        repo_type: Type of repo ("model", "dataset", or "space")
        token: HF token (if not provided, will use cached token)
        commit_message: Commit message
    """
    api = HfApi(token=token)
    local_path = Path(local_path)
    
    try:
        if local_path.is_dir():
            print(f"Uploading folder {local_path} to {repo_id}...")
            upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=commit_message
            )
        else:
            print(f"Uploading file {local_path} to {repo_id}...")
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message=commit_message
            )
        print("Upload completed successfully!")
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face Hub")
    parser.add_argument("local_path", help="Path to local file or folder")
    parser.add_argument("repo_id", help="Repository ID (username/repo-name)")
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], 
                       default="dataset", help="Repository type")
    parser.add_argument("--token", help="HF token (optional if cached)")
    parser.add_argument("--message", default="Upload files", 
                       help="Commit message")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.local_path):
        print(f"Error: {args.local_path} does not exist")
        return
    
    upload_to_hf(
        local_path=args.local_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=args.token,
        commit_message=args.message
    )

if __name__ == "__main__":
    main()