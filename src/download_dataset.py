"""
Dataset Downloader and Verification Utility for Parkinson's FoG Project.
Downloads raw IMU sensor data from Zenodo record 17838806 if absent,
and verifies file size and SHA256 checksum.
"""

import os
import sys
import hashlib
import argparse
import urllib.request

ZENODO_RECORD_ID = "17838806"
ZENODO_DOWNLOAD_URL = "https://zenodo.org/records/17838806/files/sensor_data.csv?download=1"
EXPECTED_SHA256 = "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
EXPECTED_BYTES = 119629580

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()

def download_dataset(target_path, url=ZENODO_DOWNLOAD_URL):
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    temp_target = target_path + ".download"
    print(f"[DOWNLOAD] Downloading raw IMU dataset from Zenodo (Record {ZENODO_RECORD_ID})...")
    print(f"  URL: {url}")
    print(f"  Target: {target_path}")

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as response, open(temp_target, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', EXPECTED_BYTES))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total_size * 100 if total_size > 0 else 0
            sys.stdout.write(f"\r  Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()
    print("\n[DOWNLOAD] Download complete. Moving to target path...")
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_target, target_path)

def main():
    parser = argparse.ArgumentParser(description="Download and verify Parkinson's FoG dataset.")
    parser.add_argument("--data-file", default="data/raw/sensor_data.csv", help="Path to raw CSV")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing file without downloading")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if file exists")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file) or args.force_download:
        if args.verify_only:
            print(f"[ERROR] Dataset file not found at: {args.data_file}")
            print(f"Run 'python src/download_dataset.py' to download from Zenodo record {ZENODO_RECORD_ID}.")
            sys.exit(1)
        else:
            download_dataset(args.data_file)
        
    actual_size = os.path.getsize(args.data_file)
    print(f"[CHECK] File size: {actual_size:,} bytes")
    
    print("[CHECK] Computing SHA256 checksum...")
    actual_sha = compute_sha256(args.data_file)
    print(f"  Expected: {EXPECTED_SHA256}")
    print(f"  Actual:   {actual_sha}")
    
    if actual_sha == EXPECTED_SHA256:
        print("[SUCCESS] Dataset checksum verified successfully! File is ready for preprocessing.")
    else:
        print("[WARNING] Checksum mismatch. The file may be modified, incomplete, or from a different source release.")
        sys.exit(1)

if __name__ == '__main__':
    main()
