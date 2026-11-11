"""
Dataset Downloader and Verification Utility for Parkinson's FoG Project.
Verifies the existence and SHA256 checksum of raw IMU data.
"""

import os
import hashlib
import argparse

EXPECTED_SHA256 = "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
EXPECTED_BYTES = 119629580

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest().lower()

def main():
    parser = argparse.ArgumentParser(description="Verify Parkinson's FoG dataset files.")
    parser.add_argument("--data-file", default="data/raw/sensor_data.csv", help="Path to raw CSV")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing file")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"[ERROR] Dataset file not found at: {args.data_file}")
        print("Please place 'sensor_data.csv' in 'data/raw/' as described in 'data/README.md'.")
        exit(1)
        
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

if __name__ == '__main__':
    main()
