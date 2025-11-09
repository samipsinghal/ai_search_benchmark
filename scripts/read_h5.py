#!/usr/bin/env python3
"""
read_h5.py — utility to inspect HDF5 (h5) files.
Prints dataset keys and shapes.
"""

import sys
import h5py

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/read_h5.py <path_to_h5>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"Inspecting: {file_path}\n")

    with h5py.File(file_path, "r") as f:
        print("Datasets found:")
        for k, v in f.items():
            try:
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            except Exception:
                print(f"  {k}: (unprintable shape)")

if __name__ == "__main__":
    main()
