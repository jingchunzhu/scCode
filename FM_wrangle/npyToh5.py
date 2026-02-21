import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser(description="Convert a 2D numpy .npy file to .h5 format")
parser.add_argument("input", help="Path to input .npy file")
parser.add_argument("output", help="Path to output .h5 file")
parser.add_argument("--key", default="data", help="Dataset key name inside h5 file (default: data)")
parser.add_argument("--compression", default="gzip", help="Compression type (default: gzip)")
args = parser.parse_args()

arr = np.load(args.input)
print(f"Loaded {args.input}: shape={arr.shape}, dtype={arr.dtype}")

with h5py.File(args.output, "w") as f:
    f.create_dataset(args.key, data=arr, compression=args.compression)

print(f"Saved to {args.output}")
