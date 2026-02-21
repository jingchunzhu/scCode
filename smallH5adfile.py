import anndata as ad
import scipy.sparse as sp
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Randomly subset cells from h5ad file")
parser.add_argument("input", help="Path to input .h5ad file")
parser.add_argument("output", help="Path to output .h5ad file")
parser.add_argument("--n", type=int, default=10, help="Number of cells to sample (default: 10)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

adata = ad.read_h5ad(args.input, backed="r")
print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

# Random sample
np.random.seed(args.seed)
idx = np.random.choice(adata.n_obs, size=args.n, replace=False)
subset_adata = adata[idx].to_memory()

print(f"Subset: {subset_adata.shape[0]} cells")
print(f"Sparse: {sp.issparse(subset_adata.X)}")

# Save idx alongside output
idx_output = args.output.replace(".h5ad", "_idx.npy")
np.save(idx_output, idx)
print(f"Saved idx to {idx_output}")

subset_adata.write_h5ad(args.output)
print(f"Saved to {args.output}")
