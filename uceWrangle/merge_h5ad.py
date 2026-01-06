import scanpy as sc
import anndata as ad
import glob, re, sys

pattern= "tmp/e4ddac12-f48f-4455-8e8d-c2a48a683437-logNorm1M_cell_*_uce_adata.h5ad"
final_file = "tmpdata/e4ddac12-f48f-4455-8e8d-c2a48a683437-logNorm1M_uce_adata.h5ad"

# Function to extract the starting index (e.g., 0, 10000, 20000...)
def extract_start_index(filename):
    # regex extracts the first number after "allones_cell_"
    match = re.search(r"allones_cell_(\d+)_\d+", filename)
    return int(match.group(1)) if match else float('inf')

files = sorted(glob.glob(pattern))
print("Found", len(files), "files")

# Sort numerically by extracted index
files_sorted = sorted(files, key=extract_start_index)

for file in files_sorted:
    print (file)

# Start with the first file
print("Loading initial file:", files_sorted[0])
adata_merged = sc.read_h5ad(files_sorted[0])

# Incrementally append each subsequent file
for f in files_sorted[1:]:
    print("Appending:", f)
    adata_next = sc.read_h5ad(f)

    # Append one dataset at a time (efficient, keeps sparse matrices)
    adata_merged = ad.concat([adata_merged, adata_next], axis=0)

# Final write

adata_merged.write(final_file)
print("Done. Final shape:", adata_merged.shape)




