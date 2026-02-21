import pandas as pd
import numpy as np

# === SETTINGS ===
tsv_file = "tissue_positions.tsv"       # replace with your TSV file path
physical_spacing_um = 8.0    # Visium spot spacing in microns

# === LOAD DATA ===
df = pd.read_csv(tsv_file, sep='\t')

# Sort by array_row and array_col for easy indexing
df.sort_values(['array_row', 'array_col'], inplace=True)
df.reset_index(drop=True, inplace=True)

# === VECTORIZE NEIGHBOR DISTANCES ===

# Convert pixel coordinates to numpy arrays
pxl_row = df['pxl_row_in_fullres'].values
pxl_col = df['pxl_col_in_fullres'].values
row = df['array_row'].values
col = df['array_col'].values

distances = []

# Neighbor in next column (same row)
mask_col_neighbor = df.merge(df, left_on=['array_row','array_col'], right_on=['array_row','array_col'], how='inner')
# Faster: iterate only possible neighbors
for r in np.unique(row):
    row_idx = np.where(row == r)[0]
    # sort by column
    row_cols = col[row_idx]
    row_rows = pxl_row[row_idx]
    row_cols_px = pxl_col[row_idx]
    for i in range(len(row_idx)-1):
        dx = row_rows[i+1] - row_rows[i]
        dy = row_cols_px[i+1] - row_cols_px[i]
        distances.append(np.sqrt(dx**2 + dy**2))

# Neighbor in next row (same column)
for c in np.unique(col):
    col_idx = np.where(col == c)[0]
    # sort by row
    col_rows = row[col_idx]
    col_rows_px = pxl_row[col_idx]
    col_cols_px = pxl_col[col_idx]
    for i in range(len(col_idx)-1):
        dx = col_rows_px[i+1] - col_rows_px[i]
        dy = col_cols_px[i+1] - col_cols_px[i]
        distances.append(np.sqrt(dx**2 + dy**2))

# Convert to array
distances = np.array(distances)

# === MICRONS PER PIXEL ===
um_per_pixel = physical_spacing_um / distances
mean_um_per_pixel = um_per_pixel.mean()

print(f"Estimated microns per pixel: {mean_um_per_pixel:.4f} µm/px")
print(f"Number of distances used: {len(distances)}")
