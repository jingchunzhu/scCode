#!/usr/bin/env python
# coding: utf-8
"""
Compute UMAP embedding from PCA-transformed data using a pretrained PyTorch UMAP model.

Example:
    python run_pumap_transform.py \
      --data_dir combined_UCE_5neuro \
      --pca_file pca.h5 \
      --model_iteration 11 \
      --output_format npy \
      --plot
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
import random
import matplotlib.pyplot as plt
from umap_pytorch import load_pumap
import pyarrow as pa
import pyarrow.ipc as ipc
import re, json

def run_pumap_transform(data_dir, embedding_file, embedding_key, pumap_model_file, plot=False, output_format =None):
    """
    Transform PCA data using a pretrained PyTorch UMAP model.
    Saves output as a .npy file in the same directory.

    Args:
        data_dir (str): Directory containing input data and model files.
        pca_file (str): PCA-transformed HDF5 file (e.g., pca.h5).
        pumap_model_file (str): model_file (e.g., pytorch_e5.pkl).
        plot (bool): If True, generates a scatter plot of 1M sampled points.
        output_format (str or None): 'npy' or 'tsv' to save; None to skip saving.
    """
    base = os.path.basename(pumap_model_file)        # e.g. 'pytorch_e5.pkl'
    stem, _ = os.path.splitext(base)                 # ('pytorch_e5', '.pkl')
    
    # extract iteration, the number after 'e'
    match = re.search(r'e(\d+)', stem)
    if match:
        iteration = int(match.group(1))
        print(iteration)  # e.g. 5
    else:
        iteration = None
        print("Error: expect model name with xxx_e{num}.pkl")
        sys.exit(1)
    
    embedding_path = os.path.join(data_dir, embedding_file)
    model_path = os.path.join(data_dir, pumap_model_file)
    
    config = {}
    config["pumap_model_file"] = model_path
    config["pca_model_file"]= os.path.join(data_dir,"ipca_model.pkl") # hard-coded for ipca transformation
    
    # --- Load model ---
    print(f"Loading PUMAP model from {model_path} ...", flush=True)
    pumap = load_pumap(model_path)
    
    # --- Load embedding data ---
    print(f"Loading embedding data from {embedding_path} ...", flush=True)
    with h5py.File(embedding_path, "r") as f:
        X_data = f[embedding_key][:]
    print(f"Embedding data shape: {X_data.shape}")

    # --- Convert to torch tensor ---
    data_tensor = torch.from_numpy(X_data)
    print(f"Tensor shape: {data_tensor.shape}, dtype: {data_tensor.dtype}", flush=True)

    n_samples = data_tensor.shape[0]
    embedding_dim = 2  # since n_components=2
    X_umap_pytorch = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    # --- Batch transform ---
    print("Starting batch transformation ...", flush=True)
    batch_size = 1_000_000
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = data_tensor[start:end]
        emb = pumap.transform(batch)
        X_umap_pytorch[start:end] = emb
        print(f"Processed {end}/{n_samples}", flush=True)

    print("✅ Transformation complete.")
    print("Final embedding shape:", X_umap_pytorch.shape)

    # --- Axis adjustment ---
    UMAP1_size = np.ptp(X_umap_pytorch[:, 0])
    UMAP2_size = np.ptp(X_umap_pytorch[:, 1])
    if UMAP1_size < UMAP2_size:
        print("Swapping axes (too tall UMAP)...", flush=True)
        X_umap_pytorch[:, [0, 1]] = X_umap_pytorch[:, [1, 0]]
        config["swap_xy_matrix"] =  [
        	[0, 1],
        	[1, 0]
        ]
    else:
        config["swap_xy_matrix"] =  [
        	[1, 0],
        	[0, 1]
        ]

    # --- Saveing transformation config ----
    config_path = os.path.join(data_dir, f"transform_e{iteration}.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    # --- Optional saving ---
    if output_format:
        umap_file = f"umap_e{iteration}.{output_format}"
        output_path = os.path.join(data_dir, umap_file)
        if output_format == "npy":
            np.save(output_path, X_umap_pytorch)
        elif output_format == "tsv":
            np.savetxt(output_path, X_umap_pytorch, delimiter="\t", fmt="%.6f", header="UMAP1\tUMAP2")
        elif output_format == "arrow":
            with pa.OSFile(umap_file, "wb") as sink:
                ipc.write_tensor(pa.Tensor.from_numpy(X_umap_pytorch), sink)
        else:
            raise ValueError("Invalid output_format. Choose 'npy', 'tsv', 'arrow', or None.")
        print(f"Saved UMAP embedding to: {output_path}")
        
    # --- Optional plotting ---
    if plot:
        print("Generating scatter plot (1M sampled points)...", flush=True)
        subset_size = min(1_000_000, X_umap_pytorch.shape[0])
        subset_indices = np.random.choice(X_umap_pytorch.shape[0], subset_size, replace=False)
        plt.figure(figsize=(8, 8))
        plt.scatter(
            X_umap_pytorch[subset_indices, 0],
            X_umap_pytorch[subset_indices, 1],
            alpha=0.5, s=1
        )
        plt.title("PUMAP Embedding (subset of 1M points)")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        #plt.show()
    
        # ✅ Save to disk instead of showing
        plot_path = os.path.join(data_dir,  f"umap_{stem}.scatter.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"✅ Plot saved to: {plot_path}")
    
    return X_umap_pytorch

def main():
    parser = argparse.ArgumentParser(description="Run UMAP transformation using pretrained PyTorch model.")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--embedding_file", default="pca.h5", help="Embedding file name (default: pca.h5)")
    parser.add_argument("--embedding_key", default="pca", help="Embedding key name (default: pca)")
    parser.add_argument("--pumap_model_file", type=str, required=True, help="Model file")
    parser.add_argument("--plot", action="store_true", help="Plot a random subset of embeddings")
    parser.add_argument("--output_format", default="npy", choices=["npy", "tsv", "arrow"], 
                        help="Optional: choose 'npy' or 'tsv' to save embedding. If not set, no file is written.")

    args = parser.parse_args()

    run_pumap_transform(
        data_dir=args.data_dir,
        embedding_file=args.embedding_file,
        embedding_key=args.embedding_key,
        pumap_model_file=args.pumap_model_file,
        plot=args.plot,
        output_format=args.output_format
    )

if __name__ == "__main__":
    main()
