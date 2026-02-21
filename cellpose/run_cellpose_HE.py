import torch
from cellpose import models, io
import argparse
from pathlib import Path
import os

"""
https://cellpose.readthedocs.io/en/latest/settings.html

Flow threshold
Note there is nothing keeping the neural network from predicting horizontal and vertical flows that do not correspond to any real shapes at all. In practice, most predicted flows are consistent with real shapes, because the network was only trained on image flows that are consistent with real shapes, but sometimes when the network is uncertain it may output inconsistent flows. To check that the recovered shapes after the flow dynamics step are consistent with real ROIs, we recompute the flow gradients for these putative predicted ROIs, and compute the mean squared error between them and the flows predicted by the network.

The flow_threshold parameter is the maximum allowed error of the flows for each mask. The default is flow_threshold=0.4. Increase this threshold if cellpose is not returning as many ROIs as you’d expect. Similarly, decrease this threshold if cellpose is returning too many ill-shaped ROIs.

Cellprob threshold
The network predicts 3 outputs: flows in X, flows in Y, and cell “probability”. The predictions the network makes of the probability are the inputs to a sigmoid centered at zero (1 / (1 + e^-x)), so they vary from around -6 to +6. The pixels greater than the cellprob_threshold are used to run dynamics and determine ROIs. The default is cellprob_threshold=0.0. Decrease this threshold if cellpose is not returning as many ROIs as you’d expect. Similarly, increase this threshold if cellpose is returning too ROIs particularly from dim areas.

"""

def main():
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Run Cellpose-SAM segmentation on H&E TIFF images from a directory, including grid-structured image collections, and generate cell segmentation outputs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing TIFF images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save masks (tif)"
    )
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.4,
        help="flow_threshold value (default 0.4), Increase this threshold if cellpose is not returning as many ROIs as you’d expect"
    )
    parser.add_argument(
        "--cellprob_threshold",
        type=float,
        default=0.0,
        help="cellprob_threshold value (default 0), Decrease this threshold if cellpose is not returning as many ROIs as you’d expect"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    flow_threshold = args.flow_threshold
    cellprob_threshold = args.cellprob_threshold
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Enable verbose output
    io.logger_setup()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # macOS GPU
        gpu = True
    else:
        device = torch.device("cpu")
        gpu = False
    print(f"Using device: {device}")

    # Load model
    model = models.CellposeModel(gpu=gpu, device=device) # general model SAM
    
    # Loop over all TIFF images in the directory
    for tif_file in input_dir.glob("*.tif"):
        print(f"Processing {tif_file.name}...")
        img = io.imread(tif_file)
        masks, flows, styles = model.eval(
            img,
            channels=[0,0],
            flow_threshold= flow_threshold,
            cellprob_threshold=cellprob_threshold
        )

        # Save results with _cp_masks suffix
        out_filename = os.path.join( output_dir, f"{tif_file.stem}")
        io.save_masks(img, masks, flows, str(out_filename), tif=True, png=False)
        print(f"Saved mask: {out_filename}_cp_masks.tif")

if __name__ == "__main__":
    main()


