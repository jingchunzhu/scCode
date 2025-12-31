import torch
from cellpose import models, io
import argparse
from pathlib import Path

def main():
    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Run Cellpose-SAM (cpsam) segmentation on H&E TIFF images"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing TIFF images"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save masks (tif)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

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

    # Load the SAM model
    model = models.CellposeModel(gpu=gpu, device=device, model_type="cpsam")

    # Loop over all TIFF images in the directory
    for tif_file in input_dir.glob("*.tif"):
        print(f"Processing {tif_file.name}...")
        img = io.imread(tif_file)
        masks, flows, styles = model.eval(
            img,
            channels=[0, 0],
            flow_threshold=1.0,
            cellprob_threshold=-1.0
        )

        # Save results with _cp_masks suffix
        out_filename = output_dir / f"{tif_file.stem}_cp_masks.tif"
        io.save_masks(img, masks, flows, str(out_filename), tif=True, png=False)
        print(f"Saved mask: {out_filename}")

if __name__ == "__main__":
    main()


