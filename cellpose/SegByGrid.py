import sys, os

if len(sys.argv[:]) != 4:
    print ("python SegByGrid.py.py grid(# of seg per dimension) grid_image_input_dir segmentation_output_dir\n")
    sys.exit()

grid = int(sys.argv[1])
image_dir = sys.argv[2]
seg_dir = sys.argv[3]

for row_index in range(0, grid):
    for col_index in range(0, grid):
        input = str(row_index) + "_" + str(col_index) + ".tif"
        input = os.path.join(image_dir, input)
        os.system( "python -m cellpose --image_path " + input + " --pretrained_model cyto3 --chan 1 --chan2 0 --diameter 97.2  --save_tif  --savedir " + seg_dir + " --no_npy  --verbose")
