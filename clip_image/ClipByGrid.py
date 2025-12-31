import sys,os
from skimage.io import imread, imsave

if len(sys.argv[:]) != 4:
    print ("python ClipByGrid.py grid(# of seg per dimension)  big_img_in output_dir\n")
    sys.exit()

grid = int(sys.argv[1])
imgfile = sys.argv[2]
outputdir = sys.argv[3]

img = imread(imgfile)
print (img.shape)

os.makedirs(outputdir, exist_ok=True)

h, w = img.shape[0:2]
hseg = int(h /grid)
wseg = int(w /grid)

for row_index in range(0, grid):
    for col_index in range(0, grid):
        rowS = row_index* hseg
        if row_index == grid -1: # last seg
            rowE = h
        else:
            rowE = (row_index+1)* hseg
        colS = col_index * wseg
        if col_index == grid -1: # last seg
            colE = w
        else:
            colE = (col_index+1)* wseg 
        
        clip = img[rowS: rowE, colS: colE]
        print (clip.shape)

        output = str(row_index) + "_" + str(col_index) + ".tif"
        output = os.path.join(outputdir, output)
        imsave(output, clip)

        
