import os,sys
import numpy as np
from skimage.io import imread, imsave

grid = 2
dir = "PR8_TUMOR_LATE_mask"
output_mask_file = "PR8_TUMOR_LATE_cp_masks.tif"

def matfileName(gridR, gridC):
    return os.path.join(dir, str(gridR) + "_" + str(gridC) +"_cp_masks.tif")

def mask_index(gridR, gridC, gridSize):
    return gridR * gridSize + gridC

def fix_bet_Col_mask(mask_left, mask_right):
    h,w = mask_left.shape
    fixed_mask_right = mask_right[:]
    for i in range (0, h):
        leftV = mask_left[i, -1]
        rightV = mask_right[i, 0]
        if leftV ==0 or rightV == 0 or leftV == rightV:
            pass
        else:
            fixed_mask_right[fixed_mask_right == rightV] = leftV
    return fixed_mask_right

def fix_bet_Row_mask(mask_top, mask_bottom):
    h,w = mask_top.shape
    fixed_mask_bottom = mask_bottom[:]
    for i in range (0, w):
        topV = mask_top[h-1, i]
        bottomV = mask_bottom[0, i]
        if topV ==0 or bottomV == 0 or topV == bottomV:
            pass
        else:
            fixed_mask_bottom[fixed_mask_bottom == bottomV] = topV
    return fixed_mask_bottom

# ### get cell number and mask offsets per title_grid position

# Use a larger dtype to avoid overflow
cellNumGrid = np.zeros((grid, grid), dtype=np.uint32)
cellNumOffset = np.zeros((grid, grid), dtype=np.uint32)
totalCell = 0

for gridR in range(grid):
    for gridC in range(grid):
        cellNumOffset[gridR, gridC] = totalCell
        matFile = matfileName(gridR, gridC)
        mask = imread(matFile)
        cellN = np.uint32(np.max(mask))
        cellNumGrid[gridR, gridC] = cellN
        totalCell += cellN

print("cell number")
print(cellNumGrid)
print("cell mask offset")
print(cellNumOffset)

# ### generate grid mask with offset

masks = [None] * grid * grid
# Second pass: apply offsets
for gridR in range(grid):
    for gridC in range(grid):
        offset = int(cellNumOffset[gridR, gridC])
        matFile = matfileName(gridR, gridC)
        mask = imread(matFile).astype(np.uint32)  # ensure safe type

        # Replace your current loop logic with this:
        mask = np.where(mask >0, mask + offset, mask)
        index = mask_index(gridR, gridC, grid)
        masks[index] = mask

        non_zero_vals =  masks[index][ masks[index] > 0]
        first_val = non_zero_vals[0]
        last_val = non_zero_vals[-1]

        print(gridR, gridC)
        print(offset)
        print(np.max(mask))

# ### fix the border between grids, for each row, between columns
for gridR in range (0, grid):
    for gridC in range (0, grid-1):
        index_1 = mask_index(gridR, gridC, grid)
        mask_left = masks[index_1]
        index_2 = mask_index(gridR, gridC+1, grid)
        mask_right = masks[index_2]
        
        fixed_mask_right = fix_bet_Col_mask(mask_left, mask_right)
        masks[index_2] = fixed_mask_right


# ### fix the border between grids, for each col, between rows

for gridC in range (0, grid):
    for gridR in range (0, grid-1):
        index_1 = mask_index(gridR, gridC, grid)
        mask_top = masks[index_1]
        index_2 = mask_index(gridR+1, gridC, grid)
        mask_bottom = masks[index_2]
        
        fixed_mask_bottom = fix_bet_Row_mask(mask_top, mask_bottom)
        masks[index_2] = fixed_mask_bottom

# ### merge masks

gridC = 0
total_h = 0
for gridR in range (0, grid):
    index = mask_index(gridR, gridC, grid)
    mask = masks[index]
    h, w = mask.shape
    total_h += h
    
gridR = 0
total_w = 0
for gridC in range (0, grid):
    index = mask_index(gridR, gridC, grid)
    mask = masks[index]
    h, w = mask.shape
    total_w += w
    
total_h, total_w

merged_mask = np.zeros((total_h, total_w), dtype = np.uint32)

start_h =0 
for gridR in range (0, grid):
    start_w = 0
    for gridC in range (0, grid):
        index = mask_index(gridR, gridC, grid)
        mask = masks[index]
        h,w = mask.shape
        print (gridR, gridC, h, w, start_h, start_h+h, start_w, start_w +w)
        merged_mask[start_h: start_h+h, start_w: start_w +w] = mask[:]
        start_w = start_w + w
    start_h = start_h + h

merged_mask
print("max cell number", np.max(merged_mask))
# ### export merged_mask

imsave(output_mask_file, merged_mask.astype(np.uint32))

