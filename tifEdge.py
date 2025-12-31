#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os
import numpy as np
from skimage.io import imread, imsave


# In[ ]:

if len(sys.argv[:])!=3:
    print ("python tifEdge.py mask_tif_in edge_tif_out\n")
    sys.exit()

    
mask_path = sys.argv[1]
output_path = sys.argv[2]

edge_color = 200


# In[ ]:


original_image = imread(mask_path)


# In[ ]:


original_image, original_image.shape


# ## ad hoc edge detection

# In[ ]:


rows, columns = original_image.shape


# In[ ]:


def edgePixel (image, row, column, rows, columns):
    if row % 1000 ==0 and column == columns -1:
        print (row, column)
    if image[row, column] == 0:
        return 0
    if row > 0 and image[row, column]!= image[row -1, column]:
        return 1
    if row < rows - 1 and image[row, column]!= image[row +1, column]:
        return 1
    if column > 0 and image[row, column]!= image[row, column -1]:
        return 1
    if column < columns - 1 and image[row, column]!= image[row, column +1]:
        return 1
    return 0


# In[ ]:


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing

# Specify the desired number of CPUs
num_cpus =  multiprocessing.cpu_count()

# Use ThreadPoolExecutor for parallel processing with a specified number of CPUs
with ProcessPoolExecutor(max_workers=num_cpus) as executor:
    # Define a function to process a single element
    def process_row(args):
        i, row = args
        result = []
        for j in range (0, columns):
            result.append(edgePixel (original_image, i, j, rows, columns))
        return result
        
    # Use executor.map to apply the function to each element in parallel
    edge_array = np.array(list(executor.map(process_row, list(enumerate(original_image)))))


# In[ ]:


edge_array, edge_array.shape


# In[ ]:


edge_arrayCopy = edge_array.astype(np.uint8)


# In[ ]:


edge_arrayCopy, edge_arrayCopy.shape


# In[ ]:


edge_arrayCopy[edge_arrayCopy != 0] = edge_color


# In[ ]:


imsave(output_path, edge_arrayCopy)

