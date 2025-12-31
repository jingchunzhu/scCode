#!/usr/bin/env python
# coding: utf-8

# # run cellpose command line
# https://cellpose.readthedocs.io/en/latest/cli.html#cellpose-cli
# 
# make a small sample image, both DAPI single channel for command line, and DAPI + someothere channel for gui <BR>
# calibrate DAPI stain nuclei diameter using GUI<BR>
# use the cyto3 super-generalist model (not nuclei model) on DAPI stain image (command line below) <BR>
# estimate expansion parameter =  use Xenium method expenad by 15um <BR>
# expand using the routine below, then check using cellpose gui to see if the expansion parameter

# python -m cellpose --image_path 2296.small_5_DAPI.tif  --pretrained_model cyto3 --chan 0 --chan2 0  --diameter 28.5 --save_tif --verbose
# 

# # expand nuclei mask by n pixel to reach 10um cell size
# https://www.10xgenomics.com/analysis-guides/performing-3d-nucleus-segmentation-with-cellpose-and-generating-a-feature-cell-matrix
# 
# cell diameter is about 40 pixel:  10um/scalefactor <br>
# nuclei diameter is about 28 pixel by cell pose calibration, (40-28) /2 = 6 pixel <br>
# ome.tiff xml has PhysicalSizeX=0.24806940933877356 PhysicalSizeXUnit=µm 

# In[3]:


import numpy
from skimage.io import imread, imsave
import skimage
import sys
import pandas as pd

# In[1]:

if len(sys.argv[:])!=4:
    print ("python Expand_nuclei_mask.py nuclei_mask_in cell_mask_out expand_pixel \n")
    print("cells.tsv file will also be generated\n")
    sys.exit()
    
nuclei_mask_file = sys.argv[1]
cell_mask_file = sys.argv[2]
expand_pixel = int(sys.argv[3])

cells_info_file = "cells.tsv"


# In[4]:


nuclei_mask = imread(nuclei_mask_file)


# In[5]:


numpy.max(nuclei_mask)


# In[10]:


small_expanded_mask = skimage.segmentation.expand_labels(nuclei_mask, expand_pixel)


# In[ ]:


imsave(cell_mask_file, small_expanded_mask)


# get cell property including centroid coordinate, centroid-0 for x centroid-1 for y

# In[ ]:


properties = ['label', 'centroid','area', 'axis_major_length', 'axis_minor_length']
celltable = pd.DataFrame(skimage.measure.regionprops_table(small_expanded_mask, properties=properties))
celltable.set_index("label")


# In[ ]:


# save table
celltable.to_csv(cells_info_file, sep='\t', index =False, index_label="cell")

