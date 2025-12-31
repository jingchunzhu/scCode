from skimage.io import imread
import sys
import numpy as np
from scipy.io import loadmat
import h5py

if len(sys.argv[:])!=2:
    print ("python cellNumberInImgMask.py image_mask_file \n")
    sys.exit()

imgfile = sys.argv[1]

try:
    img = imread(imgfile)
    print ("mask file:", len(np.unique(img)))
    sys.exit()
except:
    pass


try:
    img = loadmat(imgfile)
    print ("matlab older version file:", len(np.unique(img)))
    sys.exit()
except:
    pass


try:
    f = h5py.File(imgfile)
    print ("matlab -v7.3 version file")
    for key in list(f.keys()):
        print(f[key])
        variable = f[key]
        if hasattr(variable, 'shape'):
            print ("mask file:", len(np.unique(variable)))
    sys.exit()
except:
    pass

