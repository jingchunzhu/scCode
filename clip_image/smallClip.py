import sys
from skimage.io import imread, imsave

tilesize = 2048

if len(sys.argv[:]) != 4:
    print ("python smallClip.py tilesize(e.g.2048) big_img_in small_tiff_out\n")
    sys.exit()

tilesize = int(sys.argv[1])
imgfile = sys.argv[2]
output = sys.argv[3]

print(imgfile)
img = imread(imgfile)
print (img.shape)

h, w = img.shape[0:2]

if (h <= tilesize or w <= tilesize):
    print ("image is smaller than ", tilesize , "x", tilesize, "\n")
    sys.exit() 

clip = img[ int((h - tilesize)/2): int((h + tilesize)/2), int((w - tilesize)/2) : int((w + tilesize)/2) ,]
print (clip.shape)

imsave(output, clip)
