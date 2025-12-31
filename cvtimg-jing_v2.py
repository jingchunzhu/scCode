import zarr
import json
import os, sys
import math
import numpy
import xml.etree.ElementTree as ET
from functools import partial
import concurrent.futures
import multiprocessing
import tempfile
import argparse

# This isn't worth generalizing until we have more image files, e.g. it's
# unclear how we should process the file if there are multiple images, or other
# dimensions (T, Z). It's also unclear if other files will represent
# downsamples in the same way.

# XYZCT

tmp = 'build'
tile_size = 1024

def images(root):
    return len(root)

def downsamples(root):
    return list(map(lambda i: len(root[i]), range(images(root))))

# downsamples in first image
def levels(root):
    return list(reversed(range(downsamples(root)[0])))

def levels_jing(root):
    return list(reversed(range(len(root[0]))))

def split(shape, base, alpha_off, format):
    height = shape[0]
    width = shape[1]
    xtiles = math.ceil(width / tile_size)
    ytiles = math.ceil(height / tile_size)
    alpha = '-alpha off' if alpha_off else ''

    dim = f'{str(tile_size)}x{str(tile_size)}'
    temp_dir = tempfile.mkdtemp()
    os.system(f'convert "{tmp}/{base}.{format}" -crop {dim} -background none -extent {dim} {alpha} {temp_dir}/tiles_%d.{format}')
    i = 0
    for y in range(ytiles):
        for x in range(xtiles):
            os.system(f'mv {temp_dir}/tiles_{str(i)}.{format} {dir}/{base}-{str(y)}-{str(x)}.{format}')
            i = i + 1
            
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

def split_to_jpeg(shape, base, alpha_off, formatin):
    height = shape[0]
    width = shape[1]
    xtiles = math.ceil(width / tile_size)
    ytiles = math.ceil(height / tile_size)
    alpha = '-alpha off' if alpha_off else ''

    dim = f'{str(tile_size)}x{str(tile_size)}'
    temp_dir = tempfile.mkdtemp()
    os.system(f'convert "{tmp}/{base}.{formatin}" -crop {dim} -background none -extent {dim} -quality 100 {alpha} {temp_dir}/tiles_%d.jpeg')
    i = 0
    for y in range(ytiles):
        for x in range(xtiles):
            os.system(f'mv {temp_dir}/tiles_{str(i)}.jpeg {dir}/{base}-{str(y)}-{str(x)}.jpeg')
            i = i + 1
            
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)
            
def to_jpeg_jing(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -alpha off "{tmp}/{base}.png"')
    split_to_jpeg(shape, base, True, 'png')

def to_jpeg(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -quality 100 "{tmp}/{base}.jpeg"')
    split(shape, base, True, 'jpeg')

def to_png(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -alpha off {tmp}/result.png')
    os.system(f'convert {tmp}/result.png -alpha copy -fx "#fff" "{tmp}/{base}.png"')
    split(shape, base, False, 'png')

def to_png_gray(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -alpha off "{tmp}/{base}.png"')
    split(shape, base, True, 'png')

def to_png_gray8(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -alpha off -depth 8 "{tmp}/{base}.png"')
    split(shape, base, True, 'png')

def to_jpeg_gray8_jing(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -alpha off  "{tmp}/{base}.png"')
    split_to_jpeg(shape, base, True, 'png')

def to_jpeg_gray8(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 16 gray:"{tmp}/{base}.raw" -quality 100 "{tmp}/{base}.jpeg"')
    split(shape, base, True, 'jpeg')

def to_png_gray88(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 gray:"{tmp}/{base}.raw" -alpha off -depth 8 "{tmp}/{base}.png"')
    split(shape, base, True, 'png')

def to_png_binary(shape, base):
    print (base)
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 -threshold 50% gray:"{tmp}/{base}.raw" -alpha off  -depth 1 "{tmp}/{base}.png"')
    split(shape, base, True, 'png')

def to_png_3bit(shape, base):
    print (base)
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 gray:"{tmp}/{base}.raw" -alpha off  -colors 8 "{tmp}/{base}.png"')
    split(shape, base, True, 'png')

def to_jpeg_gray88_jing(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 gray:"{tmp}/{base}.raw" -alpha off  "{tmp}/{base}.png"')
    split_to_jpeg(shape, base, True, 'png')

def to_jpeg_gray88(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 gray:"{tmp}/{base}.raw" -quality 100 "{tmp}/{base}.jpeg"')
    split(shape, base, True, 'jpeg')

def to_png_rgb88(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 rgb:"{tmp}/{base}.raw" -alpha off -depth 8 "{tmp}/{base}.png"')
    split(shape, base, True, 'png')

def to_jpeg_rgb(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 rgb:"{tmp}/{base}.raw" -quality 100 "{tmp}/{base}.jpeg"')
    split(shape, base, True, 'jpeg')

def to_jpeg_rgb_jing(shape, base):
    os.system(f'convert -size {str(shape[1])}x{str(shape[0])} -depth 8 rgb:"{tmp}/{base}.raw" -alpha off -depth 8 "{tmp}/{base}.png"')
    split_to_jpeg(shape, base, True, 'png')
    
# get data from level & channel
# dump to raw file
# call fn to convert to png & split
# root: [chn][level]

def build_pyramid(channels):
    for chn in range(len(channels)):
        levels = len(channels[chn]) - 1
        for level in range(len(channels[chn])):
            print(level, chn)
            data = channels[chn][level]
            base = f'c{str(chn)}-{levels - level}'
            data.astype('<u2').tofile(f'{tmp}/{base}.raw')
            to_png_gray(data.shape, base)

def split_into_batches(iterable, batch_size):
    """Generator function to split an iterable into batches of a specified size."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def batchChn_oneLvl_pyramid (batch_chn, root, lvls, level, shape,  chn_max, start_chn=0, format="jpeg"):
    print(batch_chn, format)
    height = shape[0]
    width = shape[1]
    for chn in batch_chn:
        print(level, chn)
        base = f'c{str(start_chn+chn)}-{len(lvls) -1 - level}'
        print (base)

        if len(lvls) >=10 and level == 0:
            arr = (numpy.clip(root[0][level][0][chn][0] / chn_max[chn], 0, 1))
            ( arr * (256 * 256-1)).astype('<u2').tofile(f'{tmp}/{base}.raw')
        else:
            (root[0][level][0][chn][0]/chn_max[chn] * (256 * 256-1)).astype('<u2').tofile(f'{tmp}/{base}.raw')
            
        if format == "png":
            to_png_gray(shape, base)
        if format == "jpeg":
            if height < 65 * 1000 and width < 65 * 1000:
                to_jpeg(shape, base)
            else:
                to_jpeg_jing(shape, base)

def getChannelMax(chn, root):
    return numpy.max(root[0][0][0][chn][0])
    
def getChannelMaxBatch(batch, root):
    run = partial(getChannelMax, root= root)
    return list(map(run, batch))
    
def build_pyramid_jing (root, channelN, start_chn = 0, format = "jpeg"):
    lvls = levels_jing(root) 
    print (lvls, format)

    chns = range (channelN)

    # get channel max
    if len(lvls) >= 10:
        chn_max = [numpy.max(root[0][1][0][chn][0]) for chn in chns] # use one level up to get the max, less data, more efficient, still capture likely good estimate of max
    else:
        batch_size = math.ceil(channelN / 10)
        batches = list(split_into_batches(list(range(channelN)), batch_size))

        run = partial(getChannelMaxBatch, root=root)
        with multiprocessing.Pool() as pool:
            chn_max_batch = list(pool.map(run, batches))    
            #chn_max = numpy.array(chn_max_batch, dtype=object).flatten().tolist()
            chn_max = numpy.concatenate([numpy.ravel(arr) for arr in chn_max_batch]).tolist()
    print("chn_max:", chn_max)

    # pyramid
    for level in lvls:
        shape = root[0][level].shape[-2:]

        if level == 0 and len(lvls) >= 10:
            runN = 1
        else:
            runN = 5
        print ("runN:", runN)
        batch_size = math.ceil(channelN / runN)
        run = partial(batchChn_oneLvl_pyramid, root = root, lvls = lvls, level= level, shape=shape, \
                      chn_max = chn_max, start_chn = start_chn, format = format)

        batches = list(split_into_batches(list(range(channelN)), batch_size))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(run, batches)


def build_background_pyramid_binary(root, prefix, start_chn = 0, zoomout = 0):
    lvls = levels_jing(root)
    print ("background_pyramid_binary", lvls)
    
    for level in lvls:
        #if level > zoomout: # most zoomed in level =0 , most zoomed out level = len(lvls) -1
        #    continue

        print(level)
        print (f'{prefix}{start_chn}-{len(lvls) -1 - level}')
        
        shape = root[0][level][0][0][0].shape
        base = f's{start_chn}-{len(lvls) -1 - level}'
        base = f'{prefix}{start_chn}-{len(lvls) -1 - level}'
        data = root[0][level][0][0][0].astype('<u1')
        data.tofile(f'{tmp}/{base}.raw')

        print(shape)
        to_png_binary(shape, base)

def build_background_pyramid_3bitpng(root, start_chn = 0):
    lvls = levels(root)
    print ("background_pyramid_3bitpn", lvls)
    
    for level in lvls:
        print(level)
        print (f'm{start_chn}-{len(lvls) -1 - level}')
        
        shape = root[0][level][0][0][0].shape
        base = f'm{start_chn}-{len(lvls) -1 - level}'
        data = root[0][level][0][0][0].astype('<u1')
        data.tofile(f'{tmp}/{base}.raw')

        print(shape)
        to_png_3bit(shape, base)

            
def build_background_pyramid_rgb(root):
    lvls = levels_jing(root)

    for level in lvls:
        print(level)
        shape = root[0][level][0][0][0].shape
        base = f'i-{lvls[0] - level}'
        data = numpy.stack([root[0][level][0][chn][0] for chn in range(3)], axis = -1).astype('<u1')
        data.tofile(f'{tmp}/{base}.raw')
        #to_png_rgb88(shape, base)

        print(shape)
        height = shape[0]
        width = shape[1]
        if height < 65 * 1000 and width < 65 * 1000:
            to_jpeg_rgb(shape, base)
        else:
            to_jpeg_rgb_jing(shape, base)

            
percentile = 0.0005 # cribbed from viv
def channel_stats(ch_data, name): 
    s = sorted(numpy.asarray(ch_data[-1].flatten()))
    # have to cast to 8 bit for 16 bit image
    min = round(s[0] / 256)
    max = round(s[len(s) - 1] / 256)
    f = list(filter(lambda x: x > 0, s))
    l = len(f)
    lower = round(f[int(l * percentile)] / 256)
    upper = round(f[int(l * (1 - percentile))] / 256)

    return {'name': name, 'min': min, 'max': max, 'lower': lower, 'upper': upper}

def channel_stats_jing(smallest_ch_data, name):
    s = sorted(numpy.asarray(smallest_ch_data.flatten()))
    maxV = s[len(s) - 1] 
    # have to cast to 8 bit for 16 bit image because the UI slider requires 0-255 range
    # min = round(s[0] / 256)
    min = round(s[0] / maxV * 255)
    # max = round(s[len(s) - 1] / 256)
    max = 255
    f = list(filter(lambda x: x > 0, s))
    l = len(f)
    print (s[-1],f[0])
    # lower = round(f[int(l * percentile)] / 256)
    # upper = round(f[int(l * (1 - percentile))] / 256)
    lower = round(f[int(l * percentile)] / maxV * 255)
    upper = round(f[int(l * 0.75)] /maxV * 255)

    return {'name': name, 'min': min, 'max': max, 'lower': lower, 'upper': upper}

def build_stats(data, channels):
    return list(map(channel_stats, data, channels))

def build_stats_jing(smallest_data, channels):
    return list(map(channel_stats_jing, smallest_data, channels))

def channel_names():
    root = ET.parse(os.path.join(raw, 'OME', 'METADATA.ome.xml')).getroot()
    channels = root.findall('.//{*}Channel')
    return [channel.attrib['Name'] for channel in channels]

def channel_count(raw):
    root = ET.parse(os.path.join(raw, 'OME', 'METADATA.ome.xml')).getroot()
    channels = root.findall('.//{*}Channel')
    return len(channels)

def write_stats_jing(root, channels, defaults=[], background=False, fileformat ="png", metafilename="metadata.json"):
    lvls = len(levels_jing(root))
    chls = len(channels)

    print("getting smallest data vector")
    smallest_data = map(lambda chn: root[0][lvls-1][0][chn][0],  range(chls))
    print("build stats using smallest data vector")
    stats = build_stats_jing(smallest_data, channels)
    print (stats)
    smallest = root[0][lvls-1][0][0][0].shape
    print(smallest)

    if len(defaults) ==0: # user did not provide defaults, just set it to be the first channel
        defaults= [channels[0]]
        
    with open(metafilename, 'w') as f:
        meta = {'defaults': defaults, 'channels': stats,
                'size': [smallest[1], smallest[0]], 'tileSize': tile_size, 
                'background': background, 'levels': lvls, 'fileformat': fileformat}
        json.dump(meta, f, indent="\t")
        
def write_stats_seg(root, label, metafilename="metadata.json"):
    if os.path.exists(metafilename):
        # Open and read the JSON file
        with open(metafilename, 'r') as f:
            meta = json.load(f)
            if  "segmentation" not in meta:
                meta["segmentation"] =[]
            meta["segmentation"].append(
                {
                    "name": label,
                    "fileformat": "png",
                })

        with open(metafilename, 'w') as f:
            print (meta)
            json.dump(meta, f, indent="\t")
    else:
        print("There is no exisiting metadata.json for the image pyramid to add segmentation settings.")
        sys.exit()
        
def write_stats_background_rgb(root, fileformat ="jpeg", metafilename="metadata.json"):
    if os.path.exists(metafilename):
        with open(metafilename, 'r') as f:
            meta = json.load(f)
            meta["background"] = True
        with open(metafilename, 'w') as f:
            print (meta)
            json.dump(meta, f, indent="\t")
    else:
        lvls = len(levels_jing(root))
        print("getting smallest data vector")
        smallest = root[0][lvls-1][0][0][0].shape
    
        with open(metafilename, 'w') as f:
            meta = {'defaults': [], 'channels': [],
                    'size': [smallest[1], smallest[0]], 'tileSize': tile_size, 
                    'background': True, 'levels': lvls, 'fileformat': fileformat}
            print (meta)
            json.dump(meta, f, indent="\t")

def dump_attrs():
    # Assumes there's only one Pixels element.
    # Assumes dimension order
    root = ET.parse('tiles/OME/METADATA.ome.xml').getroot()
    pixels = root.findall('.//{*}Pixels')[0]
    dimensions = pixels.attrib['DimensionOrder']
    print(dimensions)

def mkdirs(dir):
    os.system(f'mkdir -p {tmp}')
    os.system(f'mkdir -p {dir}')


# Create the parser
parser = argparse.ArgumentParser(description="Process input arguments.")

# Define arguments
parser.add_argument("-m", "--mode", type=str, required=True, choices=["IF","rgb","seg"], help="mode (IF,rgb,seg)")
parser.add_argument("-i", "--inputRawDir", type=str, required=True, help="input rawZarrDir")
parser.add_argument("-o", "--outputDir", type=str, required=True, help="output imageDir")

# Optional --format argument (initially not required)
parser.add_argument("-s", "--start_chn", type=int, help="required for 'IF/seg':start_channel_number(index start with 0)")
parser.add_argument("-f", "--file_channels_name", type=str, help="required for 'IF':a file of channels_names, one name per line")
parser.add_argument("--seg_label", type=str, help="required for 'seg':segmentation label")
parser.add_argument("--fileformat", type=str, choices=["jpeg","png"], default="jpeg", help="optional for 'IF':image tile format (default: jpeg)")

# Parse the arguments
args = parser.parse_args()

# Conditional check for format requirement
if args.mode in ["IF", "seg"] and args.start_chn is None:
    parser.error("-s is required when mode is 'IF' or 'seg'")
if args.mode in ["IF"] and args.file_channels_name is None:
    parser.error("-f is required when mode is 'IF'")
if args.mode in ["seg"] and args.seg_label is None:
    parser.error("--seg_label is required when mode is 'seg'")

    
dir = args.outputDir
mkdirs(dir)

raw = args.inputRawDir
mode = args.mode
start_chn = args.start_chn
markerFile = args.file_channels_name
fileformat = args.fileformat

if mode =="IF":
    # always jpeg
    if args.start_chn is None:
        parser.error("-s --start_chn is required for IF mode")
        sys.exit()
    if not args.file_channels_name:
        print ("-f --file_channels_name is required for IF mode")
        sys.exit()

    channels_name = []
    fin = open(markerFile, 'r')
    for marker in fin.readlines():
        marker = marker.strip()
        channels_name.append(marker)
    print(channels_name)
    fin.close()
    
    root = zarr.open(raw, mode='r')
    build_pyramid_jing(root, len(channels_name), start_chn, format = fileformat)
    print('write_stats')
    write_stats_jing(root, channels_name, fileformat = fileformat, metafilename = os.path.join(dir, "metadata.json"))

elif mode == "rgb":
    root = zarr.open(raw, mode='r')
    # always jpeg
    build_background_pyramid_rgb(root)
    write_stats_background_rgb(root, fileformat ="jpeg", metafilename = os.path.join(dir, "metadata.json"))

elif mode == "seg":
    if args.start_chn < 0:
        parser.error("--start_channel_number is required for seg mode")
        sys.exit()

    label = args.seg_label
        
    root = zarr.open(raw, mode='r')
    prefix = "s"
    # always png
    build_background_pyramid_binary(root, prefix, start_chn = start_chn)
    label = args.seg_label
    write_stats_seg(root, label, metafilename = os.path.join(dir, "metadata.json"))
    
"""
raw = 'rawMaskBinary' 
root = zarr.open(raw, mode='r')
channels_name = ['cell mask']
prefix = "m"
nChn = 1
start_chn = 0
    
build_background_pyramid_binary(root, prefix, start_chn = start_chn)
"""




