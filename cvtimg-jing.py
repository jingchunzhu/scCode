import zarr
import json
import os
import math
import numpy
import xml.etree.ElementTree as ET
from functools import partial
import concurrent.futures
import tempfile

# This isn't worth generalizing until we have more image files, e.g. it's
# unclear how we should process the file if there are multiple images, or other
# dimensions (T, Z). It's also unclear if other files will represent
# downsamples in the same way.

# XYZCT

tmp = 'CID44971_spatial/build'
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

def batchChn_oneLvl_pyramid (batch_chn, root, lvls, level, shape,  chn_max, start_chn=0):
    print(batch_chn)
    height = shape[0]
    width = shape[1]
    for chn in batch_chn:
        print(level, chn)
        base = f'c{str(start_chn+chn)}-{len(lvls) -1 - level}'
        print (base)
        (root[0][level][0][chn][0]/chn_max[chn] * (256 * 256-1)).astype('<u2').tofile(f'{tmp}/{base}.raw')
        # to_png_gray(shape, base)

        if height < 65 * 1000 and width < 65 * 1000:
            to_jpeg(shape, base)
        else:
            to_jpeg_jing(shape, base)

def build_pyramid_jing (root, channelN, start_chn = 0):    
    lvls = levels_jing(root) #list(reversed(range(downsamples(root)[0])))
    print (lvls)

    chns = range (channelN)
    chn_max = list(map(lambda chn: numpy.max(root[0][0][0][chn][0]), chns))
    print (chn_max)
    
    for level in lvls:
        shape = root[0][level].shape[-2:]
        print(shape)

        if level == 0 and len(lvls) >= 10:
            runN = 1
        else:
            runN = 5

        batch_size = math.ceil(channelN / runN)
        run = partial(batchChn_oneLvl_pyramid, root = root, lvls = lvls, level= level, shape=shape, chn_max = chn_max, start_chn = start_chn)

        batches = list(split_into_batches(list(range(channelN)), batch_size))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(run, batches)


def build_background_pyramid_binary(root, prefix, start_chn = 0, zoomout = 0):
    lvls = levels(root)
    print ("background_pyramid_binary", lvls)
    
    for level in lvls:
        if level > zoomout: # most zoomed in level =0 , most zoomed out level = len(lvls) -1
            continue

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
    lvls = levels(root)

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
    # have to cast to 8 bit for 16 bit image because the UI slider requires 0-256 range
    # min = round(s[0] / 256)
    min = round(s[0] / maxV * 256)
    # max = round(s[len(s) - 1] / 256)
    max = 255
    f = list(filter(lambda x: x > 0, s))
    l = len(f)
    # lower = round(f[int(l * percentile)] / 256)
    # upper = round(f[int(l * (1 - percentile))] / 256)
    lower = round(f[int(l * percentile)] / maxV * 256)
    upper = round(f[int(l * (1 - percentile))] /maxV * 256)

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

    
    with open(f'{dir}/' + metafilename, 'w') as f:
        meta = {'defaults': defaults, 'channels': stats,
                'size': [smallest[1], smallest[0]], 'tileSize': tile_size, 
                'background': background, 'levels': lvls, 'fileformat': fileformat}
        print (meta)
        json.dump(meta, f)

def write_stats_background_rgb(root, fileformat ="png", metafilename="metadata.json"):
    lvls = len(levels_jing(root))

    print("getting smallest data vector")
    smallest_data = map(lambda chn: root[0][lvls-1][0][chn][0],  range(chls))
    
    with open(f'{dir}/' + metafilename, 'w') as f:
        meta = {'defaults': [], 'channels': [],
                'size': [smallest[1], smallest[0]], 'tileSize': tile_size, 
                'background': True, 'levels': lvls, 'fileformat': fileformat}
        print (meta)
        json.dump(meta, f)

def write_stats(data, channels, defaults=[], background=False):
    stats = build_stats(data, channels)
    smallest = data[0][-1].shape
    lvls = len(data[0])
    with open(f'{dir}/metadata.json', 'w') as f:
        meta = {'defaults': defaults, 'channels': stats,
                'size': [smallest[1], smallest[0]], 'tileSize': tile_size, 
                'background': background, 'levels': lvls}
        json.dump(meta, f)

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


if True:
    dir = '../CID44971/image'
    mkdirs(dir)

    """
    raw = 'rawMaskBinary' 
    root = zarr.open(raw, mode='r')
    channels_name = ['cell mask']
    prefix = "m"
    nChn = 1
    start_chn = 0
    
    build_background_pyramid_binary(root, prefix, start_chn = start_chn)
    """
    """
    raw = 'raw2296CellEdge' 
    root = zarr.open(raw, mode='r')
    channels_name = ['cell segmentation (Davis)']
    prefix = "s"
    nChn = 1
    start_chn = 2
    build_background_pyramid_binary(root, prefix, start_chn = start_chn)


    raw = 'raw2296NucEdge' 
    root = zarr.open(raw, mode='r')
    channels_name = ['nucleus segmentation (Davis)']
    prefix = "s"
    nChn = 1
    start_chn = 3
    build_background_pyramid_binary(root, prefix, start_chn = start_chn)
    """
    
    """
    raw = 'raw2296' 
    root = zarr.open(raw, mode='r')
    channels_name = ["CK", "CD11c", "F480", "CD163", "DNA (DAPI)"]
    # channels_name = ["DNA_1", "bg2a", "bg3a", "bg4a", "DNA_2", "pERK", "CD207", "SOX10", "DNA_3", "CD45RO", "SOX2", "CD25", "DNA_4", "CD4", "pan-CK", "CD8a", "DNA_5", "CD163", "FOXP3", "CD3d", "DNA_6", "pS6", "CD11c", "PDL1", "DNA_7", "MCAM", "CD68", "PD1", "DNA_8", "S100a", "ICOS", "OX40", "DNA_9", "CD40L", "HLADR", "HLAA", "DNA_10", "NCAD", "LAG3", "CD31", "DNA_11", "CD73", "CD90", "TIM3", "DNA_12", "C-Kit", "CD40", "Granz.B", "DNA_13", "HLADRB1", "GITR", "HLADPB1", "MART1", "DNA_14", "CCND1", "MYC", "CD56", "DNA_15", "LAMP1", "NOS2", "CD16"]
    # ["Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a", "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"]
    build_pyramid_jing(root, 5, 0)
    print('write_stats')
    write_stats_jing(root, channels_name, fileformat = 'jpeg')
    """

    raw = 'CID44971_spatial/raw' 
    root = zarr.open(raw, mode='r')
    build_background_pyramid_rgb(root)
    write_stats_background_rgb(root, fileformat ="png", metafilename="metadata.json")
