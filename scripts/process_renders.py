#!/usr/bin/env python3
'''
Helper script for processing dataset of LDRAW LEGO brick renders

See: https://mostwiedzy.pl/pl/open-research-data/ldraw-based-renders-of-lego-bricks-moving-on-a-conveyor-belt,524095918518778-0
'''
import os
import sys
import cv2
import glob
import shutil
import random
import argparse
import numpy as np

from tqdm import tqdm
from operator import itemgetter

TRANSFORM_LUT = {
    'blur': lambda im, size: cv2.blur(im, (size, size)),
    'brightness': lambda im, range: cv2.addWeighted(im, 1, im, 0, random.uniform(-range, +range)),
    'contrast': lambda im, range: cv2.addWeighted(im, random.uniform(1 - range, 1 + range), im, 0, 0)
}

def _parse_img_path(path):
    _, file = os.path.split(path)
    file, _ = os.path.splitext(file)
    brick, color, idx, timestamp = file.split('_')

    return {
        'path': path,
        'file': file,
        'brick': brick,
        'color': color,
        'idx': int(idx),
        'timestamp': int(timestamp),
    }

def _process_group(grp, outdir, transforms):
    if len(grp) != 10:
        return # ignore w/o error

    grp_outdir = os.path.join(outdir, grp[0]['file'])
    os.mkdir(grp_outdir)

    for img in grp:
        im = cv2.imread(img['path'])
        
        for name, param in transforms:
            im = TRANSFORM_LUT[name](im, param)

        out_path = os.path.join(grp_outdir, img['file']) + '.png'
        cv2.imwrite(out_path, im)

    print(grp_outdir)

def process_renders(indir, outdir, count, shuffle, transforms):
    imgs = glob.glob(os.path.join(indir, '**', '*.png'))
    imgs = sorted(map(_parse_img_path, imgs),
                  key=lambda img: (img['brick'], img['timestamp'])) + [{'idx': 0}]
    
    grp, grps = [], []
    for img, next_img in zip(imgs, imgs[1:]):
        grp.append(img)
        if next_img['idx'] == 0:
            grps.append(grp)
            grp = []

    if shuffle == True:
        random.shuffle(grps)

    if os.path.exists(outdir) == True:
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    for idx in tqdm(range(min(count, len(grps)))):
        _process_group(grps[idx], outdir, transforms)

def main():
    parser = argparse.ArgumentParser('Helper script for processing dataset of LDRAW LEGO brick renders')
    parser.add_argument('indir', nargs=1, help='Input directory containing the dataset')
    parser.add_argument('-o', '--outdir', default=os.path.join(os.path.curdir, 'renders'), help='WILL BE CLEARED IF EXISTS! Output directory containing processed images (default: renders)')
    parser.add_argument('-c', '--count', type=int, default=100, help='Number of render groups to process (default: 100)')
    parser.add_argument('-s', '--shuffle', default=False, action='store_true', help='Shuffle groups of renders before processing')
    parser.add_argument('-r', '--brightness', type=float, default=None, help='Change image brightness through deviating the mean pixel value by a maximum of given amount')
    parser.add_argument('-b', '--blur', type=int, default=None, help='Add Gaussian blur to each image before saving')
    parser.add_argument('-n', '--contrast', type=float, default=None, help='Change image contrast through multiplying pixel values by (1 +/- maximum of given amount)')

    args = parser.parse_args()

    transforms = [(name, param) for (name, param) in vars(args).items()
                  if param != None and ['blur', 'brightness', 'contrast'].count(name) == 1]

    process_renders(args.indir[0], args.outdir, args.count, args.shuffle, transforms)

    with open(os.path.join(args.outdir, 'COMMAND.txt'), 'w') as logfile:
        logfile.write(' '.join(sys.argv))

if __name__ == '__main__':
    main()
