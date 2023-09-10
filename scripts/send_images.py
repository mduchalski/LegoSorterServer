#!/usr/bin/env python3
'''
Utility for sending images over gRPC, mimicking LegoSorterApp
'''
import os
import cv2
import csv
import time
import grpc
import hashlib
import argparse
import numpy as np
import pandas as pd

import lego_sorter_server.generated.Messages_pb2 as Messages__pb2
import lego_sorter_server.generated.LegoSorter_pb2_grpc as LegoSorter_pb2_grpc

from tqdm import tqdm

channel = grpc.insecure_channel('localhost:50051')
stub = LegoSorter_pb2_grpc.LegoSorterStub(channel)
image_idx = 0

LOG_FIELDS = ['image_idx', 'path', 'label_ref', 'hash', 'req_time', 'resp_time', 'sort_idx', 'ymin', 'xmin', 'ymax', 'xmax', 'label', 'score']

def _process_rpc_res(rpc_res):
    if len(rpc_res.packet) > 0:
        brick = rpc_res.packet[0]
        return {
            'sort_idx': brick.index,
            'ymin': brick.bb.ymin,
            'xmin': brick.bb.xmin,
            'ymax': brick.bb.ymax,
            'xmax': brick.bb.xmax,
            'label': brick.bb.label,
            'score': brick.bb.score
        }
    return {}

def send_image(im_bytes, path, log_path, rot=0):
    global image_idx
    
    req = Messages__pb2.ImageRequest()
    req.image = im_bytes
    req.rotation = rot

    req_time = str(time.time())
    rpc_res = stub.processNextImage(req)
    resp_time = str(time.time())

    m = hashlib.sha256()
    m.update(im_bytes)

    res = {
        'image_idx': image_idx,
        'path': path,
        'label_ref': os.path.basename(path).split('_')[0],
        'req_time': req_time,
        'resp_time': resp_time,
        'hash': m.hexdigest(),
        **_process_rpc_res(rpc_res)
    }
    image_idx += 1

    with open(log_path, 'a') as logfile:
        writer = csv.DictWriter(logfile, fieldnames=LOG_FIELDS)
        writer.writerow(res)

    return res

def send_images(indir, log_path):
    for root, dirs, files in tqdm(list(os.walk(indir))):
        if not (len(dirs) == 0 and len(files) != 0 and
                all(file.endswith('.png') for file in files)):
            continue

        for file in sorted(files):
            path = os.path.join(root, file)
            with open(path, 'rb') as infile:
                im_bytes = infile.read()
            send_image(im_bytes, path, log_path)

        im_ = cv2.imread(path)
        im_blank = 255 * np.ones_like(im_)
        _, im = cv2.imencode('.png', im_blank)
        im_enc = im.tobytes()
        send_image(im_enc, path, log_path)


def main():
    parser = argparse.ArgumentParser('Utility for sending images over gRPC, mimicking LegoSorterApp')
    parser.add_argument('indir', nargs=1, help='Input directory containing folders with groups to send')
    parser.add_argument('-l', '--log', default='logs/send.csv', help='Output log file (default: logs/send.csv)')

    args = parser.parse_args()

    with open(args.log, 'w') as logfile:
        writer = csv.DictWriter(logfile, fieldnames=LOG_FIELDS)
        writer.writeheader()

    send_images(args.indir[0], args.log)

if __name__ == '__main__':
    main()
