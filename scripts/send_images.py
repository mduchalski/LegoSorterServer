#!/usr/bin/env python3
'''
Utility for sending images over gRPC, mimicking LegoSorterApp
'''
import os
import cv2
import grpc
import argparse
import numpy as np

import lego_sorter_server.generated.Messages_pb2 as Messages__pb2
import lego_sorter_server.generated.LegoSorter_pb2_grpc as LegoSorter_pb2_grpc

from tqdm import tqdm

channel = grpc.insecure_channel('localhost:50051')
stub = LegoSorter_pb2_grpc.LegoSorterStub(channel)

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

def send_image(im_bytes, rot=0):
    req = Messages__pb2.ImageRequest()
    req.image = im_bytes
    req.rotation = rot
    rpc_res = stub.processNextImage(req)
    res = _process_rpc_res(rpc_res)
    return res

def send_images(indir):
    for root, dirs, files in tqdm(list(os.walk(indir))):
        if not (len(dirs) == 0 and len(files) != 0 and
                all(file.endswith('.png') for file in files)):
            continue

        for file in files:
            path = os.path.join(root, file)
            with open(path, 'rb') as infile:
                im_bytes = infile.read()
            send_image(im_bytes)

        im_ = cv2.imread(path)
        im_blank = 255 * np.ones_like(im_)
        _, im = cv2.imencode('.png', im_blank)
        im_enc = im.tobytes()
        send_image(im_enc)

def main():
    parser = argparse.ArgumentParser('Utility for sending images over gRPC, mimicking LegoSorterApp')
    parser.add_argument('indir', nargs=1, help='Input directory containing folders with groups to send')
    args = parser.parse_args()
    send_images(args.indir[0])

if __name__ == '__main__':
    main()
