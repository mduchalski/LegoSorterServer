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
from tabulate import tabulate

channel = grpc.insecure_channel('localhost:50051')
stub = LegoSorter_pb2_grpc.LegoSorterStub(channel)

LOG_FIELDS = ['path', 'label_ref', 'hash', 'req_time', 'sort_idx', 'ymin', 'xmin', 'ymax', 'xmax', 'label', 'score']

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
    req = Messages__pb2.ImageRequest()
    req.image = im_bytes
    req.rotation = rot

    req_time = str(time.time())
    rpc_res = stub.processNextImage(req)

    m = hashlib.sha256()
    m.update(im_bytes)

    res = {
        'path': path,
        'label_ref': os.path.basename(path).split('_')[0],
        'req_time': req_time,
        'hash': m.hexdigest(),
        **_process_rpc_res(rpc_res)
    }

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

def print_summary(logpath):
    df = pd.read_csv(logpath)

    # Classification statistics
    df_c = df[df['sort_idx'] != -1]
    df_c['err_mask'] = (df_c['label'] != df_c['label_ref'])
    cacc = 1 - df_c['err_mask'].sum() / len(df_c)

    # Brick move statistics
    df_m = df[df['sort_idx'] == -1]
    df_m['err_mask'] = (df_m['label'] != df_m['label_ref'])
    macc = 1 - df_m['err_mask'].sum() / len(df_m)

    table = [['Number of images sent', len(df)],
             ['Number of classification results', len(df_c)],
             ['Number of aggregation results', len(df_m)],
             ['Classification error rate', f'{(1-cacc) * 100:.2f}% ({df_c["err_mask"].sum()} errors)'],
             ['Classification accuracy', f'{cacc * 100:.2f}%'],
             ['Aggregation error rate', f'{(1-macc) * 100:.2f}% ({df_m["err_mask"].sum()} errors)'],
             ['Aggregation accuracy', f'{macc * 100:.2f}%']]

    print(tabulate(table))

def main():
    parser = argparse.ArgumentParser('Utility for sending images over gRPC, mimicking LegoSorterApp')
    parser.add_argument('indir', nargs=1, help='Input directory containing folders with groups to send')
    parser.add_argument('-l', '--log', default='log.csv', help='Output log file (default: log.csv)')

    args = parser.parse_args()

    with open(args.log, 'w') as logfile:
        writer = csv.DictWriter(logfile, fieldnames=LOG_FIELDS)
        writer.writeheader()

    try:
        send_images(args.indir[0], args.log)
    finally:
        print_summary(args.log)

if __name__ == '__main__':
    main()
