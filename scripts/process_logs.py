#!/usr/bin/env python3
import os
import argparse
import pandas as pd

from tabulate import tabulate
from datetime import datetime

def merge_logs(send_log, recv_log):
    df_send = pd.read_csv(send_log, index_col='image_idx')
    df_recv = pd.read_csv(recv_log, index_col='image_idx')
    df = df_send.join(df_recv)
    return df

def calculate_derived(df):
    return df

def get_summary(df, ctime):
    # Classification statistics
    df_c = df[df['sort_idx'] != -1]
    df_c['err_mask'] = (df_c['label'] != df_c['label_ref'])
    cacc = 1 - df_c['err_mask'].sum() / len(df_c)

    # Brick move statistics
    df_m = df[df['sort_idx'] == -1]
    df_m['err_mask'] = (df_m['label'] != df_m['label_ref'])
    macc = 1 - df_m['err_mask'].sum() / len(df_m)

    table = [['Timestamp', ctime.strftime("%Y-%m-%d %H:%M")],
             ['First image path', df.loc[0, 'path']],
             ['Number of images sent', len(df)],
             ['Number of classification results', len(df_c)],
             ['Number of aggregation results', len(df_m)],
             ['Classification error rate', f'{(1-cacc) * 100:.2f}% ({df_c["err_mask"].sum()} errors)'],
             ['Classification accuracy', f'{cacc * 100:.2f}%'],
             ['Aggregation error rate', f'{(1-macc) * 100:.2f}% ({df_m["err_mask"].sum()} errors)'],
             ['Aggregation accuracy', f'{macc * 100:.2f}%']]

    summary = tabulate(table)
    return '\n'.join('# ' + ln for ln in summary.splitlines())

def main():
    parser = argparse.ArgumentParser('process_sorter_log.py')
    parser.add_argument('-s', '--send', default='logs/send.csv', help='Input send log file (default: logs/send.csv)')
    parser.add_argument('-r', '--recv', default='logs/recv.csv', help='Input recv log file (default: logs/recv.csv)')
    parser.add_argument('-o', '--output', default='logs/merged.csv', help='Output file containing merged logs (default: logs/merged.csv)')
    args = parser.parse_args()
    
    ctime = datetime.fromtimestamp(os.path.getctime(args.send))
    df = merge_logs(args.send, args.recv)
    df = calculate_derived(df)
    summary = get_summary(df, ctime)
    print(summary)

    with open(args.output, 'w') as outfile:
        outfile.write(summary + '\n')
        df.to_csv(outfile)

if __name__ == '__main__':
    main()
