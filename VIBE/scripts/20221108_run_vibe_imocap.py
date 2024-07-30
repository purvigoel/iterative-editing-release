import os
import os.path as osp
import sys
import subprocess
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--action', type=str)
args = parser.parse_args()
action = args.action
exp_dir = osp.join('/home/groups/syyeung/wangkua1/data/internet_dataset',
                   args.action)
all_videos = [f for f in os.listdir(exp_dir) if f.endswith('.mp4')]
for vid_name in all_videos:
    cmd = [
        'python', 'demo2.py', '--vid_file',
        f'/home/groups/syyeung/wangkua1/data/internet_dataset/{action}/{vid_name}',
        '--output_folder',
        f'/home/groups/syyeung/wangkua1/data/internet_dataset/{action}/{vid_name}.vibe',
        '--detector', 'maskrcnn'
    ]

    print(cmd)
    subprocess.call(cmd)