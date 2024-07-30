import os
import os.path as osp
import sys
import subprocess
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, required=True)
args = parser.parse_args()

actions = [
    'tennis_serve', 'tennis_swing', 'baseball_pitch', 'baseball_swing',
    'golf_swing'
]
for action in actions:
    cmd = [
        'python', 'demo2.py', '--vid_file',
        f'/home/groups/syyeung/wangkua1/data/mymocap/trimmed1/{action}.{args.n}.mp4',
        '--output_folder',
        f'/home/groups/syyeung/wangkua1/nemo/exps/mymocap_{action}_vs',
        '--detector', 'maskrcnn',
        '--tracking_method', 'pose',
        '--run_smplify'
    ]

    print(cmd)
    subprocess.call(cmd)