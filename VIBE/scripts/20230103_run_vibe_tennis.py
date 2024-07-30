import os
import os.path as osp
import sys
import subprocess
from itertools import product
from collections import OrderedDict
import argparse
import ipdb

video_names_dict = OrderedDict()
video_names_dict["carlos"] = r"Carlos Alcaraz  Court Level Practice.mp4"
video_names_dict["casper"] = r"Casper Ruud  Forehand Backhand & Volley [4k 60fps].mp4"
video_names_dict["college-baltazar"] = r"College Tennis Recruiting Video - Baltazar Wiger-Nordås - Fall 2022.mp4"
video_names_dict["dominic"] = r"Dominic Thiem Serve & Return Training Indian Wells Court Level View - ATP Tennis Serve Technique.mp4"
video_names_dict["emma"] = r"Emma Raducanu Court Level Practice Session 2022  Ground Strokes Exercises & More! (4K 60FPS).mp4"
video_names_dict["felix"] = r"Felix Auger Aliassime Court Level Practice  Australian Open (4K 60FPS).mp4"
video_names_dict["college-gabe"] = r"Gabe Mills  - College Tennis Recruiting Video.mp4"
video_names_dict["jannik"] = r"Jannik Sinner Side View & Slow Motion  Court Level Practice 2022 (4K 60FPS).mp4"
video_names_dict["novak"] = r"Novak Djokovic  forehand backhand practice.mp4"
video_names_dict["stan"] = r"Stan Wawrinka Training 2018 Court Level View (HD).mp4"
video_names_dict["college-laura"] = r"College Tennis Recruiting Video - Fall 2021- Laura Schmitz.mp4"



parser = argparse.ArgumentParser()
parser.add_argument('--vid_key', type=str)
args = parser.parse_args()
# video_name = video_names_dict[args.vid_key]
exp_dir = '/home/groups/syyeung/wangkua1/videos/youtube'
vid_file = osp.join(exp_dir, args.vid_key + '.mp4')

out_root = '/home/groups/syyeung/wangkua1/tennis/vibe'
out_dir = osp.join(out_root, args.vid_key)
cmd = [
    'python', 'demo2.py', '--vid_file',
    vid_file,
    '--output_folder',
    out_dir,
    '--detector', 'maskrcnn'
]

print(cmd)
subprocess.call(cmd)