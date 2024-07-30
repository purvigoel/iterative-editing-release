import os

sns = ['S1', 'S5', 'S6', 'S7', 'S8']
ROOT = '/scratch/groups/syyeung/hmr_datasets/h36m_train/'

all_fnames = []

for sn in sns:
	dirname = f'/scratch/groups/syyeung/hmr_datasets/h36m_train/{sn}/Videos/'
	fs = os.listdir(dirname)
	all_fnames.extend(fs)

all_action = [f.split('.')[0] for f in all_fnames]
print(sorted(list(set(all_action))))