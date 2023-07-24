import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sequence', type=int, default=0, help='sequence index')
parser.add_argument('-b', '--start', type=int, default=0, help='start frame')
args = parser.parse_args()

counts = {str(i): 0 for i in range(-1, 11)}
data_dir = '/vast/xl3136/lyft_kitti/voxel'
save_dir = 'temp/'
sequence = '{:04d}'.format(args.start + args.sequence)
save_path = os.path.join(save_dir, sequence + '.json')
for frame in tqdm(os.listdir(os.path.join(data_dir, sequence)), leave=False):
    data = np.load(os.path.join(data_dir, sequence, frame))
    label = data['label']
    invalid = data['invalid']
    label[invalid] = -1

    label_arr, count_arr = np.unique(label, return_counts=True)
    for label, count in zip(label_arr, count_arr):
        counts[str(label)] += count

for key in counts:
    counts[key] = str(counts[key])

with open(save_path, 'w') as f:
    json.dump(counts, f)
