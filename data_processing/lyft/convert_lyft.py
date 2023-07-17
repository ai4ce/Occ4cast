import os
import argparse
import yaml
from pyquaternion import Quaternion
import numpy as np
from lyft_dataset_sdk.lyftdataset import LyftDataset
from utils.convert_utils import convert_one_scene


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='/home/xl3136/lyft/train', help='path to the dataset')
parser.add_argument('-s', '--start', type=int, default=0, help='start index of the scene')
parser.add_argument('-e', '--end', type=int, default=100, help='end index of the scene')
parser.add_argument('-o', '--output_path', type=str, default='/home/xl3136/lyft_kitti/train', help='path to the output')
parser.add_argument('--a_pre', type=int, default=70, help='number of frames to aggregrate before the current frame')
parser.add_argument('--a_post', type=int, default=70, help='number of frames to aggregrate after the current frame')
parser.add_argument('--p_pre', type=int, default=10, help='number of frames to input before the current frame')
parser.add_argument('--p_post', type=int, default=10, help='number of frames to input after the current frame')
args = parser.parse_args()

kitti_to_lyft_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
kitti_to_lyft_lidar_inv = kitti_to_lyft_lidar.inverse
with open('config/class_map.yaml', 'r') as f:
    class_map = yaml.safe_load(f)

lyft_data = LyftDataset(data_path=args.data_path, json_path=args.data_path + '/{}_data'.format(args.data_path.split("/")[-1]), verbose=True)

config = {
    "aggregate_pre": args.a_pre,
    "aggregate_post": args.a_post,
    "prediction_pre": args.p_pre,
    "prediction_post": args.p_post,
    "kitti_to_lyft_lidar": kitti_to_lyft_lidar,
    "kitti_to_lyft_lidar_inv": kitti_to_lyft_lidar_inv,
    "class_map": class_map,
    "output_path": args.output_path
}

os.makedirs(args.output_path, exist_ok=True)

for scene_index in range(args.start, args.end):
    convert_one_scene(scene_index, lyft_data, **config)