# Copyright 2022 tao.jiang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple
from glob import glob
# import open3d as o3d

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from utils.ray_traversal import *

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 0
VIS = False
# IMPORTANT: not use tf32 which case pose error
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--start', type=int, default=0)
parser.add_argument('-e', '--end', type=int, default=10)
parser.add_argument('--a_pre', type=int, default=70, help='number of frames to aggregrate before the current frame')
parser.add_argument('--a_post', type=int, default=70, help='number of frames to aggregrate after the current frame')
parser.add_argument('--p_pre', type=int, default=10, help='number of frames to input before the current frame')
parser.add_argument('--p_post', type=int, default=10, help='number of frames to input after the current frame')
parser.add_argument('-i', '--input', type=str, default='/home/xl3136/lyft_kitti/train')
parser.add_argument('-v', '--vis', action='store_true')
# parser.add_argument('--input', type=str, default='/home/xl3136/nusc_kitti/mini_train_71')
args = parser.parse_args()


def pack(array):
  """ convert a boolean array into a bitwise array. """
  array = array.reshape((-1))

  #compressing bit flags.
  # yapf: disable
  compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
  # yapf: enable

  return np.array(compressed, dtype=np.uint8)


def main_func(pcd_files, pose_files, label_files, save_dir, index, args, vis=False):
    # Initialize parameters
    _device = torch.device('cuda')
    voxel_size = [0.2, 0.2, 0.2]
    point_cloud_range = [-51.2, -25.6, -2, 51.2, 25.6, 4.4]
    spatial_shape = [512, 256, 32]
    final_input = []
    final_label = []
    final_invalid = []
    
    # Read sensor pose
    pose_origin_inv = np.linalg.inv(np.load(pose_files[index])["{}_LIDAR_TOP".format(index)])

    
    # Read and process inputs
    for i in range(index-args.p_pre, index+1):
        points = []
        origins = []
        point_dict = np.load(pcd_files[i])
        pose_dict = np.load(pose_files[i])
        for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
            point = point_dict["{}_{}".format(i, lidar_name)]
            pose = pose_dict["{}_{}".format(i, lidar_name)]
            origin = pose[:, 3]

            point = pose @ point
            point = pose_origin_inv @ point
            origin = pose_origin_inv @ origin
            
            point = point[:3].T
            origin = origin[:3]
            origin = np.broadcast_to(origin, (point.shape[0], 3))
            
            points.append(point)
            origins.append(origin)

        # Aggregrate different lidar sensors
        points = np.concatenate(points, axis=0)
        origins = np.concatenate(origins, axis=0)
        pseudo_label = np.zeros(points.shape[0], dtype=np.uint8)

        rotation = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        rotation_matrix = rotation.rotation_matrix
        points = points @ rotation_matrix.T
        origins = origins @ rotation_matrix.T

        # Convert to torch tensor
        points_dev = torch.from_numpy(points).to(_device)
        origins_dev = torch.from_numpy(origins).to(_device)
        pseudo_label_dev = torch.from_numpy(pseudo_label).long().to(_device)
        _, input_voxel_state, _, _, _ = ray_traversal(
            origins_dev, points_dev, pseudo_label_dev,
            point_cloud_range, voxel_size, spatial_shape,
        )

        # Convert to numpy array
        final_input.append((input_voxel_state == 1).cpu().numpy())

    # Read and process ground truth
    for i in tqdm(range(index, index+args.p_post+1), leave=False):
        points = []
        origins = []
        labels = []
        start = max(i-args.a_pre, 0)
        end = min(i+args.a_post, len(pcd_files))
        point_dict = np.load(pcd_files[i])
        pose_dict = np.load(pose_files[i])
        label_dict = np.load(label_files[i])
        for j in range(start, end):
            # print(j)
            for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
                point = point_dict["{}_{}".format(j, lidar_name)]
                pose = pose_dict["{}_{}".format(j, lidar_name)]
                label = label_dict["{}_{}".format(j, lidar_name)]
                
                origin = pose[:, 3]
                point = pose @ point
                point = pose_origin_inv @ point
                origin = pose_origin_inv @ origin
                
                point = point[:3].T
                origin = origin[:3]
                origin = np.broadcast_to(origin, (point.shape[0], 3))
                
                points.append(point)
                origins.append(origin)
                labels.append(label)

        # Aggregrate different lidar sensors
        points = np.concatenate(points, axis=0)
        origins = np.concatenate(origins, axis=0)
        labels = np.concatenate(labels)

        rotation = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        rotation_matrix = rotation.rotation_matrix
        points = points @ rotation_matrix.T
        origins = origins @ rotation_matrix.T

        # Convert to torch tensor
        points_dev = torch.from_numpy(points).to(_device)
        origins_dev = torch.from_numpy(origins).to(_device)
        labels_dev = torch.from_numpy(labels).long().to(_device)
        _, voxel_state, voxel_label, _, _ = ray_traversal(
            origins_dev, points_dev, labels_dev,
            point_cloud_range, voxel_size, spatial_shape,
        )

        # Convert to numpy array
        invalid = torch.logical_and(voxel_state == -1, voxel_label == 0)
        final_label.append(voxel_label.cpu().numpy())
        final_invalid.append(invalid.cpu().numpy())

    # Save to disk
    final_input = np.stack(final_input, axis=0)
    final_label = np.stack(final_label, axis=0)
    final_invalid = np.stack(final_invalid, axis=0)
    
    if not vis:
        save_path = os.path.join(save_dir, "{:04d}".format(index))
        save_dict = {
            "input": final_input,
            "label": final_label,
            "invalid": final_invalid,
        }
        np.savez_compressed(save_path, **save_dict)
    else:     
        for i in range(index-args.p_pre, index+1):
            save_input = final_input[i-(index-args.p_pre)].astype(np.uint8)
            save_input = pack(save_input)
            save_input.tofile(os.path.join(save_dir, f"{str(i).zfill(6)}.bin"))
        for i in range(index, index+args.p_post+1):
            save_label = final_label[i-index].astype(np.uint16)
            save_label.tofile(os.path.join(save_dir, f"{str(i).zfill(6)}.label"))
            save_invalid = final_invalid[i-index].astype(np.uint8)
            save_invalid = pack(save_invalid)
            save_invalid.tofile(os.path.join(save_dir, f"{str(i).zfill(6)}.invalid"))


if __name__ == "__main__": 
    data_dir = args.input
    if args.vis:
        out_dir = os.path.join(args.input, "vis")
    else:
        out_dir = os.path.join(args.input, "voxel")
    velodyne_dir = os.path.join(data_dir, "point_cloud")
    
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    for i in tqdm(range(args.start, args.end)):
        velodyne_files = sorted(glob(os.path.join(velodyne_dir, "{:04d}".format(i), "*_point.npz")))
        poses_files = sorted(glob(os.path.join(velodyne_dir, "{:04d}".format(i), "*_pose.npz")))
        labels_files = sorted(glob(os.path.join(velodyne_dir, "{:04d}".format(i), "*_label.npz")))
        
        save_dir = os.path.join(out_dir, "{:04d}".format(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        start_frame = args.p_pre
        end_frame = len(velodyne_files) - args.p_post
        for j in tqdm(range(start_frame, end_frame), leave=False):
            main_func(
                velodyne_files, 
                poses_files, 
                labels_files, 
                save_dir, 
                j,
                args,
                args.vis,
            )
