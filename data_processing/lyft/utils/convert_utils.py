import os
import copy
import pathlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, transform_matrix, view_points
from pyquaternion import Quaternion
from utils.ray_traversal import *
from utils.geometry_utils import *


def loat_one_frame(sample_token, lyft_data):
    points = {}
    labels = {}
    poses = {}
    instance_dict = {}
    sample = lyft_data.get('sample', sample_token)

    # Get all instances in the frame.
    # instance_tokens = [lyft_data.get('sample_annotation', token) for token in sample['anns']]
    for token in sample['anns']:
        sample_annotation = lyft_data.get('sample_annotation', token)
        instance_token = sample_annotation['instance_token']
        instance_dict[instance_token] = {}
        for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
            _, instance_box, _ = lyft_data.get_sample_data(
                sample['data'][lidar_name],
                box_vis_level=BoxVisibility.NONE,
                selected_anntokens=[token]
            )
            instance_dict[instance_token][lidar_name] = instance_box[0]


    # Get the point cloud and sensor pose in the frame.
    for lidar_sensor in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
        lidar_data = lyft_data.get('sample_data', sample['data'][lidar_sensor])
        ego_pose = lyft_data.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_pose = lyft_data.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        # Get global pose
        ego_pose = transform_matrix(
            ego_pose['translation'], 
            Quaternion(ego_pose['rotation']), 
            inverse=False
        )
        lidar_pose = transform_matrix(
            lidar_pose['translation'], 
            Quaternion(lidar_pose['rotation']), 
            inverse=False
        )
        global_pose = np.dot(ego_pose, lidar_pose)

        # Get point cloud
        lidar_filepath = pathlib.Path(lyft_data.data_path / lidar_data['filename'])
        point_cloud = LidarPointCloud.from_file(lidar_filepath)
        point_cloud.points[-1, :] = 1
        points[lidar_sensor] = point_cloud.points
        labels[lidar_sensor] = np.ones(point_cloud.points.shape[1], dtype=np.uint8)
        poses[lidar_sensor] = global_pose
        
    return points, poses, labels, instance_dict


def convert_one_frame(cur_index, pre, post, points, poses, labels, instance_dict, **kwargs):
    pre = max(0, cur_index - pre)
    post = min(len(instance_dict), cur_index + post)

    final_points = {}
    final_labels = {}
    final_poses = {}

    for i in tqdm(range(pre, post), leave=False):
        if i != cur_index:
            for instance_token in instance_dict[i]:
                if instance_token not in instance_dict[cur_index]:
                    for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
                        mask = points_in_box(instance_dict[i][instance_token][lidar_name].corners(), points[i][lidar_name][:-1])
                        points[i][lidar_name] = points[i][lidar_name][:, ~mask]
                        labels[i][lidar_name] = labels[i][lidar_name][~mask]
                else:
                    for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
                        mask = points_in_box(instance_dict[i][instance_token][lidar_name].corners(), points[i][lidar_name][:-1])  
                        if not np.any(mask): continue
                        dst_box = np.copy(instance_dict[cur_index][instance_token]["LIDAR_TOP"].corners()).T
                        src_box = np.copy(instance_dict[i][instance_token][lidar_name].corners()).T
                        c_dst = np.mean(dst_box, axis=0)
                        c_src = np.mean(src_box, axis=0)
                        dst_box -= c_dst
                        src_box -= c_src
                        A = np.dot(dst_box.T, src_box)
                        U, _, Vt = np.linalg.svd(A, full_matrices=False)
                        V = Vt.T
                        R = np.dot(V, U.T)
                        c = np.expand_dims(c_dst - c_src @ R, 1)

                        curr_points = points[i][lidar_name][:, mask]
                        # print(R.shape, c.shape, curr_points.shape)
                        curr_points[:-1] = np.add(R.T @ curr_points[:-1], c)
                        src_pose = poses[i][lidar_name]
                        dst_pose = poses[cur_index]["LIDAR_TOP"]
                        new_points = np.linalg.inv(src_pose) @ dst_pose @ curr_points
                        points[i][lidar_name][:, mask] = new_points

                        instance_name = instance_dict[i][instance_token][lidar_name].name
                        labels[i][lidar_name][mask] = kwargs["class_map"][instance_name]
        else:
            for instance_token in instance_dict[i]:
                for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
                    mask = points_in_box(instance_dict[i][instance_token][lidar_name].corners(), points[i][lidar_name][:-1])
                    points[i][lidar_name] = points[i][lidar_name][:, ~mask]
                    labels[i][lidar_name] = labels[i][lidar_name][~mask]

            final_points["{}_{}".format(i-pre, lidar_name)] = np.dot(kwargs["kitti_to_lyft_lidar_inv"].rotation_matrix, points[i][lidar_name][:3]).T
            final_labels["{}_{}".format(i-pre, lidar_name)] = labels[i][lidar_name]
            final_poses["{}_{}".format(i-pre, lidar_name)] = poses[i][lidar_name]
    
    return final_points, final_poses, final_labels


def convert_one_scene(scene_index, lyft_data, **kwargs):
    save_path = os.path.join(kwargs["output_path"], "point_cloud", "{:04d}".format(scene_index))
    os.makedirs(save_path, exist_ok=True)
    
    scene = lyft_data.scene[scene_index]

    # Get tokens of all samples in the scene.
    sample_tokens = [scene["first_sample_token"]]
    while lyft_data.get('sample', sample_tokens[-1])['next']:
        sample_tokens.append(lyft_data.get('sample', sample_tokens[-1])['next'])

    # Load lidar calibration
    lidar_calib = {}
    for lidar_name in ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]:
        cs_token = lyft_data.get(
            "sample_data", 
            lyft_data.get("sample", sample_tokens[0])["data"][lidar_name]
        )["calibrated_sensor_token"]
        lidar_calib[lidar_name] = lyft_data.get("calibrated_sensor", cs_token)
        lidar_to_ego = transform_matrix(
            lidar_calib[lidar_name]["translation"], 
            Quaternion(lidar_calib[lidar_name]["rotation"]), 
            inverse=False
        )
        lidar_to_ego_kitti = np.dot(lidar_to_ego, kwargs["kitti_to_lyft_lidar"].transformation_matrix)
        lidar_to_ego_kitti_inv = np.linalg.inv(lidar_to_ego_kitti)
        lidar_calib[lidar_name] = {
            "Tr": lidar_to_ego_kitti, 
            "Tr_inv": lidar_to_ego_kitti_inv
        }
    
    # Load all frames in the scene.
    points = []
    poses = []
    labels = []
    instance_tokens = []

    # Load all frames in the scene.
    print("Loading frames...")
    for sample_token in tqdm(sample_tokens):
        frame_points, frame_poses, frame_labels, frame_instances = loat_one_frame(
            sample_token, 
            lyft_data
        )
        points.append(frame_points)
        poses.append(frame_poses)
        labels.append(frame_labels)
        instance_tokens.append(frame_instances)
    print("Done.\n")

    # Convert all frames in the scene.
    print("Converting frames...")
    for i in tqdm(range(len(sample_tokens))):
        final_points, final_poses, final_labels = convert_one_frame(
            i, 
            kwargs["aggregate_pre"], 
            kwargs["aggregate_post"], 
            copy.deepcopy(points), 
            copy.deepcopy(poses), 
            copy.deepcopy(labels), 
            instance_tokens, 
            **kwargs
        )

        points_path = os.path.join(save_path, "{:04d}_point".format(i))
        poses_path = os.path.join(save_path, "{:04d}_pose".format(i))
        labels_path = os.path.join(save_path, "{:04d}_label".format(i))
        np.savez_compressed(points_path, **final_points)
        np.savez_compressed(poses_path, **final_poses)
        np.savez_compressed(labels_path, **final_labels)
