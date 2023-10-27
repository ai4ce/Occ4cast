import os
import copy
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from pyquaternion import Quaternion
from utils.geometry_utils import *


LIDAR_NAMES = ["LIDAR_TOP"]
CAM_NAMES = [
    "CAM_FRONT", 
    "CAM_FRONT_LEFT", 
    "CAM_FRONT_RIGHT", 
    "CAM_BACK", 
    "CAM_BACK_LEFT", 
    "CAM_BACK_RIGHT"
]

def load_one_frame(sample_token, nusc_data, lidar_names):
    points = {}
    labels = {}
    poses = {}
    images = {}
    instance_dict = {}
    sample = nusc_data.get('sample', sample_token)

    # Get all instances in the frame.
    for token in sample['anns']:
        sample_annotation = nusc_data.get('sample_annotation', token)
        instance_token = sample_annotation['instance_token']
        instance_dict[instance_token] = {}
        for lidar_name in lidar_names:
            _, instance_box, _ = nusc_data.get_sample_data(
                sample['data'][lidar_name],
                box_vis_level=BoxVisibility.NONE,
                selected_anntokens=[token]
            )
            instance_dict[instance_token][lidar_name] = instance_box[0]

    # Get all images in the frame.
    for cam_name in CAM_NAMES:
        cam_data = nusc_data.get('sample_data', sample['data'][cam_name])
        cam_filepath = os.path.join(nusc_data.dataroot, cam_data['filename'])
        images[cam_name] = Image.open(cam_filepath)

    # Get the ego pose for cameras.
    cam_ego_pose = nusc_data.get('ego_pose', cam_data['ego_pose_token'])
    cam_ego_pose = transform_matrix(
        cam_ego_pose['translation'],
        Quaternion(cam_ego_pose['rotation']),
        inverse=False
    )

    # Get the point cloud and sensor pose in the frame.
    for lidar_sensor in lidar_names:
        lidar_data = nusc_data.get('sample_data', sample['data'][lidar_sensor])
        lidar_seg_data = nusc_data.get('lidarseg', sample['data'][lidar_sensor])
        ego_pose = nusc_data.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_pose = nusc_data.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
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
        lidar_filepath = (pathlib.Path(nusc_data.dataroot) / lidar_data['filename'])
        point_cloud = LidarPointCloud.from_file(str(lidar_filepath))
        point_cloud.points[-1, :] = 1
        
        # Get labels
        lidarseg_filename = (pathlib.Path(nusc_data.dataroot) / lidar_seg_data["filename"])
        lidarseg = np.fromfile(str(lidarseg_filename), dtype=np.uint8)

        # Remove ego vehicle points, ego vehicle has label 31.
        point_cloud.points = np.delete(point_cloud.points, lidarseg == 31, axis=1)
        lidarseg = np.delete(lidarseg, lidarseg == 31)

        # add to list
        points[lidar_sensor] = point_cloud.points
        labels[lidar_sensor] = lidarseg

        # load pose
        poses[lidar_sensor] = global_pose
        
    return points, poses, labels, images, instance_dict, cam_ego_pose


def convert_one_frame(cur_index, pre, post, points, poses, labels, lidar_names, instance_dict, kwargs):
    pre = max(0, cur_index - pre)
    post = min(len(instance_dict), cur_index + post)

    final_points = {}
    final_labels = {}
    final_poses = {}

    for i in tqdm(range(pre, post), leave=False):
        if i != cur_index:
            for instance_token in instance_dict[i]:
                if instance_token not in instance_dict[cur_index]:
                    for lidar_name in lidar_names:
                        mask = points_in_box(instance_dict[i][instance_token][lidar_name].corners(1.1), points[i][lidar_name][:-1])
                        points[i][lidar_name] = np.delete(points[i][lidar_name], mask, axis=1)
                        labels[i][lidar_name] = np.delete(labels[i][lidar_name], mask)
                else:
                    for lidar_name in lidar_names:
                        mask = points_in_box(instance_dict[i][instance_token][lidar_name].corners(1.1), points[i][lidar_name][:-1])  
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

                        curr_points = np.copy(points[i][lidar_name][:, mask])
                        curr_points[:-1] = np.add(R.T @ curr_points[:-1], c)
                        src_pose = poses[i][lidar_name]
                        dst_pose = poses[cur_index]["LIDAR_TOP"]
                        new_points = np.linalg.inv(src_pose) @ dst_pose @ curr_points
                        points[i][lidar_name][:, mask] = new_points

        for lidar_name in lidar_names:
            final_points["{}_{}".format(i, lidar_name)] = points[i][lidar_name]
            final_labels["{}_{}".format(i, lidar_name)] = labels[i][lidar_name]
            final_poses["{}_{}".format(i, lidar_name)] = poses[i][lidar_name]
        
    return final_points, final_poses, final_labels


def convert_one_scene(scene_index, nusc_data, kwargs):
    save_path_pcd = os.path.join(kwargs["output_path"], "point_cloud", "{:04d}".format(scene_index))
    save_path_calib = os.path.join(kwargs["output_path"], "calib", "{:04d}".format(scene_index))
    save_path_image = os.path.join(kwargs["output_path"], "image", "{:04d}".format(scene_index))
    save_path_cam_pose = os.path.join(kwargs["output_path"], "cam_pose", "{:04d}".format(scene_index))
    save_path_image_downsample2 = os.path.join(kwargs["output_path"], "image_downsample", "1_2", "{:04d}".format(scene_index))
    save_path_image_downsample4 = os.path.join(kwargs["output_path"], "image_downsample", "1_4", "{:04d}".format(scene_index))
    for path in [save_path_pcd, save_path_calib, save_path_image, save_path_cam_pose, save_path_image_downsample2, save_path_image_downsample4]:
        os.makedirs(path, exist_ok=True)
    
    scene = nusc_data.scene[scene_index]

    # Get tokens of all samples in the scene.
    sample_tokens = [scene["first_sample_token"]]
    while nusc_data.get('sample', sample_tokens[-1])['next']:
        sample_tokens.append(nusc_data.get('sample', sample_tokens[-1])['next'])

    # Load lidar names because not all scenes have three lidars.
    lidar_names = []
    for lidar_name in LIDAR_NAMES:
        if lidar_name in nusc_data.get("sample", sample_tokens[0])["data"]:
            lidar_names.append(lidar_name)

    # Load and save camera calibration.
    cam_intrinsics = {}
    cam_extrinsics = {}
    for cam_name in CAM_NAMES:
        cs_token = nusc_data.get(
            "sample_data",
            nusc_data.get("sample", sample_tokens[0])["data"][cam_name]
        )
        cs = nusc_data.get("calibrated_sensor", cs_token["calibrated_sensor_token"])
        intrinsic = np.array(cs["camera_intrinsic"])
        extrinsic = transform_matrix(
            cs["translation"],
            Quaternion(cs["rotation"]),
            inverse=False
        )
        cam_intrinsics[cam_name] = intrinsic # 3x3 matrix
        cam_extrinsics[cam_name] = extrinsic # 4x4 matrix

        intrinsic.tofile(os.path.join(save_path_calib, "{}_intrinsic.bin".format(cam_name)))
        extrinsic.tofile(os.path.join(save_path_calib, "{}_extrinsic.bin".format(cam_name)))
    
    # Load all frames in the scene.
    points = []
    poses = []
    labels = []
    instance_tokens = []

    # Load all frames in the scene and convert images.
    print("Loading frames...")
    for i, sample_token in enumerate(tqdm(sample_tokens)):
        frame_points, frame_poses, frame_labels, frame_images, frame_instances, frame_cam_pose = load_one_frame(
            sample_token, 
            nusc_data,
            lidar_names
        )
        points.append(frame_points)
        poses.append(frame_poses)
        labels.append(frame_labels)
        instance_tokens.append(frame_instances)

        # Save images
        for cam_name in CAM_NAMES:
            save_filename = "{:04d}_{}".format(i, cam_name)
            image_1_2 = frame_images[cam_name].resize((int(frame_images[cam_name].width / 2), int(frame_images[cam_name].height / 2)))
            image_1_4 = frame_images[cam_name].resize((int(frame_images[cam_name].width / 4), int(frame_images[cam_name].height / 4)))
            frame_images[cam_name].save(os.path.join(save_path_image, save_filename + ".jpg"))
            image_1_2.save(os.path.join(save_path_image_downsample2, save_filename + ".jpg"))
            image_1_4.save(os.path.join(save_path_image_downsample4, save_filename + ".jpg"))

        # Save poses
        frame_cam_pose.tofile(os.path.join(save_path_cam_pose, "{:04d}_pose.bin".format(i)))

    print("Done.\n")

    # Convert all frames in the scene.
    print("Converting frames...")
    for i in tqdm(range(len(sample_tokens))):
    # for i in tqdm(range(38, 62)):
        final_points, final_poses, final_labels = convert_one_frame(
            i, 
            kwargs["aggregate_pre"], 
            kwargs["aggregate_post"], 
            copy.deepcopy(points), 
            copy.deepcopy(poses), 
            copy.deepcopy(labels),
            lidar_names,
            instance_tokens, 
            kwargs
        )

        points_path = os.path.join(save_path_pcd, "{:04d}_point".format(i))
        poses_path = os.path.join(save_path_pcd, "{:04d}_pose".format(i))
        labels_path = os.path.join(save_path_pcd, "{:04d}_label".format(i))
        np.savez_compressed(points_path, **final_points)
        np.savez_compressed(poses_path, **final_poses)
        np.savez_compressed(labels_path, **final_labels)
