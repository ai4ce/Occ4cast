from argoverse.map_representation.map_api import ArgoverseMap
import tempfile
import argparse
import glob
import copy
import logging
from pathlib import Path
from typing import Any
from argoverse.data_loading.frame_label_accumulator import PerFrameLabelAccumulator
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.pkl_utils import load_pkl_dictionary
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.data_loading.object_label_record import json_label_dict_to_obj_record
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
import os
import shutil
import numpy as np
import sys
from argoverse.utils.camera_stats import RING_CAMERA_LIST
from collections import OrderedDict
import math
import pickle as pkl
from PIL import Image
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
LIDAR_NAMES = ["LIDAR_TOP"]
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

label_remap = {
    'FREE':0,
    'UNLABELED': 1,
    'VEHICLE': 2, 
    'PEDESTRIAN': 3, 
    'ON_ROAD_OBSTACLE': 4, 
    'LARGE_VEHICLE': 5, 
    'BICYCLE': 6, 
    'BICYCLIST': 7, 
    'BUS': 8, 
    'OTHER_MOVER': 9,
    'TRAILER': 10, 
    'MOTORCYCLIST': 11, 
    'MOPED':12, 
    'MOTORCYCLE': 13, 
    'STROLLER': 14, 
    'EMERGENCY_VEHICLE': 15, 
    'ANIMAL': 16
}
def points_in_box(corners, points: np.ndarray):
    """
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: [8,3]
    :param points: <np.float: 3,n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
        #     5------4
        #     |\\    |\\
        #     | \\   | \\
        #     6--\\--7  \\
        #     \\  \\  \\ \\
        # l    \\  1-------0    h
        #  e    \\ ||   \\ ||   e
        #   n    \\||    \\||   i
        #    g    \\2------3    g
        #     t      width.     h
        #      h.               t.

    # Compute normal vectors that point outwards
    p1 = corners[0]
    p_x = corners[4]
    p_y = corners[1]
    p_z = corners[3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask

# def point_in_cube(point,cube):
#         #     5------4
#         #     |\\    |\\
#         #     | \\   | \\
#         #     6--\\--7  \\
#         #     \\  \\  \\ \\
#         # l    \\  1-------0    h
#         #  e    \\ ||   \\ ||   e
#         #   n    \\||    \\||   i
#         #    g    \\2------3    g
#         #     t      width.     h
#         #      h.               t.

#     # Compute normal vectors that point outwards
#     dot_product = []
#     for i in range(3):
#         edge1 = cube[2+i] - cube[0]
#         edge2 = cube[3+i] - cube[0]
#         normal_vectors=list(np.cross(edge1,edge2))
#         dot_product.append(np.dot(normal_vectors, point - cube[0]))
#         if i!=2:
#             edge1 = cube[2+i] - cube[6]
#             edge2 = cube[1+i] - cube[6]
#             normal_vectors=list(np.cross(edge1,edge2))
#             dot_product.append(np.dot(normal_vectors, point - cube[6]))
#         else: 
#             edge1 = cube[5] - cube[6]
#             edge2 = cube[4] - cube[6]
#             normal_vectors=list(np.cross(edge1,edge2))
#             dot_product.append(np.dot(normal_vectors, point - cube[6]))

#     for result in dot_product:
#         if result > 0:
#             return False
#     return True

def setdiff2d_set(arr1, arr2):
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return np.array(list(set1 - set2))


def my_remove_close(points, x_radius: float, y_radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_front_filt = points[:, 0] < x_radius
    x_rear_filt = points[:, 0] > -x_radius * 2
    x_filt = np.logical_and(x_front_filt, x_rear_filt)
    #x_filt = np.abs(points[:, 0]) < x_radius
    y_filt = np.abs(points[:, 1]) < y_radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    #points = points[:, not_close]
    return not_close

def compute_heading(quaternion):

    x_axis_unit = (1, 0, 0)
    heading_vector = qv_mult(quaternion, x_axis_unit)
    heading_vector = heading_vector / np.linalg.norm(heading_vector)
    angle = np.degrees(np.arccos(np.clip(np.dot(x_axis_unit, heading_vector), -1.0, 1.0)))
    if angle < -180:
        angle += 360
    elif angle >=180:
        angle -=360
    # Normalize the angle following this :https://github.com/waymo-research/waymo-open-dataset/blob/om2/waymo_open_dataset/label.proto
    return  angle

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]



class DatasetTransformer:
    def __init__(
        self,
        dataset_dir: str,
        output_dir:str,
        experiment_prefix: str,
        seq_idx: int,
        use_existing_files: bool = True,
        log_id: str = None,
    ) -> None:
        """We will cache the accumulated trajectories per city, per log, and per frame
        for the tracking benchmark.
        """
        self.experiment_prefix = experiment_prefix
        self.dataset_dir = dataset_dir
        self.labels_dir = dataset_dir
        self.output_dir = output_dir
        self.seq_idx = seq_idx
        self.sdb = SynchronizationDB(self.dataset_dir)
        self.dl = SimpleArgoverseTrackingDataLoader(data_dir=dataset_dir, labels_dir=dataset_dir)
        self.am = ArgoverseMap()

        if log_id is None:
            tmp_dir = tempfile.gettempdir()
            per_city_traj_dict_fpath = f"{tmp_dir}/per_city_traj_dict_{experiment_prefix}.pkl"
            log_egopose_dict_fpath = f"{tmp_dir}/log_egopose_dict_{experiment_prefix}.pkl"
            log_timestamp_dict_fpath = f"{tmp_dir}/log_timestamp_dict_{experiment_prefix}.pkl"
            if not use_existing_files:
                # write the accumulate data dictionaries to disk
                PerFrameLabelAccumulator(dataset_dir, dataset_dir, experiment_prefix)

            self.per_city_traj_dict = load_pkl_dictionary(per_city_traj_dict_fpath)
            self.log_egopose_dict = load_pkl_dictionary(log_egopose_dict_fpath)
            self.log_timestamp_dict = load_pkl_dictionary(log_timestamp_dict_fpath)
        else:
            pfa = PerFrameLabelAccumulator(dataset_dir, dataset_dir, experiment_prefix, save=False)
            pfa.accumulate_per_log_data(log_id=log_id)
            self.per_city_traj_dict = pfa.per_city_traj_dict
            self.log_egopose_dict = pfa.log_egopose_dict
            self.log_timestamp_dict = pfa.log_timestamp_dict
    

    def get_seq_dict(self, log_id="", idx=-1,city="",pre=70,post=70):
        for city_name, trajs in self.per_city_traj_dict.items():
            if city != "":
                if city != city_name:
                    continue
            if city_name not in ["PIT", "MIA"]:
                logger.info("Unknown city")
                continue

            log_ids = []
            logger.info(f"{city_name} has {len(trajs)} tracks")

            if log_id == "":
                # first iterate over the instance axis
                for traj_idx, (traj, log_id) in enumerate(trajs):
                    log_ids += [log_id]
            else:
                log_ids = [log_id]
            # eliminate the duplicates


            for log_id in set(log_ids):
                logger.info(f"Log: {log_id}")

                ply_fpaths = sorted(glob.glob(f"{self.dataset_dir}/{log_id}/lidar/PC_*.ply"))
                img_dir = f"{self.output_dir}/image/{self.seq_idx:04d}/"
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                seq_dict = OrderedDict()
                # then iterate over the time axis
                frame_index=0
                last_timestamp = 0

                for i, ply_fpath in enumerate(ply_fpaths):
                    if idx != -1:
                        if i != idx:
                            continue
                    if i % 500 == 0:
                        logger.info(f"\tOn file {i} of {self.seq_idx:04d}")
                    lidar_timestamp = ply_fpath.split("/")[-1].split(".")[0].split("_")[-1]
                    lidar_timestamp = int(lidar_timestamp)
                    if lidar_timestamp not in self.log_egopose_dict[log_id]:
                        all_available_timestamps = sorted(self.log_egopose_dict[log_id].keys())
                        diff = (all_available_timestamps[0] - lidar_timestamp) / 1e9
                        logger.info(f"{diff:.2f} sec before first labeled sweep")
                        continue

                    # logger.info(f"\tt={lidar_timestamp}")
                    pose_city_to_ego = self.log_egopose_dict[log_id][lidar_timestamp]

                    ego_center_xyz = np.array(pose_city_to_ego["translation"]) 
                    #-------------------ego vehicle center in city coordinate---------------------

                    city_SE3_egovehicle = SE3(
                    rotation=pose_city_to_ego["rotation"],
                    translation=ego_center_xyz,
                    )

                    lidar_pts = load_ply(ply_fpath)
                    all_mask = my_remove_close(lidar_pts, x_radius=4.0, y_radius=2.0)
                    lidar_pts = lidar_pts[all_mask, :]


                    ego_pts  = copy.deepcopy(lidar_pts)
                    city_pts = city_SE3_egovehicle.transform_point_cloud(ego_pts)

                    vehicle2global = city_SE3_egovehicle.transform_matrix

                    bboxes =[]
                    bbox_labels=[]
                    bboxes_id=[]
                    point_labels_all=[]
                    bboxes_corner=[]


                    labels = self.dl.get_labels_at_lidar_timestamp(log_id, lidar_timestamp)

                    if labels is None:
                        logging.info("\tLabels missing at t=%s", lidar_timestamp)
                        continue
                    for label_idx, label in enumerate(labels):
                        obj_rec = json_label_dict_to_obj_record(label)
                        if obj_rec.occlusion == 100:
                            continue
                        
                        heading = compute_heading(obj_rec.quaternion)
                        bbox = np.array([obj_rec.translation[0],obj_rec.translation[1],obj_rec.translation[2]-obj_rec.height/2,
                                obj_rec.length,obj_rec.height,obj_rec.width,-heading])
                        bbox_id = obj_rec.track_id
                        bbox_label = label['label_class']

                        bboxes.append(bbox)
                        bboxes_id.append(bbox_id)
                        bbox_labels.append(bbox_label)
                        bbox_corner = obj_rec.as_3d_bbox()
                        bboxes_corner.append(bbox_corner)

                    #replace with number
                    # for src, tgt in label_remap.items():
                    #     point_labels_all = list(map(lambda x: tgt if x == src else x, point_labels_all))
                    
                    bboxes = np.array(bboxes)
                    bboxes_id=np.array(bboxes_id)
                    bbox_labels=np.array(bbox_labels)
                    bboxes_corner=np.array(bboxes_corner)
                    points_all = np.array(lidar_pts)
                    point_labels_all=np.ones(points_all.shape[0], dtype=np.uint8)
                    assert (len(point_labels_all) == len(points_all))

                    origin = np.array([(np.array(ego_center_xyz))])

                    points_all_city = city_SE3_egovehicle.transform_point_cloud(points_all)
                    _, not_ground_logicals = self.am.remove_ground_surface(
                            copy.deepcopy(points_all_city), city_name, return_logicals=True)
                    grd_points= points_all[np.logical_not(not_ground_logicals)]
                    grd_height=grd_points[2].mean()
                    points_all = np.hstack((points_all, np.ones((points_all.shape[0], 1))))
                    

                    seq_dict[frame_index] = {
                    'points_all': points_all.T, # adjust argo to lyft
                    'points_labels_all': point_labels_all,
                    'lidar2vehicle': np.eye(4),
                    'vehicle2global': vehicle2global,
                    'bboxes': bboxes,
                    'labels': bbox_labels,
                    'bboxes_id': bboxes_id,
                    'bboxes_corner': bboxes_corner,
                    'origin': origin,
                    'seg_labels': True,
                    'grd_height':grd_height,
                    #'box_seg_labels': box_seg_labels,
                    }
                    frame_index +=1



                    cam_intrinsics = {}
                    cam_extrinsics = {}

                    for cam_name in RING_CAMERA_LIST:
                        # if not img_paths:
                        #     img_paths[f'{cam_name}']=self.dl.get_ordered_log_cam_fpaths(log_id,cam_name)
                        cam_timestamp = self.sdb.get_closest_cam_channel_timestamp(lidar_timestamp, cam_name, log_id)

                        logger.info(f"\tt={cam_timestamp}")
                        if get_city_SE3_egovehicle_at_sensor_t(cam_timestamp, self.dataset_dir, log_id):
                            city_SE3_egovehicle = get_city_SE3_egovehicle_at_sensor_t(cam_timestamp, self.dataset_dir, log_id)
                        else:
                            raise RuntimeError(
                                f"Could not get city to egovehicle coordinate transformation at timestamp {cam_timestamp}"
                            )

                        cam_config = self.dl.get_log_camera_config(log_id,cam_name)
                        extrinsics = cam_config.extrinsic
                        intrinsics = cam_config.intrinsic
                        intrinsics = np.array([
                        [intrinsics[0][0], 0, intrinsics[0][2]],
                        [0, intrinsics[1][1], intrinsics[1][2]],
                        [0, 0,                       1]])

                        cam_intrinsics[cam_name] = intrinsics
                        cam_extrinsics[cam_name] = extrinsics

                        img_path = f"{self.dl.data_dir}/{log_id}/{cam_name}/{cam_name}_{cam_timestamp}.jpg"
                        save_file_name = "{:04d}_{}.jpg".format(i, cam_name)
                        img_save_path = os.path.join(self.output_dir, "image", f"{self.seq_idx:04d}", save_file_name)
                        shutil.copy(img_path, img_save_path)


                    if last_timestamp ==0:
                        last_timestamp = lidar_timestamp

                        save_path_calib = os.path.join(self.output_dir, "calib",f"{self.seq_idx:04d}")
                        if not os.path.exists(save_path_calib):
                            os.makedirs(save_path_calib)

                        np.savez_compressed(os.path.join(os.path.join(self.output_dir,'calib', f"{self.seq_idx:04d}"), "cam_intrinsics"), **cam_intrinsics)
                        np.savez_compressed(os.path.join(os.path.join(self.output_dir,'calib', f"{self.seq_idx:04d}"), "cam_extrinsics"), **cam_extrinsics)


                # Convert all frames in the scene.
                for k in tqdm(range(len(seq_dict))):
                    final_points, final_poses, final_labels = self.convert_one_frame(k,pre,post,copy.deepcopy(seq_dict))
                    save_path_pcd = os.path.join(self.output_dir, "point_cloud",f"{self.seq_idx:04d}")
                    if not os.path.exists(save_path_pcd):
                        os.makedirs(save_path_pcd)
                    points_path = os.path.join(self.output_dir, "point_cloud",f"{self.seq_idx:04d}", "{:04d}_point".format(k))
                    poses_path = os.path.join(self.output_dir, "point_cloud",f"{self.seq_idx:04d}", "{:04d}_pose".format(k))
                    labels_path = os.path.join(self.output_dir, "point_cloud",f"{self.seq_idx:04d}", "{:04d}_label".format(k))

                    np.savez_compressed(points_path, **final_points)
                    np.savez_compressed(poses_path, **final_poses)
                    np.savez_compressed(labels_path, **final_labels)
                    # print(f'logid:{log_id}, frame {k} saved')
                    # else:
                    #     print(f'----Already exists: logid:{log_id}, frame {k}')


    def convert_one_frame(self,cur_index, pre, post,seq_dict):
        pre = max(0, cur_index - pre)
        post = min(len(seq_dict), cur_index + post)

        final_points = {}
        final_labels = {}
        final_poses = {}


        for i in range(pre, post):
            if i != cur_index:
                for j in range(len(seq_dict[i]['bboxes_id'])):
                    if seq_dict[i]['bboxes_id'][j] not in seq_dict[cur_index]['bboxes_id']:
                        mask = points_in_box(seq_dict[i]['bboxes_corner'][j],seq_dict[i]['points_all'][:-1])
                        seq_dict[i]['points_all'] = np.delete(seq_dict[i]['points_all'],mask, axis=1)
                        seq_dict[i]['points_labels_all']=np.delete(seq_dict[i]['points_labels_all'],mask)
                    else:
                        mask = points_in_box(seq_dict[i]['bboxes_corner'][j],seq_dict[i]['points_all'][:-1])
                        if not np.any(mask): continue
                        index = np.where(seq_dict[cur_index]['bboxes_id']==seq_dict[i]['bboxes_id'][j])[0][0]
                        dst_box = np.copy(seq_dict[cur_index]['bboxes_corner'][index])
                        src_box = np.copy(seq_dict[i]['bboxes_corner'][j])
                        c_dst = np.mean(dst_box, axis=0)
                        c_src = np.mean(src_box, axis=0)
                        dst_box -= c_dst
                        src_box -= c_src
                        A = np.dot(dst_box.T, src_box)
                        U, _, Vt = np.linalg.svd(A, full_matrices=False)
                        V = Vt.T
                        R = np.dot(V, U.T)
                        c = np.expand_dims(c_dst - c_src @ R, 1)

                        curr_points = np.copy(seq_dict[i]['points_all'][:, mask])
                        curr_points[:-1] = np.add(R.T @ curr_points[:-1], c)

                        src_pose = seq_dict[i]['vehicle2global']
                        dst_pose = seq_dict[cur_index]['vehicle2global']
                        new_points = np.linalg.inv(src_pose) @ dst_pose @ curr_points
                        seq_dict[i]['points_all'][:, mask] = new_points
                        seq_dict[i]['points_labels_all'][mask] = label_remap[seq_dict[i]['labels'][j]]


            final_points[f"{i}_LIDAR_TOP"] = seq_dict[i]['points_all']
            final_labels[f"{i}_LIDAR_TOP"] = seq_dict[i]['points_labels_all']
            final_poses[f"{i}_LIDAR_TOP"] = seq_dict[i]['vehicle2global']
        
        return final_points, final_poses, final_labels

def process_one_log(log_id, seq_idx,args):

    df = DatasetTransformer(
        args.dataset_dir,
        args.output_path,
        args.experiment_prefix,
        seq_idx,
        log_id=log_id,
        use_existing_files=args.use_existing_files,
    )
    df.get_seq_dict(pre=args.a_pre,post=args.a_post)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,default='/home/yiming/guofeng/openssc/dataset/Argo/train4', help="path to where the logs live")
    parser.add_argument('-o', '--output_path', type=str, default='/home/yiming/guofeng/openssc/argoverse-api/transform/output/train4', help='path to the output')
    parser.add_argument(
        "--experiment_prefix",
        default="argoverse_bev_viz",
        type=str,
        help="results will be saved in a folder with this prefix for its name",
    )

    parser.add_argument(
        "--use_existing_files",
        action="store_true",
        help="load pre-saved log data from pkl dictionaries instead of scraping it",
    )
    parser.add_argument('--a_pre', type=int, default=70, help='number of frames to aggregrate before the current frame')
    parser.add_argument('--a_post', type=int, default=70, help='number of frames to aggregrate after the current frame')
    parser.add_argument('-i', '--idx', type=int, default=0, help='index of in batch job')

    args = parser.parse_args()
    logger.info(args)
    folders = glob.glob(args.dataset_dir+ '/*/')
    log_ids = [folder.split('/')[-2] for folder in folders]
    print(log_ids)
    process_one_log(log_ids[args.idx], args.idx, args)
