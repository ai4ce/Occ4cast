from dataset import ApolloScape
from geometry_utils import points_in_box
from tqdm import tqdm
import numpy as np
import argparse
import os
from geometry_utils import transform_points_from_src_to_dest, get_homo_pose
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, default='/media/rua/44BCDF28BCDF12F21/ApolloScape/tracking', help='path to the dataset')
parser.add_argument('-b', '--batch', type=int, default=0, help='batch job index')
parser.add_argument('-s', '--start', type=int, default=0, help='start index of the scene')
parser.add_argument('-o', '--output_path', type=str, default='./ApolloScape', help='path to the output')
parser.add_argument('--a_pre', type=int, default=70, help='number of frames to aggregrate before the current frame')
parser.add_argument('--a_post', type=int, default=70, help='number of frames to aggregrate after the current frame')
parser.add_argument('--p_pre', type=int, default=10, help='number of frames to input before the current frame')
parser.add_argument('--p_post', type=int, default=10, help='number of frames to input after the current frame')
parser.add_argument('--bbox_diff_threshold', type=float, default=0.8, help='threshold of bbox size difference')
args = parser.parse_args()

dataset = ApolloScape(args.data_path)

config = {
    "aggregate_pre": args.a_pre,
    "aggregate_post": args.a_post,
    "prediction_pre": args.p_pre,
    "prediction_post": args.p_post,
    "output_path": args.output_path
}


def convert_one_scene(id, trip_id, segment_id, absoulte_scene_index):
    save_path_pcd = os.path.join(args.output_path, "point_cloud", "{:04d}".format(absoulte_scene_index))
    os.makedirs(save_path_pcd, exist_ok=True)
    
    for frame_idx in tqdm(range(len(id[trip_id][segment_id]))):
        # if frame_idx % 10 == 0:

        points_path = os.path.join(save_path_pcd, "{:04d}_point".format(frame_idx))
        poses_path = os.path.join(save_path_pcd, "{:04d}_pose".format(frame_idx))
        labels_path = os.path.join(save_path_pcd, "{:04d}_label".format(frame_idx))
            
        return_points, return_labels, return_poses = convert_one_frame(id, trip_id, segment_id, frame_idx)
        
        np.savez_compressed(points_path, **return_points)
        np.savez_compressed(labels_path, **return_labels)
        np.savez_compressed(poses_path, **return_poses)
        # return
        

def convert_one_frame(id, trip_id, segment_id, frame_idx):
    pre = max(0, frame_idx - args.a_pre)
    post = min(len(id[trip_id][segment_id]), frame_idx + args.a_post)

    return_points = {}
    return_labels = {}
    return_poses = {}
    
    target_data = dataset.get_data(trip_id, segment_id, id[trip_id][segment_id][frame_idx])
    target_points = target_data['lidar']
    target_points[:, 3] = 1
    target_pose_data = target_data['pose']['pose_data']
    target_pose_matrix = get_homo_pose(target_pose_data)

    target_label = target_data['label']
    
    # separate point labels into object removed points and object points
    # NOTE: 0 as label is FREE, 1 as label is UNLABELED
    target_point_label = np.ones(len(target_points), dtype=np.uint8)
    for target_instance in target_label:
        mask = points_in_box(target_instance['bbox'], target_points[:, :3], scale_factor=1.0)
        target_point_label[mask] = target_instance['object_type']

    for i in tqdm(range(pre, post), leave=False, disable=True):
        if i != frame_idx:
            data = dataset.get_data(trip_id, segment_id, id[trip_id][segment_id][i])
            points = data['lidar']
            points[:, 3] = 1
            pose_data = data['pose']['pose_data']
            label = data['label']
            points_label = np.ones(len(points), dtype=np.uint8)
            pose_matrix = get_homo_pose(pose_data)
  
            src_object_ids = [instance['object_id'] for instance in label]
            # src_object_types = [instance['object_type'] for instance in label]
            src_bboxes = [instance['bbox'] for instance in label] 
            target_object_ids = [instance['object_id'] for instance in target_label]
            target_object_types = [instance['object_type'] for instance in target_label]
            target_bboxes = [instance['bbox'] for instance in target_label]

            # match bboxes with the same object id
            matched_bboxes = [[],[]]
            matched_obj_id = []
            matched_types = []
            for object_id, src_bbox in zip(src_object_ids, src_bboxes):
                target_bbox = [bbox for bbox, id in zip(target_bboxes, target_object_ids) if id == object_id]
                
                if len(target_bbox) > 0:
                    # kick out the bbox with size difference larger than threshold
                    src_l, src_w, src_h = src_bbox[3:6]
                    dest_l, dest_w, dest_h = target_bbox[0][3:6]
                    size_diff_threshold = args.bbox_diff_threshold
                    size_match = (abs(src_l - dest_l) / dest_l < size_diff_threshold) and (abs(src_w - dest_w) / dest_w < size_diff_threshold) and (abs(src_h - dest_h) / dest_h < size_diff_threshold)
                    
                    if size_match:
                        matched_bboxes[0].append(src_bbox)
                        matched_bboxes[1].append(target_bbox[0])
                        matched_obj_id.append(object_id)
                        matched_types.append(target_object_types[target_object_ids.index(object_id)])

            # Delete the points in the unmatched bboxes
            for object_id, src_bbox in zip(src_object_ids, src_bboxes):
                if object_id not in matched_obj_id:
                    mask = points_in_box(src_bbox, points[:, :3], scale_factor=1.0)
                    points = points[~mask]
                    points_label = points_label[~mask]
                   
            src_bbox = np.array(matched_bboxes[0])
            target_bbox = np.array(matched_bboxes[1])            
            
            # transform the points in the matched bboxes
            for j in range(len(src_bbox)):
                # for each box, get points to be transformed inside the box
                mask = points_in_box(src_bbox[j], points[:, :3].copy(), scale_factor=1.0)
        
                obj_points = points[mask].copy()
           
                transformed_obj_points = transform_points_from_src_to_dest(src_bbox[j], target_bbox[j], obj_points)
                
                transformed_homo_points = np.hstack([transformed_obj_points[:,:3], np.ones((transformed_obj_points.shape[0],1))])
                
                # Apply revert pose to prepare for further aggregation in ray-tracing
                transformed_obj_points = np.linalg.inv(pose_matrix) @ target_pose_matrix @ transformed_homo_points.T

                points[mask, :3] = transformed_obj_points.T[:,:3]
                points_label[mask] = matched_types[j]

            return_points["{}_LIDAR_TOP".format(i)] = points.T
            return_labels["{}_LIDAR_TOP".format(i)] = points_label
            return_poses["{}_LIDAR_TOP".format(i)] = pose_matrix

    return_points["{}_LIDAR_TOP".format(frame_idx)] = target_points.T
    return_labels["{}_LIDAR_TOP".format(frame_idx)] = target_point_label
    return_poses["{}_LIDAR_TOP".format(frame_idx)] = target_pose_matrix

    return return_points, return_labels, return_poses


if __name__ == "__main__":
    print('Start converting dataset, scene: {}'.format(args.start+args.batch))
    id = dataset.id
    absoulte_scene_index = 0
    
    # create a mapping from scene index to trip id and segment id
    id_mapping = {}
    for trip_id in dataset.id:
        segment_ids = dataset.id[trip_id].keys()
        for segment_id in segment_ids:
            id_mapping[absoulte_scene_index] = (trip_id, segment_id)
            absoulte_scene_index += 1

    i = args.start + args.batch
    trip_id, segment_id = id_mapping[i]
    convert_one_scene(id, trip_id, segment_id, i)
