import glob
import os
import numpy as np
import re

class ApolloScape:
    def __init__(self, base_path):
        self.base_path = base_path
        self.id = self._organize_id()
        # self.pcd_pattern = os.path.join(self.base_path, f'tracking_train_pcd', f'result_{trip_id:04d}_{segment_id}_frame', f'{frame_id:03d}.bin')
        # self.label_pattern = os.path.join(self.base_path, f'tracking_train_label', f'{trip_id:04d}_{segment_id}', f'{frame_id:03d}.txt')
        # self.pose_pattern = os.path.join(self.base_path, f'tracking_train_pose', f'result_{trip_id:04d}_{segment_id}_frame', f'{frame_id:03d}_pose.txt')


    def get_data(self, trip_id, segment_id, frame_id):
        data = {}
        # Define file patterns
        pcd_path = os.path.join(self.base_path, f'tracking_train_pcd', f'result_{trip_id:04d}_{segment_id}_frame', f'{frame_id}.bin')
        label_path = os.path.join(self.base_path, f'tracking_train_label', f'{trip_id:04d}_{segment_id}', f'{frame_id}.txt')
        pose_path = os.path.join(self.base_path, f'tracking_train_pose', f'result_{trip_id:04d}_{segment_id}_frame', f'{frame_id}_pose.txt')

        data['lidar'] = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
        pose = np.loadtxt(pose_path, dtype=np.float64, delimiter=' ')
        data['pose'] = {'frame_index': pose[0],
                        'lidar_time': pose[1],
                        'pose_data': pose[2:]}
                        # position_(x, y, z), quaternion_(x, y, z ,w)
        label = np.loadtxt(label_path, dtype=np.float32, delimiter=' ')
        if label.ndim == 1:
            label = label.reshape(1, -1)
        data['label'] = {'object_id': label[:, 0].astype(int),
                         'object_type': label[:, 1].astype(int),
                         'bbox': label[:, 2:]}
                        #  'position_x': label[:, 2],
                        #  'position_y': label[:, 3],
                        #  'position_z': label[:, 4],
                        #  'object_length': label[:, 5],
                        #  'object_width': label[:, 6],
                        #  'object_height': label[:, 7],
                        #  'heading': label[:, 8]
        labels = data['label']
        object_ids = labels['object_id']
        object_types = labels['object_type']
        bboxes = labels['bbox']

        objects_list = []
        for i in range(len(object_ids)):
            obj_dict = {
                'object_id': object_ids[i],
                # NOTE: object_type in original ApolloScape label file is 1, 2, 3, 4, 5, 6, 7, 8, 9; 
                # We need to add 1 to make it 2, 3, 4, 5, 6, 7, 8, leave 1 for unlabeled, 0 for free
                'object_type': object_types[i] + 1,
                'bbox': bboxes[i]
            }
            objects_list.append(obj_dict)

        data['label'] = objects_list
        return data

    def _get_all_ids(self):
        trip_ids = sorted([int(os.path.basename(path)[:4]) for path in glob.glob(os.path.join(self.base_path, 'tracking_train_pcd', 'result_*'))])
        segment_ids = sorted([int(os.path.basename(path)[5]) for path in glob.glob(os.path.join(self.base_path, 'tracking_train_label', '*'))])
        frame_ids = sorted([int(os.path.basename(path)[:3]) for path in glob.glob(os.path.join(self.base_path, 'tracking_train_pcd', 'result_*', '*.bin'))])
        return (trip_ids, segment_ids, frame_ids)

    def _organize_id(self):
        id = {}
        pcd_dirs = glob.glob(os.path.join(self.base_path, 'tracking_train_pcd', 'result_*_*_frame'))
        # pcd_dirs has like 9055_1_frame also 9055_10_frame, but normal sort will place 9055_10 before 9055_1, so we need to sort first number like 9055, then second number like 1 or 10
        pcd_dirs = sorted(pcd_dirs, key=lambda x: (int(x[-19:].split('_')[1]), int(x[-19:].split('_')[2])))
        for pcd_dir in pcd_dirs:
            for bin_path in glob.glob(os.path.join(pcd_dir, '*.bin')):
                # print('pcd_path', bin_path)
                trip_id, segment_id, frame_id = map(int, re.findall(r'\d+', bin_path[-30:]))
                if trip_id not in id:
                    id[trip_id] = {}
                if segment_id not in id[trip_id]:
                    id[trip_id][segment_id] = []
                id[trip_id][segment_id].append(frame_id)

        # sort frame_id
        for trip_id, segments in id.items():
            for segment_id, frames in segments.items():
                id[trip_id][segment_id] = sorted(frames)

        return id




if __name__ == "__main__":
    dataset = ApolloScape("/media/rua/44BCDF28BCDF12F21/ApolloScape/tracking")
    # trip_id = 9048
    # segment_id = 1
    # frame_id = 233
    id = dataset.id
    for trip_id in id:
        for segment_id in id[trip_id]:
            for frame_idx in range(len(id[trip_id][segment_id])):
                data = dataset.get_data(trip_id, segment_id, id[trip_id][segment_id][frame_idx])
                # print(trip_id, segment_id, id[trip_id][segment_id][frame_idx])
                lidar = data['lidar']
                pose = data['pose']
                label = data['label']

    # data = dataset.get_data(trip_id, segment_id, frame_id)

    # print(data)
    