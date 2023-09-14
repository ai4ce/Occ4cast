import os
import yaml
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class ArgoverseDataset(Dataset):
    """Occ4D-Argoverse dataset class."""
    def __init__(self, root_dir, split='train', p_pre=10, p_post=10, semantic=False):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): 'train' or 'test' split.
            p_pre (int): Number of frames before the current frame.
            p_post (int): Number of frames after the current frame.
        """
        if split == 'train':
            self.root_dir = os.path.join(root_dir,'trainval')
        elif split == 'valid':
            self.root_dir = os.path.join(root_dir,'trainval')
        else:
            self.root_dir = os.path.join(root_dir,'test')

        if split == 'train':
            self.split = [0,1,2,3,4,5,6,8,10,12,13,15,16,17,18,19,20,22,23,25,26,27,28,29,30,31,32,35,37,39,41,42,43,44,45,46,47,48,49,50]
        elif split == 'valid':
            self.split = [51,52,53,54,55,56,57,58,59,60,61,62,63,64]
        elif split == 'test':
            self.split = [0,1,2,3,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        else:
            raise ValueError('Invalid split name: {}'.format(split))
        
        self.p_pre = p_pre + 1 # +1 for the current frame
        self.p_post = p_post + 1
        self.semantic = semantic
        
        with open("configs/argoverse.yaml", 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self._load_data()

    def _load_data(self):
        self.data_path = []
        for i in self.split:
            sample_path = glob(os.path.join(self.root_dir, 'voxel', '{:04d}'.format(i), '*.npz'))
            self.data_path.extend(sample_path)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        with open(self.data_path[idx], 'rb') as f:
            data = np.load(f)
            input = data['input'][-self.p_pre:]
            label = data['label'][:self.p_post]
            invalid = data['invalid'][:self.p_post]
        input_tensor = torch.tensor(input, dtype=torch.float32)
        label_tensor = torch.tensor(label)
        invalid_tensor = torch.from_numpy(invalid)
        return input_tensor, label_tensor, invalid_tensor

    def get_voxel_size(self):
        temp_data = np.load(self.data_path[0])['input']
        X, Y, Z = temp_data.shape[1:]
        T = self.p_post
        return X, Y, Z, T

    def get_class_fc(self):
        if self.semantic:
            return self.config['label_fc']
        else:
            fc_dict = self.config['label_fc']
            for i in range(2, len(fc_dict)):
                fc_dict[1] += fc_dict[i]
                del fc_dict[i]
            return fc_dict