import os
import yaml
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class LyftDataset(Dataset):
    """Occ4D-Lyft dataset class."""
    def __init__(self, root_dir, split='train', p_pre=10, p_post=10, semantic=False):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): 'train' or 'test' split.
            p_pre (int): Number of frames before the current frame.
            p_post (int): Number of frames after the current frame.
        """
        self.root_dir = root_dir
        
        if split == 'train':
            self.split = list(range(120))
        elif split == 'valid':
            self.split = list(range(120, 150))
        elif split == 'test':
            self.split = list(range(150, 180))
        else:
            raise ValueError('Invalid split name: {}'.format(split))
        
        self.p_pre = p_pre + 1 # +1 for the current frame
        self.p_post = p_post + 1
        self.semantic = semantic
        
        with open("configs/lyft.yaml", 'r') as f:
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
        data = np.load(self.data_path[idx])
        input = data['input'][-self.p_pre:]
        label = data['label'][:self.p_post]
        invalid = data['invalid'][:self.p_post]
        input_tensor = torch.tensor(input, dtype=torch.float16)
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