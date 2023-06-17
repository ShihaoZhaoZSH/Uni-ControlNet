import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset

from .util import *


class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        file_ids, self.annos = read_anno(anno_path)
        self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in file_ids]
        self.local_paths = {}
        for local_type in local_type_list:
            self.local_paths[local_type] = [os.path.join(condition_root, local_type, file_id + '.jpg') for file_id in file_ids]
        self.global_paths = {}
        for global_type in global_type_list:
            self.global_paths[global_type] = [os.path.join(condition_root, global_type, file_id + '.npy') for file_id in file_ids]
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = (image.astype(np.float32) / 127.5) - 1.0

        anno = self.annos[index]
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(self.local_paths[local_type][index])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])

        local_conditions = []
        for local_file in local_files:
            condition = cv2.imread(local_file)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0
            local_conditions.append(condition)
        global_conditions = []
        for global_file in global_files:
            condition = np.load(global_file)
            global_conditions.append(condition)

        if random.random() < self.drop_txt_prob:
            anno = ''
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
        
    def __len__(self):
        return len(self.annos)
        