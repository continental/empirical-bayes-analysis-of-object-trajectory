'''
Copyright (c) 2021-2022 Continental AG.

@author: Yue Yao
'''
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
import os

def generate_file_list_dataset(path_list, outlier_list):
    '''
    Generate a tf dataloader for traj files without outlier
    '''
    
    path_list_without_outlier = []

    for root, dirs, files in os.walk(os.path.abspath(path_list)):
        files.sort()
        for idx, file in enumerate(tqdm(files)):
            if idx in outlier_list:
                continue
            else:
                path_list_without_outlier.append(os.path.join(root, file))

    file_list_dataset = tf.data.Dataset.from_tensor_slices(path_list_without_outlier)
    
    return file_list_dataset


def generate_start_indicies_dataset(start_indicies_file, outlier_list):
    '''
    Generate a tf dataloader for start indicies without outlier
    '''
    start_indicies_without_outlier = []

    with open(start_indicies_file, "r") as read_file:
        start_indicies_all = np.array((json.load(read_file)))
    
    for idx, start_index in enumerate(tqdm(start_indicies_all)):
        if idx in outlier_list:
            continue
        else:
            start_indicies_without_outlier.append(start_index)

    start_indicies_dataset = tf.data.Dataset.from_tensor_slices(start_indicies_without_outlier)
    
    return start_indicies_dataset


def generate_selected_file_list_dataset(path_list, idx_trajs_select):
    path_list_select = []
    for root, dirs, files in os.walk(os.path.abspath(path_list)):
        files.sort()
    
    files = np.array(files)
    path_list_select_array = files[idx_trajs_select]
    
    for idx, file in enumerate(tqdm(path_list_select_array)):
        path_list_select.append(os.path.join(root, path_list_select_array[idx]))

    file_list_dataset = tf.data.Dataset.from_tensor_slices(path_list_select)
    
    return file_list_dataset


def generate_selected_start_indicies_dataset(start_indicies_file):
    with open(start_indicies_file, "r") as read_file:
        start_indicies_all = np.array((json.load(read_file)))

    start_indicies_dataset = tf.data.Dataset.from_tensor_slices(start_indicies_all)
    
    return start_indicies_dataset


class DataProcessor(object):
    '''
    Dataloader with load the trajectory json file and output training data.
    '''
    def __init__(self, batch_size, dataset, num_points_in_one_traj, traj_type, with_heading = False):
        self.batch_size = batch_size
        self.dataset = dataset  
        self.num_points_in_one_traj = num_points_in_one_traj
        self.loaded_dataset = None # the processed dataloader
        self.traj_type = traj_type # 'ego_traj' or 'agt_traj'
        self.with_heading = with_heading # boolean
    
    def _extract_ego_trajs(self, file_path, start_idx):
        file_str = str(file_path.numpy())[2:-1]
        ego_trajs_all = []
        times_all = []
        with open(file_str, "r") as read_file:
            traj_data = json.load(read_file)
            
        ego_traj_temp = np.array(traj_data['ego_traj'])[start_idx : start_idx+self.num_points_in_one_traj]
        heading_vector = np.array([np.cos(ego_traj_temp[:, 3]), np.sin(ego_traj_temp[:, 3])]).T
        

        ego_traj_temp = ego_traj_temp[:, :2] - ego_traj_temp[0,:2] # let trajectories start from (0,0)
        ego_traj = np.concatenate((ego_traj_temp[:, 0], ego_traj_temp[:, 1]), axis = 0)
        
        times = np.array(traj_data['timestamp'])[start_idx: start_idx+self.num_points_in_one_traj]
        times = times - times[0]
        
        if self.with_heading:
            return tf.convert_to_tensor(times, dtype=tf.float32), tf.convert_to_tensor(ego_traj, dtype=tf.float32), heading_vector
        else:
            return tf.convert_to_tensor(times, dtype=tf.float32), tf.convert_to_tensor(ego_traj, dtype=tf.float32)
    
    def _extract_agt_trajs(self, file_path, start_idx):
        file_str = str(file_path.numpy())[2:-1]
        ego_trajs_all = []
        times_all = []
        with open(file_str, "r") as read_file:
            traj_data = json.load(read_file)
            
        ego_traj_temp = np.array(traj_data['ego_traj'])[start_idx : start_idx+self.num_points_in_one_traj]
        agt_traj_temp = np.array(traj_data['agt_traj'])[start_idx : start_idx+self.num_points_in_one_traj]
        
        heading_vector = np.array([np.cos(agt_traj_temp[:, 3]), np.sin(agt_traj_temp[:, 3])]).T
        
        d = agt_traj_temp[:, [0,1]] - ego_traj_temp[:, [0,1]] 
        
        d_norm = np.linalg.norm(d, axis=1)
        theta = np.arctan2(d[:,1], d[:,0]) - ego_traj_temp[:, 3]

        c_ego_to_map, s_ego_to_map = np.cos(ego_traj_temp[:, 3]), np.sin(ego_traj_temp[:, 3])
        R_ego_to_map = np.transpose(np.array(((c_ego_to_map, -s_ego_to_map), (s_ego_to_map, c_ego_to_map))), (2, 0, 1))
        
        agt_traj_temp = agt_traj_temp[:, :2] - agt_traj_temp[0,:2] # let agt trajectories start from zero
        agt_traj = np.concatenate((agt_traj_temp[:, 0], agt_traj_temp[:, 1]), axis = 0)     
        
        times = np.array(traj_data['timestamp'])[start_idx: start_idx+self.num_points_in_one_traj]
        times = times - times[0]
        
        times = tf.convert_to_tensor(times, dtype=tf.float32)
        agt_traj = tf.convert_to_tensor(agt_traj, dtype=tf.float32)
        d_norm = tf.convert_to_tensor(d_norm, dtype=tf.float32)
        heading_vector = tf.convert_to_tensor(heading_vector, dtype=tf.float32)
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        R_ego_to_map = tf.convert_to_tensor(R_ego_to_map, dtype=tf.float32)
        
        if self.with_heading:
            return times, agt_traj, heading_vector, d_norm, theta, R_ego_to_map
        else:
            return times, agt_traj, d_norm, theta, R_ego_to_map

    
    def _load_data(self, file_path, start_idx):
        if self.traj_type == 'ego_traj':
            if self.with_heading:
                return tf.py_function(self._extract_ego_trajs, [file_path, start_idx], [tf.float32, tf.float32, tf.float32])
            else:
                return tf.py_function(self._extract_ego_trajs, [file_path, start_idx], [tf.float32, tf.float32])
        elif self.traj_type == 'agt_traj':
            if self.with_heading:
                return tf.py_function(self._extract_agt_trajs, [file_path, start_idx], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
            else:
                return tf.py_function(self._extract_agt_trajs, [file_path, start_idx], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        else:
            raise ValueError('Unknown trajectory type.')
            
        
    def load_process(self, shuffle = False):
        self.loaded_dataset = self.dataset.map(map_func = self._load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        if shuffle:
            self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=self.loaded_dataset.__len__())
        
        # Set batch size for dataset
        self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        
    def get_batch(self):
        return next(iter(self.loaded_dataset))