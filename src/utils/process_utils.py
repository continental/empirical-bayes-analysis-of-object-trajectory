'''
Copyright (c) 2021-2022 Continental AG.

@author: Yue Yao
'''
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import numpy as np
import json
import os

import utils.utils as utils

# function to unpack the recorded data
def process_raw_data(filename, object_type_id, ego_save_path, agt_save_path, save_ego = True, save_agt = True): 
    i_ego, i_agt = 0, 0

    dataset = tf.data.TFRecordDataset(filename)
    
    for data in dataset:
        ego_sub_data = {'filename': os.path.split(filename)[-1], 'scenario_id': None, 'timestamp': None, 'ego_traj': None}
        
        proto_string = data.numpy()
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)        

        ego_traj = np.array([[state.center_x, state.center_y,state.center_z, state.heading, state.velocity_x, state.velocity_y]for state in proto.tracks[proto.sdc_track_index].states]).astype(np.float32)
        
        ego_sub_data['scenario_id'] = proto.scenario_id
        ego_sub_data['timestamp'] = np.array(proto.timestamps_seconds).astype(np.float32)
        ego_sub_data['ego_traj'] = ego_traj

        if save_ego:
            with open(ego_save_path+ os.path.split(filename)[-1]+ '_' + str(i_ego).zfill(5)+ '.json', "w") as write_file:
                json.dump(ego_sub_data, write_file, cls=utils.NumpyEncoder)
        
        i_ego = i_ego + 1
        
        for track in proto.tracks_to_predict:
            if proto.tracks[track.track_index].object_type == object_type_id and track.track_index != proto.sdc_track_index: # agt should not be ego
                agt_sub_data = {'filename': os.path.split(filename)[-1], 'scenario_id': proto.scenario_id, 'timestamp': None, 'ego_traj': ego_traj, 'agt_traj': None}
                agt_traj = np.array([[state.center_x, state.center_y,state.center_z, state.heading, state.velocity_x, state.velocity_y] for state in proto.tracks[track.track_index].states]).astype(np.float32)
               
                agt_sub_data['timestamp'] = np.array(proto.timestamps_seconds).astype(np.float32)
                agt_sub_data['agt_traj'] = agt_traj
                
                if save_agt:
                    with open(agt_save_path + os.path.split(filename)[-1]+ '_' + str(i_ego).zfill(5) + '_' + str(i_agt).zfill(5)+ '.json', "w") as write_file:
                        json.dump(agt_sub_data, write_file, cls=utils.NumpyEncoder)
                        
                i_agt = i_agt+1
            
    return  i_ego, i_agt


def get_smoothed_trajs_wo_outlier(trajs_smooth, idx_invalid):
    idx_valid = list(set(range(trajs_smooth.shape[0])) - set(idx_invalid))
    trajs_smooth = trajs_smooth[idx_valid]
            
    return trajs_smooth