import torch
from torch.utils.data import Dataset
import numpy as np

import glob

def seq_collate(data):
    # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

    (pre_motion_3D, fut_motion_3D,pre_motion_mask,fut_motion_mask, route_lanes, map_lanes, map_crosswalks)  = zip(*data)

    pre_motion_3D = torch.stack(pre_motion_3D,dim=0)
    fut_motion_3D = torch.stack(fut_motion_3D,dim=0)
    fut_motion_mask = torch.stack(fut_motion_mask,dim=0)
    pre_motion_mask = torch.stack(pre_motion_mask,dim=0)
    route_lanes = torch.stack(route_lanes,dim=0)
    map_lanes = torch.stack(map_lanes,dim=0)
    map_crosswalks = torch.stack(map_crosswalks,dim=0)



    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'fut_motion_mask': fut_motion_mask,
        'pre_motion_mask': pre_motion_mask,
        'route_lanes': route_lanes,
        'map_lanes': map_lanes,
        'map_crosswalks': map_crosswalks,
        'traj_scale': 1,
        'pred_mask': None,
        'seq': 'nuplan',
    }
   

    return data

class nuplanDB(Dataset):
    def __init__(self, obs_len=10, pred_len=40, training=True):
        super(nuplanDB, self).__init__()
        self.obs_len = obs_len  #NOTES: LED ref to train_led_trajectory_augment_input.py line 33, gameformer/data_process line 18-21
        self.pred_len = pred_len 
        self._lane_len = 50
        self._lane_freature = 7
        # self.seq_len = self.obs_len + self.pred_len
        self.n_neighbors = 10
        self.obs_dim = 6
        self.pred_dim = 3
        
        if training:
            data_dir = '/home/arc/nuplan/processed_data_5fps/train/*.npz'
        else:
            data_dir = '/home/arc/nuplan/processed_data_5fps/valid/*.npz'

        self.data_list = glob.glob(data_dir)
        self.batch_len = len(self.data_list)

        print('Loading nuPlan dataset for ' + 'training' if training else 'validation')
        print('Batch length: ', self.batch_len)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego_past = data['ego_agent_past'][:self.obs_len, :self.obs_dim] #dim:(11, 7)  #NOTES: please refer to /home/arc/GameFormer-Planner/GameFormer/train_utils.py line 34 
        ego_future = data['ego_agent_future'][:self.pred_len, :self.pred_dim] #dim: (40, 3)
        neighbors_past = data['neighbor_agents_past'][:self.n_neighbors, :self.obs_len, :self.obs_dim] #dim: (20, 11, 11)
        neighbors_future = data['neighbor_agents_future'][:self.n_neighbors, :self.pred_len, :self.pred_dim] #dim: (20, 40, 3)
        route_lanes = data['route_lanes'] #dim: (10, 50, 3)
        map_lanes = data['lanes'] #dim: (40, 50, 7)
        map_crosswalks = data['crosswalks'] #dim: (5, 30, 3)
        

        
        past_traj = np.zeros((self.obs_len, 11, self.obs_dim)) #dim: (30, 11, 6)
        past_traj[:, 0, :] = ego_past  # Ego takes the first spot
        
        future_traj = np.zeros((self.pred_len, 11, self.pred_dim)) #dim: (30, 11, 3)
        future_traj[:, 0, :] = ego_future # Ego takes the first spot
        
        for i in range(self.n_neighbors):
            past_traj[:, i + 1, :] = neighbors_past[i]  # Neighbor takes the subsequent spots
            future_traj[:, i + 1, :] = neighbors_future[i]  # Neighbor takes the subsequent spots

       
        route_lanes = torch.from_numpy(route_lanes).type(torch.float) #dim: torch.Size([10, 50, 3])
        map_lanes = torch.from_numpy(map_lanes).type(torch.float) #dim: torch.Size([40, 50, 7])
        map_crosswalks = torch.from_numpy(map_crosswalks).type(torch.float) #dim: torch.Size([5, 30, 3])
        
        
        pre_motion_3D = torch.from_numpy(past_traj).type(torch.float).permute(1, 0, 2) #dim: torch.Size([11, 10, 6])
        fut_motion_3D = torch.from_numpy(future_traj).type(torch.float).permute(1, 0, 2) #dim: torch.Size([11, 20, 3])
        pre_motion_mask = torch.ones(11, self.obs_len) #dim: torch.Size([11, 10])
        fut_motion_mask = torch.ones(11, self.pred_len) #dim: torch.Size([11, 20])

        out = [pre_motion_3D, fut_motion_3D, pre_motion_mask, fut_motion_mask, route_lanes, map_lanes, map_crosswalks]
        return out
