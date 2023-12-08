import torch
from torch.utils.data import Dataset
import numpy as np

import glob

class NBADataset_FAKE(Dataset):
    def __init__(self, obs_len=5, pred_len=10, training=True):
        super(NBADataset_FAKE, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_neighbors = 10
        
        if training:
            data_dir = '/home/arc/nuplan/processed_data/train/*.npz'
        else:
            data_dir = '/home/arc/nuplan/processed_data/valid/*.npz'

        self.data_list = glob.glob(data_dir)
        self.batch_len = len(self.data_list)

        print('Loading FAKE NBA dataset for ' + 'training' if training else 'validation')
        print('Batch length: ', self.batch_len)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego_past = data['ego_agent_past'][:self.obs_len, :2]
        ego_future = data['ego_agent_future'][:self.pred_len, :2]
        neighbors_past = data['neighbor_agents_past'][:self.n_neighbors, :self.obs_len, :2]
        neighbors_future = data['neighbor_agents_future'][:self.n_neighbors, :self.pred_len, :2]

        full_traj_ego = np.concatenate((ego_past, ego_future), axis=0)
        full_traj = np.zeros((self.seq_len, 11, 2))

        full_traj[:, 0, :] = full_traj_ego  # Ego takes the first spot

        for i in range(self.n_neighbors):
            full_traj_neighbor = np.concatenate((neighbors_past[i], neighbors_future[i]), axis=0)
            full_traj[:, i + 1, :] = full_traj_neighbor  # Neighbor takes the subsequent spots

        traj_abs = torch.from_numpy(full_traj).type(torch.float).permute(1, 0, 2)
        
        pre_motion_3D = traj_abs[:, :self.obs_len, :]
        fut_motion_3D = traj_abs[:, self.obs_len:, :]
        pre_motion_mask = torch.ones(11, self.obs_len)
        fut_motion_mask = torch.ones(11, self.pred_len)

        out = [pre_motion_3D, fut_motion_3D, pre_motion_mask, fut_motion_mask]
        return out
