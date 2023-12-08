import os, random, numpy as np, copy

from utils.utils import print_log
from torch.utils.data import Dataset
import torch


def seq_collate(data):
    # batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum = inputs

    (pre_motion_3D, fut_motion_3D,pre_motion_mask,fut_motion_mask) = zip(*data)

    pre_motion_3D = torch.stack(pre_motion_3D,dim=0)
    fut_motion_3D = torch.stack(fut_motion_3D,dim=0)
    fut_motion_mask = torch.stack(fut_motion_mask,dim=0)
    pre_motion_mask = torch.stack(pre_motion_mask,dim=0)

    # print(pre_motion_3D.shape)
    # print(fut_motion_3D.shape)
    # print(fut_motion_mask.shape)
    # print(pre_motion_mask.shape)
    # time.sleep(1000)
    # batch_abs = torch.cat(batch_abs_list,dim=0).permute(1,0,2)
    # # print(batch_abs.shape)
    # # .permute(1,0,2,3)
    # # batch_abs = batch_abs.view(batch_abs.shape[0],batch_abs.shape[1]*batch_abs.shape[2],batch_abs.shape[3])
    # batch_norm = torch.cat(batch_norm_list,dim=0).permute(1,0,2)
    # # batch_norm = batch_abs.view(batch_norm.shape[0],batch_norm.shape[1]*batch_norm.shape[2],batch_norm.shape[3])
    # shift_value = torch.cat(shift_value_list,dim=0).permute(1,0,2)
    # # shift_value = shift_value.view(shift_value.shape[0],shift_value.shape[1]*shift_value.shape[2],shift_value.shape[3])
    # seq_list = torch.ones(batch_abs.shape[0],batch_abs.shape[1])
    # batch_size = int(batch_abs.shape[1] / 11)
    # nei_list = torch.from_numpy(np.kron(np.diag([1]*batch_size),np.ones((11,11),dtype='float32'))-np.eye(batch_size*11)).repeat(batch_abs.shape[0],1,1)
    # nei_num = torch.ones(batch_abs.shape[0],batch_abs.shape[1]) * 10
    # batch_pednum = torch.from_numpy(np.array([11]*batch_size))

    data = {
        'pre_motion_3D': pre_motion_3D,
        'fut_motion_3D': fut_motion_3D,
        'fut_motion_mask': fut_motion_mask,
        'pre_motion_mask': pre_motion_mask,
        'traj_scale': 1,
        'pred_mask': None,
        'seq': 'nba',
    }
    # out = [
    #     batch_abs, batch_norm, shift_value, seq_list, nei_list, nei_num, batch_pednum 
    # ]

    return data

class NBADataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, obs_len=5, pred_len=10, training=True
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        # self.norm_lap_matr = norm_lap_matr

        if training:
            data_root = './data/files/nba_train.npy'
        else:
            data_root = './data/files/nba_test.npy'

        self.trajs = np.load(data_root) #(N,15,11,2)
        self.trajs /= (94/28) 
        if training:
            self.trajs = self.trajs[:32500]
        else:
            self.trajs = self.trajs[:12500]
            # self.trajs = self.trajs[12500:25000]

        self.batch_len = len(self.trajs)
        print(self.batch_len)
        

        self.traj_abs = torch.from_numpy(self.trajs).type(torch.float)
        self.traj_norm = torch.from_numpy(self.trajs-self.trajs[:,self.obs_len-1:self.obs_len]).type(torch.float)

        self.traj_abs = self.traj_abs.permute(0,2,1,3)
        self.traj_norm = self.traj_norm.permute(0,2,1,3)
        self.actor_num = self.traj_abs.shape[1]
        # print(self.traj_abs.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):
        # print(self.traj_abs.shape)
        pre_motion_3D = self.traj_abs[index, :, :self.obs_len, :]
        fut_motion_3D = self.traj_abs[index, :, self.obs_len:, :]
        pre_motion_mask = torch.ones(11,self.obs_len)
        fut_motion_mask = torch.ones(11,self.pred_len)
        out = [
            pre_motion_3D, fut_motion_3D,
            pre_motion_mask, fut_motion_mask
        ]
        return out
    