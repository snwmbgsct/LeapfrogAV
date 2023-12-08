from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
from PIL import Image, ImageDraw


class DrivingData(Dataset):
    def __init__(self, data_dir, n_neighbors):
        self.data_list = glob.glob(data_dir)
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes'] 
        map_lanes = data['lanes']
        map_crosswalks = data['crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt

train_set_path = '/home/arc/nuplan/processed_data/train/*.npz'
train_set = DrivingData(train_set_path, 14)  # try num_neighbors + ego = 11
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)

for i, data in enumerate(train_loader):
    ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt = data
    # print(ego.shape)                   # 1, 21, 7       (21 timesteps, 7 features): x, y, heading, vx, vy, ax, ay
    # print(neighbors.shape)             # 1, 20, 21, 11  (20 neighbors, 21 timesteps, 11 features): x, y, heading, vx, vy, heading_rate, length, width
    # print(ego_future_gt.shape)         # 1, 80, 2       (80 timesteps, 2 features)
    # print(neighbors_future_gt.shape)   # 1, 10, 80, 3   (10 neighbors, 80 timesteps, 3 features), 10 = n_neighbors
    neighbor0 = neighbors[0, 0, :, :]
    
    image = Image.new('RGB', (200, 200), (255, 255, 255))
    drawer = ImageDraw.Draw(image)

    for t in range(20):
        # visualize the trajectory with bounding box
        occupacy = t / 20
        x = int((neighbor0[t, 0].item() + 15) * 5)
        y = int((neighbor0[t, 1].item() + 15) * 5)
        print(x, y)
        drawer.point((x, y), fill=(t * 10, t * 10, t * 10))
    
    
    break

image.show()