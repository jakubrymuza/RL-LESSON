import torch.nn as nn
from rl_algorithm.dqn.config import device

class BaseModel(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        shape = n*m
        
        # self.image_conv = nn.Sequential(
        #     nn.Linear(shape, shape//2),
        #     nn.ReLU(),
        #     nn.Linear(shape//2, shape//4),
        #     nn.ReLU(),
        #     nn.Linear(shape//4, 64),
        #     nn.ReLU()
        # ).to(device)
        
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), # TODO: 4 past frames 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        ).to(device)
        
    def proc(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = x.reshape(x.shape[0], -1)
        q_values = self.classifier(x)
        return q_values