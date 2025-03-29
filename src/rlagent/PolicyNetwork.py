import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import HeteroData


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_space, learning_rate=0.001):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return torch.softmax(self.fc2(x), dim=-1).squeeze(0)  # Remove batch dim


    def update(self, state_batch, action_batch, rewards):
        action_probs = self.forward(state_batch)

        # Ensure action_probs is at least 2D
        if action_probs.dim() == 1:
            action_probs = action_probs.unsqueeze(0)

        # Ensure action_batch is (batch_size, 1)
        action_batch = action_batch.view(-1, 1)

        # # Debugging Prints
        # print(f"action_probs shape: {action_probs.shape}")  # (batch_size, action_space)
        # print(f"action_batch shape: {action_batch.shape}")  # (batch_size, 1)
        # print(f"action_probs: {action_probs}")  
        # print(f"action_batch: {action_batch}")  

        # Validate indices before gathering
        max_index = action_probs.shape[1] - 1
        action_batch = torch.clamp(action_batch, 0, max_index)

        # Apply gather and compute log probability
        action_log_probs = torch.log(action_probs.gather(1, action_batch).squeeze())

        loss = -torch.sum(action_log_probs * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()