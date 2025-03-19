import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import HeteroData


class RLGraphEnv:
    def __init__(self, hetero_data, trainer):
        self.hetero_data = hetero_data
        self.trainer = trainer
        self.current_node = None
        self.path = []
        self.tracked_paths = []

    def reset(self, start_node):
        self.current_node = start_node
        self.path = [start_node]
        return self.current_node

    def step(self, action):
        next_node = action
        self.path.append(next_node)
        self.current_node = next_node
        done = len(self.path) >= 2
        if done:
            self.tracked_paths.append(self.path.copy())
        return next_node, done

    def get_neighbors(self, node):
        neighbors = set()
        for (src, _, tgt) in self.trainer.unique_edges:
            if src == node:
                neighbors.add(tgt)
        return list(neighbors) if neighbors else list(self.trainer.global_to_node.keys())

    def get_node_embedding(self, node):
        return self.trainer.get_embedding(node)