import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import HeteroData


class RLGraphEnv:
    def __init__(self, hetero_data, trainer):
        self.hetero_data = hetero_data
        self.trainer = trainer #transE trainer
        self.current_node = None
        self.path = []
        self.tracked_paths = []

    def reset(self, start_node):
        self.current_node = start_node
        self.path = [start_node]
        self.typed_path = []  # reset typed path for tracking full transitions
        return self.get_state_embedding()

    def step(self, action_node, edge_type):
        # print(f"Moving from node {self.current_node} to node {action_node} via edge {edge_type}")

        src_local = self.trainer.global_to_node[self.current_node]
        dst_local = self.trainer.global_to_node[action_node]

        self.typed_path.append((self.current_node, edge_type, action_node, src_local, dst_local))
        # print(self.typed_path)
        self.path.append(action_node)
        self.current_node = action_node

        return self.get_state_embedding(), False  # Always False; RLTrainer controls rollout length


    def get_neighbors(self, node_global):
        neighbors = []
       
        if node_global not in self.trainer.global_to_node:
            return []

        node_type, local_index = self.trainer.global_to_node[node_global]

        for edge_type in self.hetero_data.edge_types:
            if edge_type[0] != node_type:
                continue  # only check edges starting from the right node type

            edge_index = self.hetero_data[edge_type].edge_index
            for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                if src == local_index:
                    dst_global = self.trainer.node_mapping.get((edge_type[2], dst))
                    if dst_global is not None:
                        neighbors.append((dst_global, edge_type))

        return neighbors  # returns empty list if no valid neighbors found

    def get_node_embedding(self, node):
        return self.trainer.get_embedding(node)
    
    
    def get_state_embedding(self):
        current_emb = self.trainer.transe_model.entity_embeddings(
            torch.tensor([self.current_node])
        ).squeeze(0)

        if not hasattr(self, "typed_path") or not self.typed_path:
            path_emb = torch.zeros_like(current_emb)
        else:
            embeddings = []
            for src, edge_type, dst, _, _ in self.typed_path:
                src_emb = self.trainer.transe_model.entity_embeddings(
                    torch.tensor([src])
                ).squeeze(0)
                rel_index = self.trainer.relation_to_index[edge_type]
                rel_emb = self.trainer.transe_model.relation_embeddings(
                    torch.tensor([rel_index])
                ).squeeze(0)
                dst_emb = self.trainer.transe_model.entity_embeddings(
                    torch.tensor([dst])
                ).squeeze(0)

                embeddings.append(src_emb + rel_emb + dst_emb)

            path_emb = torch.stack(embeddings).mean(dim=0)

        return torch.cat([current_emb, path_emb], dim=0)
