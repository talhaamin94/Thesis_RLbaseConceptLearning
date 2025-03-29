import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv, RGCNConv, SAGEConv, JumpingKnowledge
from sklearn.metrics import accuracy_score, f1_score
import time
from utils.Graph_to_Homogeneous import Graph_to_Homogeneous
import os
from datetime import date,datetime
import glob
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class FastRGCNGNN(nn.Module):
    def __init__(self, homo_data, hidden_dim=16, num_bases=30, dropout_rate=0.3):
        super().__init__()
        in_channels = homo_data.x.shape[1]
        num_relations = len(homo_data.edge_type.unique())
        
        num_classes = int(homo_data.y.max().item()) + 1

        self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations, num_bases=num_bases)
        self.conv2 = FastRGCNConv(hidden_dim, num_classes, num_relations=num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(p=dropout_rate)  # Add dropout


        self._reset_parameters()

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.dropout(x)  # Apply dropout after first layer
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)

# class FastRGCNGNN(nn.Module):
#     def __init__(self, homo_data, hidden_dim=16, num_bases=30, dropout_rate=0.3):
#         super().__init__()
#         in_channels = homo_data.x.shape[1]
#         num_relations = len(homo_data.edge_type.unique())
#         num_classes = int(homo_data.y.max().item()) + 1

#         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations, num_bases=num_bases)
#         self.conv2 = FastRGCNConv(hidden_dim, hidden_dim * 2, num_relations=num_relations, num_bases=num_bases)
#         self.conv3 = FastRGCNConv(hidden_dim * 2, num_classes, num_relations=num_relations, num_bases=num_bases)

#         self.dropout = nn.Dropout(p=dropout_rate)  

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for param in self.parameters():
#             if param.dim() > 1:
#                 torch.nn.init.xavier_uniform_(param)

#     def forward(self, x, edge_index, edge_type):
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.dropout(x)

#         x = F.relu(self.conv2(x, edge_index, edge_type))
#         x = self.dropout(x)  

#         x = self.conv3(x, edge_index, edge_type)  # Third layer
#         return F.log_softmax(x, dim=1)  # Classification

class GNNTrainer:
    def __init__(self, hetero_data, node_type, hidden_dim=16, learning_rate=0.005, dropout_rate=0.3, wd = 1e-2):
        self.hetero_data = hetero_data
        self.node_type = node_type

        if len(hetero_data.edge_types) > 5:
            self.num_bases = max(10, len(hetero_data.edge_types)//3)
        else:
            self.num_bases = len(hetero_data.edge_types)

        # print(f"Number of Bases: {self.num_bases}")

        # Convert heterogeneous data to homogeneous
        self.graph_converter = Graph_to_Homogeneous(self.hetero_data)
        self.homo_data = self.graph_converter.get_homogeneous_data()
        self.node_mappings = self.graph_converter.get_node_mapping()
        # print(self.node_mappings)
        self.model = FastRGCNGNN(self.homo_data, hidden_dim, self.num_bases, dropout_rate)
       

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay = wd)
        # Use weighted loss to handle imbalance
        self.criterion = nn.NLLLoss()
        # Generate model save path
        date_str = datetime.now().strftime("%Y%m%d")
        self.best_model_path = f"models/{self.node_type}_{date_str}.pth"
        self.best_test_f1 = 0 
        self.best_train_f1 = 0
        
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        x = self.homo_data.x
        edge_index = self.homo_data.edge_index
        edge_type = self.homo_data.edge_type

        out = self.model(x, edge_index, edge_type)
        loss = self.criterion(out[self.homo_data.train_mask], self.homo_data.y[self.homo_data.train_mask])

        loss.backward()
        self.optimizer.step()

        return float(loss)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        x = self.homo_data.x
        edge_index = self.homo_data.edge_index
        edge_type = self.homo_data.edge_type

        pred = self.model(x, edge_index, edge_type).argmax(dim=-1)
        y_train = self.homo_data.y[self.homo_data.train_mask]
        y_test = self.homo_data.y[self.homo_data.test_mask]

        train_acc = (pred[self.homo_data.train_mask] == y_train).float().mean().item()
        test_acc = (pred[self.homo_data.test_mask] == y_test).float().mean().item()

        train_f1 = f1_score(y_train.cpu().numpy(), pred[self.homo_data.train_mask].cpu().numpy(), average='weighted')
        test_f1 = f1_score(y_test.cpu().numpy(), pred[self.homo_data.test_mask].cpu().numpy(), average='weighted')

        return train_acc, test_acc, train_f1, test_f1


        
    def run_training(self, epochs=50):
        times = []
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            start = time.time()
            loss = self.train()
            train_acc, test_acc, train_f1, test_f1 = self.test()
            times.append(time.time() - start)

            if test_f1 >= self.best_test_f1 and train_f1 >= self.best_train_f1:
                self.best_test_f1 = test_f1
                self.best_train_f1 = train_f1
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.best_model_path)

        median_time = torch.tensor(times).median().item()

        # Load best model before returning stats
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))

        return {
            "best_epoch": best_epoch,
            "best_test_f1": round(self.best_test_f1,4),
            "best_train_f1": round(self.best_train_f1,4),
            "median_epoch_time": round(median_time,4),
        }


    def get_positive_nodes(self):
        """Retrieve positive nodes (label = 1) in terms of hetero_data indexes."""
        positive_nodes = []
        for _, (node_type, instance_id) in self.node_mappings.items():
            if node_type == self.node_type:
                label = self.hetero_data[node_type].y[instance_id].item()
                if label != -1 and label == 1:
                    positive_nodes.append(instance_id)
        # print("positive_nodes in gnn: ", positive_nodes)
        return positive_nodes

    def get_negative_nodes(self):
        """Retrieve negative nodes (label = 0) in terms of hetero_data indexes."""
        negative_nodes = []
        for _, (node_type, instance_id) in self.node_mappings.items():
            if node_type == self.node_type:
                label = self.hetero_data[node_type].y[instance_id].item()
                if label != -1 and label == 0:
                    negative_nodes.append(instance_id)
        return negative_nodes

    
    def load_model(self, model_path=None):
            """Loads the latest trained model from the 'models' folder or a specified path."""

            models_folder = "models/"
            
       
            if model_path is None:
                # Find all models that match the expected naming format
                model_files = sorted(glob.glob(os.path.join(models_folder, f"best_model_{self.node_type}_*.pth")), reverse=True)

                if not model_files:
                    print("⚠ No saved models found in 'models/' directory!")
                    return

                model_path = model_files[0]  # Select the latest saved model by date

            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                print(f" Model loaded from {model_path}")
            else:
                print(f" Error: Model file {model_path} not found!")