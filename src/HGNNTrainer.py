import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNNModel(nn.Module):
    """GNN model for heterogeneous data with per-node-type processing."""
    
    def __init__(self, metadata, hidden_dim, output_dim):
        super().__init__()
        
        self.convs = nn.ModuleDict()
        self.linears = nn.ModuleDict()
        
        # Define layers separately for each node type
        for node_type in metadata[0]:  # metadata[0] contains node types
            self.convs[node_type] = nn.ModuleList([
                SAGEConv((-1, -1), hidden_dim),
                SAGEConv((-1, -1), hidden_dim),
                SAGEConv((-1, -1), hidden_dim)
            ])
            self.linears[node_type] = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_dict, edge_index_dict):
        """Processes each node type separately."""
        out_dict = {}

        for node_type, x in x_dict.items():
            if node_type in self.convs:
                x = F.relu(self.convs[node_type][0](x, edge_index_dict.get(node_type, None)))
                x = F.relu(self.convs[node_type][1](x, edge_index_dict.get(node_type, None)))
                x = F.relu(self.convs[node_type][2](x, edge_index_dict.get(node_type, None)))
                x = self.linears[node_type](x)
            out_dict[node_type] = x

        return out_dict

import torch
import torch.optim as optim
from torch_geometric.nn import to_hetero

class HGNNTrainer:
    def __init__(self, hetero_data, node_type, hidden_dim=32, learning_rate=0.001, dropout_rate=0.5, wd=1e-3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hetero_data = hetero_data.to(self.device)
        self.node_type = node_type

        # Get output classes dynamically
        output_dim = int(self.hetero_data[node_type].y.max().item()) + 1

        # Initialize model
        self.model = GNNModel(self.hetero_data.metadata(), hidden_dim, output_dim)
        
        # Convert model to heterogeneous
        self.model = to_hetero(self.model, self.hetero_data.metadata(), aggr='sum').to(self.device)

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=wd)
        self.criterion = torch.nn.CrossEntropyLoss()
