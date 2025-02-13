# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import FastRGCNConv
# import os

# # Define the FastRGCN-based model
# class FastRGCNGNN(torch.nn.Module):
#     def __init__(self, num_relations, in_channels=3, hidden_dim=32, out_dim=64):
#         super().__init__()
#         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
#         self.conv2 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
#         self.lin = torch.nn.Linear(out_dim, 2)  # Binary classification (positive or negative)

#     def forward(self, x, edge_index, edge_type=None):
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.conv2(x, edge_index, edge_type)
#         return self.lin(x)


# # Define the GNN Trainer Class
# class GNNTrainer:
#     def __init__(self, hetero_data, node_type, in_channels=3, hidden_dim=32, out_dim=64):
#         self.hetero_data = hetero_data
#         self.node_type = node_type

#         # ✅ Define model path
#         self.model_dir = "./saved_models"
#         self.model_path = f"{self.model_dir}/{node_type}_gnn.pth"

#         # ✅ Ensure directory exists
#         os.makedirs(self.model_dir, exist_ok=True)

#         self.model = FastRGCNGNN(num_relations=len(hetero_data.edge_types), in_channels=in_channels, hidden_dim=hidden_dim, out_dim=out_dim)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def save_model(self):
#         """Saves the trained model to a file."""
#         torch.save(self.model.state_dict(), self.model_path)
#         print(f"GNN model saved to {self.model_path}")

#     def load_model(self):
#         """Loads the trained model if it exists."""
#         if os.path.exists(self.model_path):
#             self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
#             print(f" Loaded GNN model from {self.model_path}")

#     def train(self, epochs=20):
#         """Train the GNN model and save it."""
#         if os.path.exists(self.model_path):
#             self.load_model()
#             return self.model

#         for epoch in range(epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
#             x = self.hetero_data[self.node_type].x
#             edge_index = self.hetero_data[('A', 'connected_to', 'C')].edge_index
#             edge_type = self.hetero_data[('A', 'connected_to', 'C')].edge_type
#             out = self.model(x, edge_index, edge_type)
#             loss = self.criterion(out[self.hetero_data[self.node_type].train_mask], self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask])
#             loss.backward()
#             self.optimizer.step()
#             preds = out.argmax(dim=1)
#             accuracy = (preds[self.hetero_data[self.node_type].train_mask] == self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask]).float().mean().item()
#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

#         return self.model

    
#     def get_positive_nodes(self):
#         """Returns the indices of nodes classified as positive by the trained GNN model."""
#         self.model.eval()  # Ensure the model is in evaluation mode

#         with torch.no_grad():
#             # Get features & edges
#             x = self.hetero_data[self.node_type].x
#             edge_index = self.hetero_data[('A', 'connected_to', 'C')].edge_index
#             edge_type = self.hetero_data[('A', 'connected_to', 'C')].edge_type
            
#             # Forward pass through the model
#             out = self.model(x, edge_index, edge_type)

#             # Apply softmax to get probabilities
#             probs = torch.softmax(out, dim=1)

#             # Classify nodes: 1 = Positive, 0 = Negative
#             preds = torch.argmax(probs, dim=1)

#             # Filter only positively classified nodes
#             pos_nodes = torch.where(preds == 1)[0].tolist()

#             # print(f"Predicted Class Labels: {preds.tolist()}")  # Debugging print
#             # print(f"Positive Nodes: {pos_nodes}")  # Debugging print

#             return pos_nodes

#     def get_negative_nodes(self):
#         """Returns the indices of nodes classified as negative by the trained GNN model."""
#         self.model.eval()  # Ensure the model is in evaluation mode

#         with torch.no_grad():
#             # Get features & edges
#             x = self.hetero_data[self.node_type].x
#             edge_index = self.hetero_data[('A', 'connected_to', 'C')].edge_index
#             edge_type = self.hetero_data[('A', 'connected_to', 'C')].edge_type
            
#             # Forward pass through the model
#             out = self.model(x, edge_index, edge_type)

#             # Apply softmax to get probabilities
#             probs = torch.softmax(out, dim=1)

#             # Classify nodes: 1 = Positive, 0 = Negative
#             preds = torch.argmax(probs, dim=1)

#             # Filter only negatively classified nodes
#             neg_nodes = torch.where(preds == 0)[0].tolist()

#             return neg_nodes
    

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import FastRGCNConv
# import os

# class FastRGCNGNN(torch.nn.Module):
#     def __init__(self, hetero_data, node_type, num_relations, hidden_dim=32, out_dim=64):
#         super().__init__()

#         # Dynamically determine the input feature dimension
#         in_channels = hetero_data[node_type].x.shape[1]
#         print(f"[DEBUG] Detected in_channels for '{node_type}': {in_channels}")

#         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
#         self.conv2 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
#         self.lin = torch.nn.Linear(out_dim, 2)  # Binary classification

#     def forward(self, x, edge_index, edge_type):
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.conv2(x, edge_index, edge_type)
#         return self.lin(x)


# class GNNTrainer:
#     def __init__(self, hetero_data, node_type, hidden_dim=32, out_dim=64):
#         self.hetero_data = hetero_data
#         self.node_type = node_type

#         self.model_dir = "./saved_models"
#         self.model_path = f"{self.model_dir}/{node_type}_gnn.pth"

#         os.makedirs(self.model_dir, exist_ok=True)

#         num_relations = len(hetero_data.edge_types)
#         self.model = FastRGCNGNN(hetero_data, node_type, num_relations, hidden_dim, out_dim)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
#         self.criterion = torch.nn.CrossEntropyLoss()

#     def train(self, epochs=20):
#         """Train the GNN model dynamically across all edge types associated with the target node type."""
#         self.model.train()

#         # Dynamically retrieve all edges associated with the target node type
#         edge_indices = []
#         edge_types = []
#         for edge_type in self.hetero_data.edge_types:
#             if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
#                 edge_indices.append(self.hetero_data[edge_type].edge_index)
#                 if 'edge_type' in self.hetero_data[edge_type]:
#                     edge_types.append(self.hetero_data[edge_type].edge_type)
#                 else:
#                     edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

#         # Concatenate edges
#         if edge_indices:
#             edge_index = torch.cat(edge_indices, dim=1)
#             edge_type = torch.cat(edge_types) if edge_types else None
#         else:
#             raise ValueError(f"No edges found for node type '{self.node_type}'")

#         x = self.hetero_data[self.node_type].x
#         y = self.hetero_data[self.node_type].y
#         train_mask = self.hetero_data[self.node_type].train_mask

#         # Adjust input channels if needed
#         in_channels = x.shape[1] if x.dim() == 2 else 1
#         self.model.conv1.in_channels = in_channels

#         for epoch in range(epochs):
#             self.optimizer.zero_grad()
#             out = self.model(x, edge_index, edge_type)
#             loss = self.criterion(out[train_mask], y[train_mask])
#             loss.backward()
#             self.optimizer.step()

#             # Calculate accuracy on the training set
#             preds = out.argmax(dim=1)
#             accuracy = (preds[train_mask] == y[train_mask]).float().mean().item()

#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

#         self.save_model()


#     def evaluate(self):
#         """Evaluates the model dynamically across all edge types associated with the target node type."""
#         self.model.eval()
#         with torch.no_grad():
#             x = self.hetero_data[self.node_type].x
#             edge_indices = []
#             edge_types = []

#             # Collect all edges associated with the target node type
#             for edge_type in self.hetero_data.edge_types:
#                 if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
#                     edge_indices.append(self.hetero_data[edge_type].edge_index)
#                     if 'edge_type' in self.hetero_data[edge_type]:
#                         edge_types.append(self.hetero_data[edge_type].edge_type)
#                     else:
#                         # Assign a default edge type if missing
#                         edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

#             # Concatenate all edges
#             if edge_indices:
#                 edge_index = torch.cat(edge_indices, dim=1)
#                 edge_type = torch.cat(edge_types) if edge_types else None
#             else:
#                 raise ValueError(f"No edges found for node type '{self.node_type}'")

#             # Forward pass
#             out = self.model(x, edge_index, edge_type)
#             preds = out.argmax(dim=1)
#             test_mask = self.hetero_data[self.node_type].test_mask
#             y_true = self.hetero_data[self.node_type].y[test_mask]
#             y_pred = preds[test_mask]

#             # Calculate metrics
#             accuracy = (y_pred == y_true).float().mean().item()
#             print(f"Test Accuracy: {accuracy:.4f}")

#             return accuracy

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import FastRGCNConv
# from sklearn.metrics import f1_score
# import os

# # Define the FastRGCN-based model
# class FastRGCNGNN(torch.nn.Module):
#     def __init__(self, hetero_data, node_type, num_relations, hidden_dim=32, out_dim=64):
#         super().__init__()

#         # Dynamically determine the input feature dimension
#         in_channels = hetero_data[node_type].x.shape[1]
#         print(f"[DEBUG] Detected in_channels for '{node_type}': {in_channels}")

#         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
#         self.conv2 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
#         self.lin = torch.nn.Linear(out_dim, 2)  # Binary classification

#     def forward(self, x, edge_index, edge_type=None):
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.conv2(x, edge_index, edge_type)
#         return self.lin(x)


# # Define the GNN Trainer Class
# class GNNTrainer:
#     def __init__(self, hetero_data, node_type, hidden_dim=32, out_dim=64):
#         self.hetero_data = hetero_data
#         self.node_type = node_type

#         # ✅ Define model path
#         self.model_dir = "./saved_models"
#         self.model_path = f"{self.model_dir}/{node_type}_gnn.pth"

#         # ✅ Ensure directory exists
#         os.makedirs(self.model_dir, exist_ok=True)

#         num_relations = len(hetero_data.edge_types)
#         self.model = FastRGCNGNN(hetero_data, node_type, num_relations, hidden_dim, out_dim)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
#         self.criterion = torch.nn.CrossEntropyLoss()

#         self.positive_nodes = []
#         self.negative_nodes = []

#     def save_model(self):
#         """Saves the trained model to a file."""
#         torch.save(self.model.state_dict(), self.model_path)
#         print(f"GNN model saved to {self.model_path}")

#     def load_model(self):
#         """Loads the trained model if it exists."""
#         if os.path.exists(self.model_path):
#             self.model.load_state_dict(torch.load(self.model_path))
#             print(f" Loaded GNN model from {self.model_path}")

#     def train(self, epochs=100):
#         """Train the GNN model and save it."""
#         if os.path.exists(self.model_path):
#             self.load_model()
#             return self.model

#         for epoch in range(epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
#             x = self.hetero_data[self.node_type].x

#             # Dynamically collect all edges and types
#             edge_indices = []
#             edge_types = []
#             for edge_type in self.hetero_data.edge_types:
#                 if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
#                     edge_indices.append(self.hetero_data[edge_type].edge_index)
#                     if 'edge_type' in self.hetero_data[edge_type]:
#                         edge_types.append(self.hetero_data[edge_type].edge_type)
#                     else:
#                         edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

#             # Concatenate edges
#             if edge_indices:
#                 edge_index = torch.cat(edge_indices, dim=1)
#                 edge_type = torch.cat(edge_types) if edge_types else None
#             else:
#                 raise ValueError(f"No edges found for node type '{self.node_type}'")

#             out = self.model(x, edge_index, edge_type)
#             loss = self.criterion(out[self.hetero_data[self.node_type].train_mask], self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask])
#             loss.backward()
#             self.optimizer.step()
#             preds = out.argmax(dim=1)
#             accuracy = (preds[self.hetero_data[self.node_type].train_mask] == self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask]).float().mean().item()

#             # Calculate F1 score
#             true_labels = self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask].cpu().numpy()
#             pred_labels = preds[self.hetero_data[self.node_type].train_mask].cpu().numpy()
#             f1 = f1_score(true_labels, pred_labels, average='weighted')

#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

#         # Cache positive and negative nodes
#         self._cache_node_predictions()

#         self.save_model()
#         return self.model

#     def _cache_node_predictions(self):
#         """Caches the indices of positive and negative nodes for quick access."""
#         self.model.eval()
#         with torch.no_grad():
#             x = self.hetero_data[self.node_type].x

#             edge_indices = []
#             edge_types = []
#             for edge_type in self.hetero_data.edge_types:
#                 if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
#                     edge_indices.append(self.hetero_data[edge_type].edge_index)
#                     if 'edge_type' in self.hetero_data[edge_type]:
#                         edge_types.append(self.hetero_data[edge_type].edge_type)
#                     else:
#                         edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

#             if edge_indices:
#                 edge_index = torch.cat(edge_indices, dim=1)
#                 edge_type = torch.cat(edge_types) if edge_types else None
#             else:
#                 raise ValueError(f"No edges found for node type '{self.node_type}'")

#             out = self.model(x, edge_index, edge_type)
#             probs = torch.softmax(out, dim=1)
#             preds = torch.argmax(probs, dim=1)
#             self.positive_nodes = torch.where(preds == 1)[0].tolist()
#             self.negative_nodes = torch.where(preds == 0)[0].tolist()

#     def get_positive_nodes(self):
#         """Returns the indices of nodes classified as positive (cached during training)."""
#         return self.positive_nodes

#     def get_negative_nodes(self):
#         """Returns the indices of nodes classified as negative (cached during training)."""
#         return self.negative_nodes
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import os

# Define the FastRGCN-based model
class FastRGCNGNN(torch.nn.Module):
    def __init__(self, hetero_data, node_type, num_relations, hidden_dim=32, out_dim=64):
        super().__init__()

        # Dynamically determine the input feature dimension
        in_channels = hetero_data[node_type].x.shape[1]
        print(f"[DEBUG] Detected in_channels for '{node_type}': {in_channels}")

        self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
        self.conv2 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
        self.lin = torch.nn.Linear(out_dim, 2)  # Binary classification

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return self.lin(x)


# Define the GNN Trainer Class
class GNNTrainer:
    def __init__(self, hetero_data, node_type, hidden_dim=32, out_dim=64):
        self.hetero_data = hetero_data
        self.node_type = node_type

        #  Define model path
        self.model_dir = "./saved_models"
        self.model_path = f"{self.model_dir}/{node_type}_gnn.pth"

        # Ensure directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        num_relations = len(hetero_data.edge_types)
        self.model = FastRGCNGNN(hetero_data, node_type, num_relations, hidden_dim, out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.positive_nodes = []
        self.negative_nodes = []

    def save_model(self):
        """Saves the trained model to a file."""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"GNN model saved to {self.model_path}")

    def load_model(self):
        """Loads the trained model if it exists."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f" Loaded GNN model from {self.model_path}")

    def train(self, epochs=100):
        """Train the GNN model and save it."""
        if os.path.exists(self.model_path):
            self.load_model()
            return self.model

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            x = self.hetero_data[self.node_type].x

            # Dynamically collect all edges and types
            edge_indices = []
            edge_types = []
            for edge_type in self.hetero_data.edge_types:
                if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
                    edge_indices.append(self.hetero_data[edge_type].edge_index)
                    if 'edge_type' in self.hetero_data[edge_type]:
                        edge_types.append(self.hetero_data[edge_type].edge_type)
                    else:
                        edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

            # Concatenate edges
            if edge_indices:
                edge_index = torch.cat(edge_indices, dim=1)
                edge_type = torch.cat(edge_types) if edge_types else None
            else:
                raise ValueError(f"No edges found for node type '{self.node_type}'")

            out = self.model(x, edge_index, edge_type)
            loss = self.criterion(out[self.hetero_data[self.node_type].train_mask], self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask])
            loss.backward()
            self.optimizer.step()
            preds = out.argmax(dim=1)
            accuracy = (preds[self.hetero_data[self.node_type].train_mask] == self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask]).float().mean().item()

            # Calculate F1 score
            true_labels = self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask].cpu().numpy()
            pred_labels = preds[self.hetero_data[self.node_type].train_mask].cpu().numpy()
            f1 = f1_score(true_labels, pred_labels, average='weighted')

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

        # Cache positive and negative nodes
        self._cache_node_predictions()
        self._compute_training_statistics()

        self.save_model()
        return self.model

    def _cache_node_predictions(self):
        """Caches the indices of positive and negative nodes for quick access."""
        self.model.eval()
        with torch.no_grad():
            x = self.hetero_data[self.node_type].x

            edge_indices = []
            edge_types = []
            for edge_type in self.hetero_data.edge_types:
                if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
                    edge_indices.append(self.hetero_data[edge_type].edge_index)
                    if 'edge_type' in self.hetero_data[edge_type]:
                        edge_types.append(self.hetero_data[edge_type].edge_type)
                    else:
                        edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

            if edge_indices:
                edge_index = torch.cat(edge_indices, dim=1)
                edge_type = torch.cat(edge_types) if edge_types else None
            else:
                raise ValueError(f"No edges found for node type '{self.node_type}'")

            out = self.model(x, edge_index, edge_type)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            self.positive_nodes = torch.where(preds == 1)[0].tolist()
            self.negative_nodes = torch.where(preds == 0)[0].tolist()

    def _compute_training_statistics(self):
        """Computes and prints statistics for the training set."""
        train_mask = self.hetero_data[self.node_type].train_mask
        true_labels = self.hetero_data[self.node_type].y[train_mask].cpu().numpy()
        x = self.hetero_data[self.node_type].x

        edge_indices = []
        edge_types = []
        for edge_type in self.hetero_data.edge_types:
            if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
                edge_indices.append(self.hetero_data[edge_type].edge_index)
                if 'edge_type' in self.hetero_data[edge_type]:
                    edge_types.append(self.hetero_data[edge_type].edge_type)
                else:
                    edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_type = torch.cat(edge_types) if edge_types else None
        else:
            raise ValueError(f"No edges found for node type '{self.node_type}'")

        self.model.eval()
        with torch.no_grad():
            out = self.model(x, edge_index, edge_type)
            preds = out.argmax(dim=1)[train_mask].cpu().numpy()

        accuracy = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='weighted')
        cm = confusion_matrix(true_labels, preds)

        print("\n[TRAINING SET STATISTICS]")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

    def get_positive_nodes(self):
        """Returns the indices of nodes classified as positive (cached during training)."""
        return self.positive_nodes

    def get_negative_nodes(self):
        """Returns the indices of nodes classified as negative (cached during training)."""
        return self.negative_nodes



    def evaluate_test_set(self):
        """Evaluates the model on the test set and prints predictions and metrics."""
        self.model.eval()
        with torch.no_grad():
            x = self.hetero_data[self.node_type].x

            # Collect all edges associated with the target node type
            edge_indices = []
            edge_types = []
            for edge_type in self.hetero_data.edge_types:
                if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
                    edge_indices.append(self.hetero_data[edge_type].edge_index)
                    if 'edge_type' in self.hetero_data[edge_type]:
                        edge_types.append(self.hetero_data[edge_type].edge_type)
                    else:
                        edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

            if edge_indices:
                edge_index = torch.cat(edge_indices, dim=1)
                edge_type = torch.cat(edge_types) if edge_types else None
            else:
                raise ValueError(f"No edges found for node type '{self.node_type}'")

            # Forward pass
            out = self.model(x, edge_index, edge_type)
            preds = out.argmax(dim=1)
            test_mask = self.hetero_data[self.node_type].test_mask
            y_true = self.hetero_data[self.node_type].y[test_mask].cpu().numpy()
            y_pred = preds[test_mask].cpu().numpy()

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            cm = confusion_matrix(y_true, y_pred)

            # Print results
            print("\n[TEST SET PREDICTIONS]")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)

            # Print predictions
            print(f"\nPredictions (first 20): {y_pred[:20]}")

            return y_pred, y_true, cm


    # Add this method to your `GNNTrainer` class
    # Then you can call it like:
    # test_predictions, test_labels, test_cm = gnn.evaluate_test_set()
