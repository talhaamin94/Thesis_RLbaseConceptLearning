# # import torch
# # import torch.nn as nn
# # from torch.nn import BatchNorm1d
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch_geometric.nn import FastRGCNConv
# # from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
# # import os

# # class FastRGCNGNN(torch.nn.Module):
# #     def __init__(self, hetero_data, node_type, num_relations, hidden_dim=64, out_dim=64, dropout_rate = 0.3):
# #         super().__init__()

# #         # Dynamically determine input feature dimension
# #         in_channels = hetero_data[node_type].x.shape[1]
# #         print(f"[DEBUG] Detected in_channels for '{node_type}': {in_channels}")

# #         # Define 3-layer GNN with BatchNorm and Dropout
# #         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
# #         self.batch_norm1 = BatchNorm1d(hidden_dim)
        
# #         self.conv2 = FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
# #         self.batch_norm2 = BatchNorm1d(hidden_dim)

# #         self.conv3 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
# #         self.batch_norm3 = BatchNorm1d(out_dim)

# #         self.lin = torch.nn.Linear(out_dim, 2)  # Binary classification (0 or 1)
# #         self.dropout = nn.Dropout(p=dropout_rate)  # Dropout to prevent overfitting

# #     def forward(self, x, edge_index, edge_type=None):
# #         x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_type)))
# #         x = self.dropout(x)

# #         x = F.relu(self.batch_norm2(self.conv2(x, edge_index, edge_type)))
# #         x = self.dropout(x)

# #         x = self.batch_norm3(self.conv3(x, edge_index, edge_type))
# #         return self.lin(x)


# # # Define the GNN Trainer Class
# # class GNNTrainer:
# #     def __init__(self, hetero_data, node_type, hidden_dim=16, out_dim=64, learning_rate=0.01, dropout_rate = 0.3):
# #         self.hetero_data = hetero_data
# #         self.node_type = node_type

# #         #  Define model path
# #         self.model_dir = "./saved_models"
# #         self.model_path = f"{self.model_dir}/{node_type}_gnn.pth"

# #         # Ensure directory exists
# #         os.makedirs(self.model_dir, exist_ok=True)

# #         num_relations = len(hetero_data.edge_types)
# #         self.model = FastRGCNGNN(hetero_data, node_type, num_relations, hidden_dim, out_dim, dropout_rate)
# #         # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
# #         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
# #         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

# #         self.criterion = torch.nn.CrossEntropyLoss()

# #         self.positive_nodes = []
# #         self.negative_nodes = []

# #     def save_model(self):
# #         """Saves the trained model to a file."""
# #         torch.save(self.model.state_dict(), self.model_path)
# #         print(f"GNN model saved to {self.model_path}")

# #     def load_model(self):
# #         """Loads the trained model if it exists."""
# #         if os.path.exists(self.model_path):
# #             self.model.load_state_dict(torch.load(self.model_path))
# #             print(f" Loaded GNN model from {self.model_path}")

# #     def train(self, epochs = 20):
# #         """Train the GNN model and save it."""
# #         if os.path.exists(self.model_path):
# #             self.load_model()
# #             return self.model

# #         for epoch in range(epochs):
# #             self.model.train()
# #             self.optimizer.zero_grad()
# #             x = self.hetero_data[self.node_type].x

# #             # Dynamically collect all edges and types
# #             edge_indices = []
# #             edge_types = []
# #             for edge_type in self.hetero_data.edge_types:
# #                 if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
# #                     edge_indices.append(self.hetero_data[edge_type].edge_index)
# #                     if 'edge_type' in self.hetero_data[edge_type]:
# #                         edge_types.append(self.hetero_data[edge_type].edge_type)
# #                     else:
# #                         edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

# #             # Concatenate edges
# #             if edge_indices:
# #                 edge_index = torch.cat(edge_indices, dim=1)
# #                 edge_type = torch.cat(edge_types) if edge_types else None
# #             else:
# #                 raise ValueError(f"No edges found for node type '{self.node_type}'")

# #             out = self.model(x, edge_index, edge_type)
# #             loss = self.criterion(out[self.hetero_data[self.node_type].train_mask], self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask])
# #             loss.backward()
# #             self.optimizer.step()
# #             self.scheduler.step()
# #             preds = out.argmax(dim=1)
# #             accuracy = (preds[self.hetero_data[self.node_type].train_mask] == self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask]).float().mean().item()

# #             # Calculate F1 score
# #             true_labels = self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask].cpu().numpy()
# #             pred_labels = preds[self.hetero_data[self.node_type].train_mask].cpu().numpy()
# #             f1 = f1_score(true_labels, pred_labels, average='weighted')

# #             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

# #         # Cache positive and negative nodes
# #         self._cache_node_predictions()
# #         self._compute_training_statistics()

# #         # self.save_model()
# #         return self.model

# #     def _cache_node_predictions(self):
# #         """Caches the indices of positive and negative nodes for quick access."""
# #         self.model.eval()
# #         with torch.no_grad():
# #             x = self.hetero_data[self.node_type].x

# #             edge_indices = []
# #             edge_types = []
# #             for edge_type in self.hetero_data.edge_types:
# #                 if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
# #                     edge_indices.append(self.hetero_data[edge_type].edge_index)
# #                     if 'edge_type' in self.hetero_data[edge_type]:
# #                         edge_types.append(self.hetero_data[edge_type].edge_type)
# #                     else:
# #                         edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

# #             if edge_indices:
# #                 edge_index = torch.cat(edge_indices, dim=1)
# #                 edge_type = torch.cat(edge_types) if edge_types else None
# #             else:
# #                 raise ValueError(f"No edges found for node type '{self.node_type}'")

# #             out = self.model(x, edge_index, edge_type)
# #             probs = torch.softmax(out, dim=1)
# #             preds = torch.argmax(probs, dim=1)
# #             self.positive_nodes = torch.where(preds == 1)[0].tolist()
# #             self.negative_nodes = torch.where(preds == 0)[0].tolist()

# #     def _compute_training_statistics(self):
# #         """Computes and prints statistics for the training set."""
# #         train_mask = self.hetero_data[self.node_type].train_mask
# #         true_labels = self.hetero_data[self.node_type].y[train_mask].cpu().numpy()
# #         x = self.hetero_data[self.node_type].x

# #         edge_indices = []
# #         edge_types = []
# #         for edge_type in self.hetero_data.edge_types:
# #             if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
# #                 edge_indices.append(self.hetero_data[edge_type].edge_index)
# #                 if 'edge_type' in self.hetero_data[edge_type]:
# #                     edge_types.append(self.hetero_data[edge_type].edge_type)
# #                 else:
# #                     edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

# #         if edge_indices:
# #             edge_index = torch.cat(edge_indices, dim=1)
# #             edge_type = torch.cat(edge_types) if edge_types else None
# #         else:
# #             raise ValueError(f"No edges found for node type '{self.node_type}'")

# #         self.model.eval()
# #         with torch.no_grad():
# #             out = self.model(x, edge_index, edge_type)
# #             preds = out.argmax(dim=1)[train_mask].cpu().numpy()

# #         accuracy = accuracy_score(true_labels, preds)
# #         f1 = f1_score(true_labels, preds, average='weighted')
# #         cm = confusion_matrix(true_labels, preds)

# #         print("\n[TRAINING SET STATISTICS]")
# #         print(f"Accuracy: {accuracy:.4f}")
# #         print(f"F1 Score: {f1:.4f}")
# #         print("Confusion Matrix:")
# #         print(cm)

# #     def get_positive_nodes(self):
# #         """Returns the indices of nodes classified as positive (cached during training)."""
# #         return self.positive_nodes

# #     def get_negative_nodes(self):
# #         """Returns the indices of nodes classified as negative (cached during training)."""
# #         return self.negative_nodes



# #     def evaluate_test_set(self):
# #         """Evaluates the model on the test set and prints predictions and metrics."""
# #         self.model.eval()
# #         with torch.no_grad():
# #             x = self.hetero_data[self.node_type].x

# #             # Collect all edges associated with the target node type
# #             edge_indices = []
# #             edge_types = []
# #             for edge_type in self.hetero_data.edge_types:
# #                 if edge_type[0] == self.node_type or edge_type[2] == self.node_type:
# #                     edge_indices.append(self.hetero_data[edge_type].edge_index)
# #                     if 'edge_type' in self.hetero_data[edge_type]:
# #                         edge_types.append(self.hetero_data[edge_type].edge_type)
# #                     else:
# #                         edge_types.append(torch.zeros(self.hetero_data[edge_type].edge_index.shape[1], dtype=torch.long))

# #             if edge_indices:
# #                 edge_index = torch.cat(edge_indices, dim=1)
# #                 edge_type = torch.cat(edge_types) if edge_types else None
# #             else:
# #                 raise ValueError(f"No edges found for node type '{self.node_type}'")

# #             # Forward pass
# #             out = self.model(x, edge_index, edge_type)
# #             preds = out.argmax(dim=1)
# #             test_mask = self.hetero_data[self.node_type].test_mask
# #             y_true = self.hetero_data[self.node_type].y[test_mask].cpu().numpy()
# #             y_pred = preds[test_mask].cpu().numpy()

# #             # Calculate metrics
# #             accuracy = accuracy_score(y_true, y_pred)
# #             f1 = f1_score(y_true, y_pred, average='weighted')
# #             precision = precision_score(y_true, y_pred, average='weighted')
# #             recall = recall_score(y_true, y_pred, average='weighted')
# #             cm = confusion_matrix(y_true, y_pred)

# #             # Print results
# #             print("\n[TEST SET PREDICTIONS]")
# #             print(f"Accuracy: {accuracy:.4f}")
# #             print(f"Precision: {precision:.4f}")
# #             print(f"Recall: {recall:.4f}")
# #             print(f"F1 Score: {f1:.4f}")
# #             print("Confusion Matrix:")
# #             print(cm)

# #             # Print predictions
# #             # print(f"\nPredictions (first 20): {y_pred[:20]}")

# #             return y_pred, y_true, cm, f1


# import torch
# import torch.nn as nn
# from torch.nn import BatchNorm1d
# import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import FastRGCNConv
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
# import os

# class FastRGCNGNN(torch.nn.Module):
#     def __init__(self, hetero_data, node_type, num_relations, hidden_dim=64, out_dim=64, dropout_rate=0.3):
#         super().__init__()

#         # Dynamically determine input feature dimension
#         in_channels = hetero_data[node_type].x.shape[1]
#         print(f"[DEBUG] Detected in_channels for '{node_type}': {in_channels}")

#         # Define 4-layer GNN with BatchNorm, Dropout, and residual connections
#         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
#         self.batch_norm1 = BatchNorm1d(hidden_dim)

#         self.conv2 = FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
#         self.batch_norm2 = BatchNorm1d(hidden_dim)

#         self.conv3 = FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
#         self.batch_norm3 = BatchNorm1d(hidden_dim)

#         self.conv4 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
#         self.batch_norm4 = BatchNorm1d(out_dim)

#         self.lin = torch.nn.Linear(out_dim, 2)  # Binary classification (0 or 1)
#         self.dropout = nn.Dropout(p=dropout_rate)

#     def forward(self, x, edge_index, edge_type=None):
#         x1 = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_type)))
#         x1 = self.dropout(x1)

#         x2 = F.relu(self.batch_norm2(self.conv2(x1, edge_index, edge_type)))
#         x2 = self.dropout(x2) + x1  # Residual connection

#         x3 = F.relu(self.batch_norm3(self.conv3(x2, edge_index, edge_type)))
#         x3 = self.dropout(x3) + x2  # Residual connection

#         x4 = self.batch_norm4(self.conv4(x3, edge_index, edge_type))
#         return self.lin(x4)


# # Define the GNN Trainer Class
# class GNNTrainer:
#     def __init__(self, hetero_data, node_type, hidden_dim=128, out_dim=64, learning_rate=0.005, dropout_rate=0.4):
#         self.hetero_data = hetero_data
#         self.node_type = node_type

#         # Define model path
#         self.model_dir = "./saved_models"
#         self.model_path = f"{self.model_dir}/{node_type}_gnn.pth"

#         # Ensure directory exists
#         os.makedirs(self.model_dir, exist_ok=True)

#         num_relations = len(hetero_data.edge_types)
#         self.model = FastRGCNGNN(hetero_data, node_type, num_relations, hidden_dim, out_dim, dropout_rate)

#         # Use AdamW with decoupled weight decay
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)

#         # Learning rate scheduler with plateau detection
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

#         # Precision-focused loss with weighted classes
#         self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]))

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

#     def train(self, epochs=30):
#         """Train the GNN model and save it."""
#         best_f1 = 0.0
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

#             # Update scheduler based on validation performance
#             self.scheduler.step(f1)

#             if f1 > best_f1:
#                 best_f1 = f1
#                 self.save_model()

#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

#         print(f"Best F1 Score Achieved: {best_f1:.4f}")

#     def evaluate_test_set(self):
#         """Evaluates the model on the test set and prints predictions and metrics."""
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
#             preds = out.argmax(dim=1)
#             test_mask = self.hetero_data[self.node_type].test_mask
#             y_true = self.hetero_data[self.node_type].y[test_mask].cpu().numpy()
#             y_pred = preds[test_mask].cpu().numpy()

#             accuracy = accuracy_score(y_true, y_pred)
#             f1 = f1_score(y_true, y_pred, average='weighted')
#             precision = precision_score(y_true, y_pred, average='weighted')
#             recall = recall_score(y_true, y_pred, average='weighted')
#             cm = confusion_matrix(y_true, y_pred)

#             print("\n[TEST SET PREDICTIONS]")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1 Score: {f1:.4f}")
#             print("Confusion Matrix:")
#             print(cm)

#             return y_pred, y_true, cm, f1


import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import os


class FastRGCNGNN(torch.nn.Module):
    def __init__(self, hetero_data, node_type, num_relations, hidden_dim=32, out_dim=64, dropout_rate=0.4):
        super().__init__()

        # Dynamically determine input feature dimension
        in_channels = hetero_data[node_type].x.shape[1]
        print(f"[DEBUG] Detected in_channels for '{node_type}': {in_channels}")

        # Define a 4-layer GNN with BatchNorm and Dropout
        self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
        self.batch_norm1 = BatchNorm1d(hidden_dim)

        self.conv2 = FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.batch_norm2 = BatchNorm1d(hidden_dim)

        self.conv3 = FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.batch_norm3 = BatchNorm1d(hidden_dim)

        self.conv4 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)
        self.batch_norm4 = BatchNorm1d(out_dim)

        self.lin = nn.Linear(out_dim, 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.batch_norm1(self.conv1(x, edge_index, edge_type)))
        x = self.dropout(x)

        x = F.relu(self.batch_norm2(self.conv2(x, edge_index, edge_type)))
        x = self.dropout(x)

        x = F.relu(self.batch_norm3(self.conv3(x, edge_index, edge_type)))
        x = self.dropout(x)

        x = self.batch_norm4(self.conv4(x, edge_index, edge_type))
        return self.lin(x)


class GNNTrainer:
    def __init__(self, hetero_data, node_type, hidden_dim=32, out_dim=64, learning_rate=0.01, dropout_rate=0.4):
        self.hetero_data = hetero_data
        self.node_type = node_type

        num_relations = len(hetero_data.edge_types)
        self.model = FastRGCNGNN(hetero_data, node_type, num_relations, hidden_dim, out_dim, dropout_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.positive_nodes = []
        self.negative_nodes = []

    def save_model(self, path="./saved_models/gnn_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to {path}")

    def load_model(self, path="./saved_models/gnn_model.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print(f"[INFO] Model loaded from {path}")
        else:
            print(f"[WARNING] No model found at {path}")

    def train(self, epochs=25):
        best_f1 = 0

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
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
            loss = self.criterion(out[self.hetero_data[self.node_type].train_mask],
                                  self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask])
            loss.backward()
            self.optimizer.step()

            preds = out.argmax(dim=1)
            true_labels = self.hetero_data[self.node_type].y[self.hetero_data[self.node_type].train_mask].cpu().numpy()
            pred_labels = preds[self.hetero_data[self.node_type].train_mask].cpu().numpy()

            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            self.scheduler.step(f1)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

            if f1 > best_f1:
                best_f1 = f1

        print(f"[INFO] Training complete. Best F1 Score: {best_f1:.4f}")

    def evaluate_test_set(self):
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
            preds = out.argmax(dim=1)

            test_mask = self.hetero_data[self.node_type].test_mask
            y_true = self.hetero_data[self.node_type].y[test_mask].cpu().numpy()
            y_pred = preds[test_mask].cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            cm = confusion_matrix(y_true, y_pred)

            print("\n[TEST SET PREDICTIONS]")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)

            return y_pred, y_true, cm, f1
