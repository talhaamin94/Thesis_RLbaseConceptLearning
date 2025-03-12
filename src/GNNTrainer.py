# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.nn.functional as F
# # from torch_geometric.nn import FastRGCNConv
# # from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
# # import os
# # from Graph_to_Homogeneous import Graph_to_Homogeneous

# # class FastRGCNGNN(nn.Module):
# #     def __init__(self, homo_data, hidden_dim=64, out_dim=128, dropout_rate=0.3):
# #         super().__init__()

# #         in_channels = homo_data.x.shape[1]
# #         num_relations = len(homo_data.edge_type.unique())

# #         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations)
# #         self.conv2 = FastRGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
# #         self.conv3 = FastRGCNConv(hidden_dim, out_dim, num_relations=num_relations)

# #         self.dropout = nn.Dropout(p=dropout_rate)
# #         self.lin = nn.Linear(out_dim, 2)  # Binary classification

# #         # Initialize weights properly
# #         self._reset_parameters()

# #     def _reset_parameters(self):
# #         for param in self.parameters():
# #             if param.dim() > 1:
# #                 torch.nn.init.xavier_uniform_(param)

# #     def forward(self, x, edge_index, edge_type):
# #         x = F.relu(self.conv1(x, edge_index, edge_type))
# #         x = self.dropout(x)

# #         x = F.relu(self.conv2(x, edge_index, edge_type))
# #         x = self.dropout(x)

# #         x = self.conv3(x, edge_index, edge_type)
        
# #         return self.lin(x)


# # class GNNTrainer:
# #     def __init__(self, hetero_data, node_type, hidden_dim=64, out_dim=128, learning_rate=0.005, dropout_rate=0.3):
# #         self.hetero_data = hetero_data
# #         self.node_type = node_type

# #         # Convert heterogeneous data to homogeneous
# #         self.graph_converter = Graph_to_Homogeneous(self.hetero_data)
# #         self.homo_data = self.graph_converter.get_homogeneous_data()
# #         self.node_mappings = self.graph_converter.get_node_mapping()

# #         self.model = FastRGCNGNN(self.homo_data, hidden_dim, out_dim, dropout_rate)
# #         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
# #         self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=25)
# #         self.criterion = nn.CrossEntropyLoss()

# #     def train(self, epochs=25):
# #         best_f1 = 0

# #         for epoch in range(epochs):
# #             self.model.train()
# #             self.optimizer.zero_grad()

# #             x = self.homo_data.x
# #             edge_index = self.homo_data.edge_index
# #             edge_type = self.homo_data.edge_type
            
# #             out = self.model(x, edge_index, edge_type)
# #             loss = self.criterion(out[self.homo_data.train_mask], self.homo_data.y[self.homo_data.train_mask])
            
# #             loss.backward()
# #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient Clipping
# #             self.optimizer.step()
# #             self.scheduler.step()

# #             preds = out.argmax(dim=1)
# #             true_labels = self.homo_data.y[self.homo_data.train_mask].cpu().numpy()
# #             pred_labels = preds[self.homo_data.train_mask].cpu().numpy()

# #             accuracy = accuracy_score(true_labels, pred_labels)
# #             f1 = f1_score(true_labels, pred_labels, average='weighted')

# #             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

# #             if f1 > best_f1:
# #                 best_f1 = f1

# #         print(f"[INFO] Training complete. Best F1 Score: {best_f1:.4f}")

# #     def evaluate_test_set(self):
# #         self.model.eval()
# #         with torch.no_grad():
# #             x = self.homo_data.x
# #             edge_index = self.homo_data.edge_index
# #             edge_type = self.homo_data.edge_type
            
# #             out = self.model(x, edge_index, edge_type)
# #             preds = out.argmax(dim=1)

# #             test_mask = self.homo_data.test_mask
# #             y_true = self.homo_data.y[test_mask].cpu().numpy()
# #             y_pred = preds[test_mask].cpu().numpy()

# #             accuracy = accuracy_score(y_true, y_pred)
# #             f1 = f1_score(y_true, y_pred, average='weighted')
# #             precision = precision_score(y_true, y_pred, average='weighted')
# #             recall = recall_score(y_true, y_pred, average='weighted')
# #             cm = confusion_matrix(y_true, y_pred)

# #             print("\n[TEST SET RESULTS]")
# #             print(f"Accuracy: {accuracy:.4f}")
# #             print(f"Precision: {precision:.4f}")
# #             print(f"Recall: {recall:.4f}")
# #             print(f"F1 Score: {f1:.4f}")
# #             print("Confusion Matrix:")
# #             print(cm)

# #             return y_pred, y_true, cm, f1


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import FastRGCNConv
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
# import os
# from Graph_to_Homogeneous import Graph_to_Homogeneous

# class FastRGCNGNN(nn.Module):
#     def __init__(self, homo_data, hidden_dim=16, num_bases=30):
#         super().__init__()

#         in_channels = homo_data.x.shape[1]
#         num_relations = len(homo_data.edge_type.unique())
#         num_classes = int(homo_data.y.max().item()) + 1  # Determine number of classes dynamically

#         self.conv1 = FastRGCNConv(in_channels, hidden_dim, num_relations=num_relations, num_bases=num_bases)
#         self.conv2 = FastRGCNConv(hidden_dim, num_classes, num_relations=num_relations, num_bases=num_bases)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         """Better weight initialization"""
#         for param in self.parameters():
#             if param.dim() > 1:
#                 torch.nn.init.xavier_uniform_(param)

#     def forward(self, x, edge_index, edge_type):
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.conv2(x, edge_index, edge_type)
#         return F.log_softmax(x, dim=1)  # Ensuring compatibility with nll_loss


# # Modify the train function to print epochs in the required format

# class GNNTrainer:
#     def __init__(self, hetero_data, node_type, hidden_dim=16, learning_rate=0.01):
#         self.hetero_data = hetero_data
#         self.node_type = node_type

#         self.num_bases = max(10, len(hetero_data.edge_types) // 2)

#         print(f"Number of Bases: {self.num_bases}")

#         # Convert heterogeneous data to homogeneous
#         self.graph_converter = Graph_to_Homogeneous(self.hetero_data)
#         self.homo_data = self.graph_converter.get_homogeneous_data()
#         self.node_mappings = self.graph_converter.get_node_mapping()

#         self.model = FastRGCNGNN(self.homo_data, hidden_dim, self.num_bases)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
#         self.criterion = nn.NLLLoss()

#     def train(self, epochs=50):
#         best_f1 = 0

#         for epoch in range(1, epochs + 1):
#             self.model.train()
#             self.optimizer.zero_grad()

#             x = self.homo_data.x  # Explicitly using node features
#             edge_index = self.homo_data.edge_index
#             edge_type = self.homo_data.edge_type
            
#             out = self.model(x, edge_index, edge_type)
#             loss = self.criterion(out[self.homo_data.train_mask], self.homo_data.y[self.homo_data.train_mask])
            
#             loss.backward()
#             self.optimizer.step()

#             preds = out.argmax(dim=1)
#             true_labels = self.homo_data.y[self.homo_data.train_mask].cpu().numpy()
#             pred_labels = preds[self.homo_data.train_mask].cpu().numpy()

#             accuracy = accuracy_score(true_labels, pred_labels)
#             f1 = f1_score(true_labels, pred_labels, average='weighted')

#             print(f"Epoch: {epoch}, Loss: {loss:.4f}, Train: {accuracy:.4f}, Test: {f1:.4f}")

#             if f1 > best_f1:
#                 best_f1 = f1

#         print(f"Training complete. Best F1 Score: {best_f1:.4f}")

#     def evaluate_test_set(self):
#         self.model.eval()
#         with torch.no_grad():
#             x = self.homo_data.x
#             edge_index = self.homo_data.edge_index
#             edge_type = self.homo_data.edge_type
            
#             out = self.model(x, edge_index, edge_type)
#             preds = out.argmax(dim=1)

#             test_mask = self.homo_data.test_mask
#             y_true = self.homo_data.y[test_mask].cpu().numpy()
#             y_pred = preds[test_mask].cpu().numpy()

#             accuracy = accuracy_score(y_true, y_pred)
#             f1 = f1_score(y_true, y_pred, average='weighted')
#             precision = precision_score(y_true, y_pred, average='weighted')
#             recall = recall_score(y_true, y_pred, average='weighted')
#             cm = confusion_matrix(y_true, y_pred)

#             print("\n[TEST SET RESULTS]")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1 Score: {f1:.4f}")
#             print("Confusion Matrix:")
#             print(cm)

#             return y_pred, y_true, cm, f1


# # Example usage
# # hetero_data = Load your heterogeneous data here
# # gnn_trainer = GNNTrainer(hetero_data, "Person")
# # gnn_trainer.train(epochs=50)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
from sklearn.metrics import accuracy_score, f1_score
import time
from Graph_to_Homogeneous import Graph_to_Homogeneous

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

class GNNTrainer:
    def __init__(self, hetero_data, node_type, hidden_dim=16, learning_rate=0.005, dropout_rate=0.3):
        self.hetero_data = hetero_data
        self.node_type = node_type

        self.num_bases = max(10, len(hetero_data.edge_types) // 2)

        # print(f"Number of Bases: {self.num_bases}")

        # Convert heterogeneous data to homogeneous
        self.graph_converter = Graph_to_Homogeneous(self.hetero_data)
        self.homo_data = self.graph_converter.get_homogeneous_data()
        self.node_mappings = self.graph_converter.get_node_mapping()
        # print(self.node_mappings)
        self.model = FastRGCNGNN(self.homo_data, hidden_dim, self.num_bases, dropout_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.criterion = nn.NLLLoss()
    # Path to save the best model
        self.best_model_path = "best_model.pth"
        self.best_test_f1 = 0 

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

        for epoch in range(1, epochs + 1):
            start = time.time()
            loss = self.train()
            train_acc, test_acc, train_f1, test_f1 = self.test()

            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
                  f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
            times.append(time.time() - start)

            # Save model if test F1 score improves
            if test_f1 > self.best_test_f1:
                self.best_test_f1 = test_f1
                torch.save(self.model.state_dict(), self.best_model_path)  # Save best model
                print(f" Best model saved at epoch {epoch} with Test F1: {test_f1:.4f}")

        print(f"Training complete. Median time per epoch: {torch.tensor(times).median():.4f}s")

        # Load the best model at the end
        print(f" Loading best model from {self.best_model_path}")
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))

    def get_positive_nodes(self):
        """Retrieve positive nodes (label = 1) in terms of hetero_data indexes."""
        positive_nodes = []
        for homo_index, (node_type, instance_id) in self.node_mappings.items():
            if node_type == self.node_type and self.hetero_data[node_type].y[instance_id].item() == 1:
                positive_nodes.append(instance_id)  # Store original hetero_data index
        return positive_nodes

    def get_negative_nodes(self):
        """Retrieve negative nodes (label = 0) in terms of hetero_data indexes."""
        negative_nodes = []
        for homo_index, (node_type, instance_id) in self.node_mappings.items():
            if node_type == self.node_type and self.hetero_data[node_type].y[instance_id].item() == 0:
                negative_nodes.append(instance_id)  # Store original hetero_data index
        return negative_nodes