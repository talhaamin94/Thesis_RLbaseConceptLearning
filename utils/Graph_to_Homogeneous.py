import torch
import torch.nn.functional as F
from torch_geometric.data import Data

class Graph_to_Homogeneous:
    def __init__(self, hetero_data):
        """
        Converts heterogeneous graph data to a homogeneous representation 
        while preserving mappings for nodes and edges.

        Parameters:
        - hetero_data (HeteroData): The heterogeneous graph object.
        """
        self.hetero_data = hetero_data
        self.node_mappings = {}  # {homogeneous_index: (node_type, instance_id)}
        self.edge_mappings = {}  # {homogeneous_edge_id: (src_type, relation, dst_type)}
        self.homo_data = self._convert_to_homogeneous()

    def _convert_to_homogeneous(self):
        """
        Converts the heterogeneous graph into a homogeneous graph while keeping 
        edges directed and ensuring consistency in indexing.
        """

        # Step 1: Convert using PyG's `to_homogeneous()`
        homo_data = self.hetero_data.to_homogeneous(add_node_type=True, add_edge_type=True)

        # Extract node type names from hetero_data
        node_type_names = list(self.hetero_data.node_types)  # Correct way to get node type names

        # Step 2: Store Node Mappings for Interpretation (Fixing Index Continuation)
        type_counters = {node_type: 0 for node_type in node_type_names}  # Track index per node type
        self.node_mappings = {}

        for homo_idx, node_type_idx in enumerate(homo_data.node_type):
            node_type = node_type_names[node_type_idx.item()]  # Get node type name
            original_idx = type_counters[node_type]  # Get correct original index
            type_counters[node_type] += 1  # Increment per type

            self.node_mappings[homo_idx] = (node_type, original_idx)  # Store mapping correctly

        # Step 3: Store Edge Mappings for Interpretation
        edge_type_names = list(self.hetero_data.edge_types)  # Extract edge type names
        self.edge_mappings = {
            i: edge_type_names[edge_type_idx.item()]  # Map edge type indices correctly
            for i, edge_type_idx in enumerate(homo_data.edge_type)
        }

        # Step 4: Ensure Feature Consistency
        feature_dims = {self.hetero_data[node_type].x.shape[1] for node_type in self.hetero_data.node_types}
        
        if len(feature_dims) > 1:
            # print(f"[WARNING] Node features have different dimensions: {feature_dims}")
            # print("[INFO] Proceeding with padding to make dimensions consistent.")
            
            # Determine the maximum feature dimension
            max_feature_dim = max(feature_dims)

            # Pad features to ensure uniform shape
            node_features_list = []
            for node_type in self.hetero_data.node_types:
                features = self.hetero_data[node_type].x
                feature_dim = features.shape[1]

                if feature_dim < max_feature_dim:
                    padding = torch.zeros((features.shape[0], max_feature_dim - feature_dim), dtype=torch.float)
                    features = torch.cat([features, padding], dim=1)

                node_features_list.append(features)

            # Concatenate all node features
            node_features = torch.cat(node_features_list, dim=0)
        else:
            # print("[INFO] Node feature dimensions are already consistent. Skipping padding.")
            node_features = homo_data.x  # Use PyG's standardized features

        # Step 5: Add One-Hot Encoding for Node Type Only If Needed
        if len(feature_dims) > 1:  # Only add node_type encoding if dimensions were inconsistent
            node_type_one_hot = F.one_hot(homo_data.node_type, num_classes=len(self.hetero_data.node_types)).float()
            final_node_features = torch.cat([node_features, node_type_one_hot], dim=1)
        else:
            final_node_features = node_features  # No need to add extra encoding

        # Step 6: Merge Train & Test Masks
        train_mask = torch.zeros(homo_data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(homo_data.num_nodes, dtype=torch.bool)
        labels = torch.full((homo_data.num_nodes,), -1, dtype=torch.long)

        for node_type in self.hetero_data.node_types:
            node_indices = (homo_data.node_type == node_type_names.index(node_type)).nonzero(as_tuple=True)[0]
            
            if "train_mask" in self.hetero_data[node_type]:
                train_mask[node_indices] = self.hetero_data[node_type].train_mask
            if "test_mask" in self.hetero_data[node_type]:
                test_mask[node_indices] = self.hetero_data[node_type].test_mask
            if "y" in self.hetero_data[node_type]:
                labels[node_indices] = self.hetero_data[node_type].y

        # Step 7: Create Homogeneous Data Object with Directed Edges
        homo_data.x = final_node_features
        homo_data.y = labels
        homo_data.train_mask = train_mask
        homo_data.test_mask = test_mask

        # print("[INFO] Homogeneous Graph Conversion Complete (Directed Edges Preserved)")
        # print(f"Total Nodes: {homo_data.num_nodes}, Total Edges: {homo_data.edge_index.shape[1]}")
        # print(f"Node Features Shape: {homo_data.x.shape}, Edge Type Count: {len(self.edge_mappings)}")

        return homo_data

    def get_homogeneous_data(self):
        """Returns the converted homogeneous graph."""
        return self.homo_data

    def get_node_mapping(self):
        """Returns the mapping of homogeneous nodes to original node types and instance IDs."""
        return self.node_mappings

    def get_edge_mapping(self):
        """Returns the mapping of homogeneous edge types to original heterogeneous relations."""
        return self.edge_mappings
