# import torch
# from torch_geometric.data import HeteroData
# from rdflib import Graph, RDF
# from collections import defaultdict
# import numpy as np

# # Load RDF graph
# g = Graph()
# g.parse("data/aifb/raw/aifb_stripped.nt", format="nt")

# # Step 1: Extract All Nodes and Assign Types
# nodes_by_type = defaultdict(set)
# node_id_map = {}  # Maps each URI to a unique node index
# node_type_map = {}  # Maps each URI to its detected node type
# node_predicates = defaultdict(set)  # Track predicates per node type

# current_node_id = 0

# for subject, predicate, obj in g.triples((None, RDF.type, None)):  
#     node_type = str(obj).split("#")[-1]  # Extract last part after "#"
#     if node_type:
#         if subject not in node_id_map:
#             node_id_map[subject] = current_node_id
#             node_type_map[subject] = node_type
#             current_node_id += 1
#         nodes_by_type[node_type].add(subject)

# # Step 2: Collect All Possible Features Per Node Type
# node_features = defaultdict(lambda: defaultdict(dict))  # Stores features per node
# predicate_index_map = defaultdict(dict)  # Maps predicates to fixed feature index per node type

# for subject, predicate, obj in g:
#     if subject in node_id_map and not predicate.startswith(str(RDF)):  # Ignore RDF metadata
#         predicate_name = str(predicate).split("#")[-1] if "#" in str(predicate) else str(predicate).split("/")[-1]
#         node_type = node_type_map[subject]

#         # Ensure predicate has a fixed index for ordering
#         if predicate_name not in predicate_index_map[node_type]:
#             predicate_index_map[node_type][predicate_name] = len(predicate_index_map[node_type])

#         try:
#             feature_value = float(obj)  # Convert to numeric if possible
#         except ValueError:
#             feature_value = hash(obj) % 10000  # Hash categorical values
        
#         # Store feature at the correct index
#         node_features[node_type][subject][predicate_name] = feature_value
#         node_predicates[node_type].add(predicate_name)  # Track unique predicates per node type

# # Step 3: Ensure Features Are Correctly Aligned
# hetero_data = HeteroData()

# for node_type, nodes in nodes_by_type.items():
#     # Determine the final feature length for this node type
#     num_features = len(predicate_index_map[node_type])

#     feature_matrix = []
#     for node in nodes:
#         feature_vector = [0.0] * num_features  # Initialize with zeros
        
#         for predicate, index in predicate_index_map[node_type].items():
#             if predicate in node_features[node_type][node]:  # Assign value if available
#                 feature_vector[index] = node_features[node_type][node][predicate]
        
#         feature_matrix.append(feature_vector)

#     hetero_data[node_type].x = torch.tensor(feature_matrix, dtype=torch.float)

# # Step 4: Extract All Edges
# edge_types = defaultdict(lambda: ([], []))

# for subject, predicate, obj in g:
#     if subject in node_id_map and obj in node_id_map:
#         subject_id = node_id_map[subject]
#         object_id = node_id_map[obj]
#         source_type = node_type_map[subject]
#         target_type = node_type_map[obj]

#         relation = str(predicate).split("#")[-1] if "#" in str(predicate) else str(predicate).split("/")[-1]
#         edge_type = (source_type, relation, target_type)

#         edge_types[edge_type][0].append(subject_id)
#         edge_types[edge_type][1].append(object_id)

# # Add edges to HeteroData
# for edge_type, (src, dst) in edge_types.items():
#     hetero_data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)

# # Step 5: Compute Statistics Before Filtering
# total_nodes_before = sum(len(nodes) for nodes in nodes_by_type.values())
# total_edge_types_before = len(edge_types)
# total_edges_before = sum(len(edges[0]) for edges in edge_types.values())

# # Step 6: Filter to Keep Only the Relevant Concept Classes
# relevant_concepts = {
#     "Person", "Publication", "InCollection", "InProceedings", "Proceedings",
#     "Misc", "TechnicalReport", "Article", "Book", "Organization", "ResearchTopic", "Project"
# }

# filtered_nodes_by_type = {k: v for k, v in nodes_by_type.items() if k in relevant_concepts}
# filtered_edge_types = {k: v for k, v in edge_types.items() if k[0] in relevant_concepts and k[2] in relevant_concepts}

# # Create a filtered HeteroData object
# filtered_hetero_data = HeteroData()

# for node_type, nodes in filtered_nodes_by_type.items():
#     num_features = len(predicate_index_map[node_type])

#     feature_matrix = []
#     for node in nodes:
#         feature_vector = [0.0] * num_features  # Initialize with zeros
        
#         for predicate, index in predicate_index_map[node_type].items():
#             if predicate in node_features[node_type][node]:  # Assign value if available
#                 feature_vector[index] = node_features[node_type][node][predicate]
        
#         feature_matrix.append(feature_vector)

#     filtered_hetero_data[node_type].x = torch.tensor(feature_matrix, dtype=torch.float)

# for edge_type, (src, dst) in filtered_edge_types.items():
#     filtered_hetero_data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)

# # Step 7: Compute Statistics After Filtering
# total_nodes_after = sum(len(nodes) for nodes in filtered_nodes_by_type.values())
# total_edge_types_after = len(filtered_edge_types)
# total_edges_after = sum(len(edges[0]) for edges in filtered_edge_types.values())

# # Step 8: Prepare Summary of Statistics
# stats_summary = {
#     "Before Filtering": {
#         "Total Node Types": len(nodes_by_type),
#         "Total Nodes": total_nodes_before,
#         "Total Edge Types": total_edge_types_before,
#         "Total Edges": total_edges_before
#     },
#     "After Filtering": {
#         "Total Node Types": len(filtered_nodes_by_type),
#         "Total Nodes": total_nodes_after,
#         "Total Edge Types": total_edge_types_after,
#         "Total Edges": total_edges_after
#     }
# }

# # Final Output
# print(stats_summary, filtered_hetero_data)

# for node_type in filtered_hetero_data.node_types:
#     print(node_type, filtered_hetero_data[node_type].x.shape)
import torch
from torch_geometric.data import HeteroData
from rdflib import Graph, RDF
from collections import defaultdict

class RDFGraphConverter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.hetero_data = HeteroData()
        self.filtered_hetero_data = HeteroData()
        self.node_id_map = {}  # Maps URI to a unique node index
        self.node_type_map = {}  # Maps URI to its node type
        self.node_features = defaultdict(lambda: defaultdict(dict))  # Stores node features
        self.nodes_by_type = defaultdict(set)  # Store nodes by type
        self.edge_types = defaultdict(lambda: ([], []))  # Store edges
        self.predicate_index_map = defaultdict(dict)  # Fixed predicate order per node type

        # Define relevant concept classes
        self.relevant_concepts = {
            "Person", "Publication", "InCollection", "InProceedings", "Proceedings",
            "Misc", "TechnicalReport", "Article", "Book", "Organization", "ResearchTopic", "Project"
        }

    def load_dataset(self):
        """Loads the dataset based on the selected name."""
        if self.dataset_name == "AIFB":
            return self._load_aifb()
        elif self.dataset_name == "MUTAG":
            return self._load_mutag()
        else:
            raise ValueError("Unsupported dataset. Choose 'AIFB' or 'MUTAG'.")

    def _load_aifb(self):
        """Loads and processes the AIFB dataset."""
        g = Graph()
        g.parse("data/aifb/raw/aifb_stripped.nt", format="nt")
        return self._process_graph(g)

    def _load_mutag(self):
        """Loads and processes the MUTAG dataset."""
        g = Graph()
        g.parse("data/mutag/raw/mutag_stripped.nt", format="nt")
        return self._process_graph(g)

    def _process_graph(self, g):
        """Processes the RDF graph and converts it into a filtered HeteroData object."""
        self._extract_nodes(g)
        # print("Detected Node Types Before Filtering:", list(self.nodes_by_type.keys()))  # Debugging step
        self._extract_features(g)
        self._assign_features()
        self._extract_edges(g)  # Now properly included
        self._filter_data()
        # print("Node Types After Filtering:", list(self.filtered_hetero_data.node_types))  # Debugging step
        return self.filtered_hetero_data

    def _extract_nodes(self, g):
        """Extracts nodes and assigns them unique IDs."""
        current_node_id = 0
        for subject, predicate, obj in g.triples((None, RDF.type, None)):  
            node_type = str(obj).split("#")[-1]  
            if node_type:
                if subject not in self.node_id_map:
                    self.node_id_map[subject] = current_node_id
                    self.node_type_map[subject] = node_type
                    current_node_id += 1
                self.nodes_by_type[node_type].add(subject)

    def _extract_features(self, g):
        """Extracts all unique features per node type and assigns them a fixed order."""
        for subject, predicate, obj in g:
            if subject in self.node_id_map and not predicate.startswith(str(RDF)):
                predicate_name = str(predicate).split("#")[-1] if "#" in str(predicate) else str(predicate).split("/")[-1]
                node_type = self.node_type_map[subject]
                if predicate_name not in self.predicate_index_map[node_type]:
                    self.predicate_index_map[node_type][predicate_name] = len(self.predicate_index_map[node_type])
                try:
                    feature_value = float(obj)  
                except ValueError:
                    feature_value = hash(obj) % 10000  
                self.node_features[node_type][subject][predicate_name] = feature_value

    def _assign_features(self):
        """Ensures feature alignment and assigns them to the HeteroData object."""
        for node_type, nodes in self.nodes_by_type.items():
            num_features = len(self.predicate_index_map[node_type])
            feature_matrix = []
            for node in nodes:
                feature_vector = [0.0] * num_features  
                for predicate, index in self.predicate_index_map[node_type].items():
                    if predicate in self.node_features[node_type][node]:  
                        feature_vector[index] = self.node_features[node_type][node][predicate]
                feature_matrix.append(feature_vector)
            self.hetero_data[node_type].x = torch.tensor(feature_matrix, dtype=torch.float)

    def _extract_edges(self, g):
        """Extracts all edges from the RDF graph."""
        for subject, predicate, obj in g:
            if subject in self.node_id_map and obj in self.node_id_map:
                subject_id = self.node_id_map[subject]
                object_id = self.node_id_map[obj]
                source_type = self.node_type_map[subject]
                target_type = self.node_type_map[obj]
                relation = str(predicate).split("#")[-1] if "#" in str(predicate) else str(predicate).split("/")[-1]
                edge_type = (source_type, relation, target_type)
                self.edge_types[edge_type][0].append(subject_id)
                self.edge_types[edge_type][1].append(object_id)

        for edge_type, (src, dst) in self.edge_types.items():
            self.hetero_data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)

    def _filter_data(self):
        """Filters the dataset to keep only relevant node types."""
        filtered_nodes_by_type = {k: v for k, v in self.nodes_by_type.items() if k in self.relevant_concepts}
        if not filtered_nodes_by_type:
            print("Warning: No nodes matched the filtering criteria. Check relevant_concepts.")

        filtered_edge_types = {k: v for k, v in self.edge_types.items() if k[0] in self.relevant_concepts and k[2] in self.relevant_concepts}

        for node_type, nodes in filtered_nodes_by_type.items():
            num_features = len(self.predicate_index_map[node_type])
            feature_matrix = []
            for node in nodes:
                feature_vector = [0.0] * num_features  
                for predicate, index in self.predicate_index_map[node_type].items():
                    if predicate in self.node_features[node_type][node]:  
                        feature_vector[index] = self.node_features[node_type][node][predicate]
                feature_matrix.append(feature_vector)
            self.filtered_hetero_data[node_type].x = torch.tensor(feature_matrix, dtype=torch.float)

        for edge_type, (src, dst) in filtered_edge_types.items():
            self.filtered_hetero_data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)

    def get_statistics(self):
        """Computes and returns dataset statistics before and after filtering."""
        
        # Total node types before filtering
        total_node_types_before = len(self.nodes_by_type)
        total_nodes_before = sum(len(nodes) for nodes in self.nodes_by_type.values())
        
        # Total edge types and edges before filtering
        total_edge_types_before = len(self.edge_types)
        total_edges_before = sum(len(edges[0]) for edges in self.edge_types.values())

        # Total node types after filtering
        filtered_nodes_by_type = {k: v for k, v in self.nodes_by_type.items() if k in self.relevant_concepts}
        total_node_types_after = len(filtered_nodes_by_type)
        total_nodes_after = sum(len(nodes) for nodes in filtered_nodes_by_type.values())

        # Total edge types and edges after filtering
        filtered_edge_types = {k: v for k, v in self.edge_types.items() if k[0] in self.relevant_concepts and k[2] in self.relevant_concepts}
        total_edge_types_after = len(filtered_edge_types)
        total_edges_after = sum(len(edges[0]) for k, edges in self.edge_types.items() if k[0] in self.relevant_concepts and k[2] in self.relevant_concepts)

        return {
            "Before Filtering": {
                "Total Node Types": total_node_types_before,
                "Total Nodes": total_nodes_before,
                "Total Edge Types": total_edge_types_before,
                "Total Edges": total_edges_before
            },
            "After Filtering": {
                "Total Node Types": total_node_types_after,
                "Total Nodes": total_nodes_after,
                "Total Edge Types": total_edge_types_after,
                "Total Edges": total_edges_after
            }
        }


converter = RDFGraphConverter("AIFB")
hetero_data = converter.load_dataset()
print(converter.get_statistics())
