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
import os
import torch
from torch_geometric.data import HeteroData
from rdflib import Graph, RDF
from collections import defaultdict
from TransformLabels import TransformLabels

class RDFGraphConverter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        self.hetero_data = HeteroData()
        self.filtered_hetero_data = HeteroData()
        self.node_id_map = {}  # Maps URI to a unique node index
        self.node_type_map = {}  # Maps URI to its node type
        self.nodes_by_type = defaultdict(set)
        self.edge_types = defaultdict(lambda: ([], []))
        self.predicate_index_map = defaultdict(dict)
        self.node_features = defaultdict(lambda: defaultdict(dict))

        # Assign relevant classes dynamically based on dataset
        self.relevant_classes = self._get_default_relevant_classes()

    def _get_default_relevant_classes(self):
        """Returns default relevant concepts based on the dataset."""
        if self.dataset_name == "aifb":
            return {
                "Person", "Publication", "InCollection", "InProceedings", "Proceedings",
                "Misc", "TechnicalReport", "Article", "Book", "Organization", "ResearchTopic", "Project"
            }
        elif self.dataset_name == "mutag":
            return {"MutagenicCompound", "Atom", "Bond", "Molecule"}
        else:
            return set()  # Empty set for unknown datasets

    def load_dataset(self):
        """Loads the dataset, processes it, and saves the filtered data."""
        if self.dataset_name == "aifb":
            g = Graph()
            g.parse("data/aifb/raw/aifb_stripped.nt", format="nt")
        elif self.dataset_name == "mutag":
            g = Graph()
            g.parse("data/mutag/raw/mutag_stripped.nt", format="nt")
        else:
            raise ValueError("Unsupported dataset. Choose 'AIFB' or 'MUTAG'.")

        # Process RDF graph
        self._extract_nodes(g)
        self._extract_features(g)
        self._assign_features()
        self._extract_edges(g)
        self._filter_data()

        # Assign labels to Person nodes
        self._assign_labels()
        
        # Save processed data
        self._save_processed_data()

        return self.filtered_hetero_data

    def _extract_nodes(self, g):
        """Extracts nodes, assigns them unique IDs per node type, and stores a detailed mapping."""
        self.node_mapping = {}  # Maps (node_type, instance_name) → node index, full URI
        self.node_type_counters = defaultdict(int)  # Separate counters for each node type

        for subject, predicate, obj in g.triples((None, RDF.type, None)):  
            node_type = str(obj).split("#")[-1]  # Extract class name

            if node_type:
                if subject not in self.node_id_map:
                    # Assign a unique index for this node type
                    node_index = self.node_type_counters[node_type]
                    self.node_type_counters[node_type] += 1  # Increment counter for this type

                    self.node_id_map[subject] = node_index
                    self.node_type_map[subject] = node_type
                    self.nodes_by_type[node_type].add(subject)

                    # Extract both full URI and instance name
                    full_uri = str(subject)  # Full RDF URI
                    instance_name = full_uri.split("#")[-1] if "#" in full_uri else full_uri.split("/")[-1]

                    # Store in `node_mapping`
                    self.node_mapping[(node_type, instance_name)] = {
                        "node_index": node_index,  # Per-type indexing
                        "node_type": node_type,
                        "instance_name": instance_name,
                        "full_uri": full_uri
                    }

        print(f"\n[DEBUG] Extracted {sum(self.node_type_counters.values())} nodes across {len(self.node_type_counters)} types.")
        for node_type, count in self.node_type_counters.items():
            print(f"[DEBUG] {node_type}: {count} nodes")

        print("\n[DEBUG] Checking node_mapping consistency for 'Person' nodes:")
        i = 0
        for key, value in list(self.node_mapping.items()):
            if key[0] == "Person" and i < 10:  # Only check Person nodes
                print(f"[DEBUG] Mapping - Instance: {key[1]}, URI: {value['full_uri']}, Index: {value['node_index']}")
                i += 1





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
        """Filters data based on relevant classes, ensuring 'Person' is retained."""
        # print(f"\n[DEBUG] Relevant classes before filtering: {self.relevant_classes}")

        # # Ensure that Person is actually in relevant classes
        # if "Person" not in self.relevant_classes:
        #     print("\n[DEBUG] Warning! 'Person' is missing from relevant classes.")
        
        filtered_nodes_by_type = {
            k: v for k, v in self.nodes_by_type.items() if k in self.relevant_classes
        }

        # print(f"[DEBUG] Node types before filtering: {list(self.nodes_by_type.keys())}")
        # print(f"[DEBUG] Node types after filtering: {list(filtered_nodes_by_type.keys())}")

        # if "Person" in filtered_nodes_by_type:
        #     print(f"[DEBUG] 'Person' nodes after filtering: {len(filtered_nodes_by_type['Person'])}")
        # else:
        #     print("\n[DEBUG] No 'Person' nodes exist after filtering! They were removed!")

        # Assign filtered nodes to HeteroData
        for node_type, nodes in filtered_nodes_by_type.items():
            num_features = len(self.predicate_index_map.get(node_type, {}))
            feature_matrix = []
            for node in nodes:
                feature_vector = [0.0] * num_features
                for predicate, index in self.predicate_index_map.get(node_type, {}).items():
                    if predicate in self.node_features[node_type][node]:
                        feature_vector[index] = self.node_features[node_type][node][predicate]
                feature_matrix.append(feature_vector)

            self.filtered_hetero_data[node_type].x = torch.tensor(feature_matrix, dtype=torch.float)

        # Assign filtered edges
        for edge_type, (src, dst) in self.edge_types.items():
            self.filtered_hetero_data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)


    def _save_processed_data(self):
        """Saves the filtered dataset and labels in the respective dataset directory."""
        dataset_dir = f"data/{self.dataset_name}/processed"
        os.makedirs(dataset_dir, exist_ok=True)

        # Save filtered hetero data object
        torch.save(self.filtered_hetero_data, os.path.join(dataset_dir, "hetero_data.pt"))

        # Save transformed labels
        label_transformer = TransformLabels(self.dataset_name)
        train_file = f"data/{self.dataset_name}/raw/trainingSet.tsv"
        test_file = f"data/{self.dataset_name}/raw/testSet.tsv"

        train_output = os.path.join(dataset_dir, "trainingSet_updated.csv")
        test_output = os.path.join(dataset_dir, "testSet_updated.csv")

        # Call the function with correct arguments
        label_transformer.transform_and_save_labels(train_file, test_file, train_output, test_output)

    def _assign_labels(self):
        """Assigns labels from both training and test sets, using node_mapping for direct lookup."""
        train_file = f"data/{self.dataset_name}/raw/trainingSet.tsv"
        test_file = f"data/{self.dataset_name}/raw/testSet.tsv"

        label_transformer = TransformLabels(self.dataset_name)
        train_df, test_df = label_transformer.assign_labels(train_file, test_file)

        target_node_type = "Person" if self.dataset_name == "aifb" else "MutagenicCompound"
        target_nodes = list(self.nodes_by_type.get(target_node_type, []))

        labels = torch.zeros(len(target_nodes), dtype=torch.long)
        train_indices = []
        test_indices = []

        # print(f"[DEBUG] Sample instances from train_df: {train_df['person'].unique()[:10]}")
        # print(f"[DEBUG] Sample node_mapping keys: {list(self.node_mapping.keys())[:10]}")

        #  Assign labels using `node_mapping` for direct lookup
        for dataset, df, indices in [("train", train_df, train_indices), ("test", test_df, test_indices)]:
            for _, row in df.iterrows():
                raw_instance = str(row["person"]).strip() if "person" in row else str(row["compound"]).strip()

                # Extract instance name from full URI
                instance_name = raw_instance.split("#")[-1] if "#" in raw_instance else raw_instance.split("/")[-1]

                # Directly check in `node_mapping`
                if (target_node_type, instance_name) in self.node_mapping:
                    node_info = self.node_mapping[(target_node_type, instance_name)]
                    node_index = node_info["node_index"]

                    labels[node_index] = row["new_label"]
                    indices.append(node_index)
                else:
                    print(f"[DEBUG] Skipping '{instance_name}' - Not found in node_mapping for type '{target_node_type}'")

        

        # Assign labels to `HeteroData`
        self.filtered_hetero_data[target_node_type].y = labels
        self.train_indices = torch.tensor(train_indices, dtype=torch.long)
        self.test_indices = torch.tensor(test_indices, dtype=torch.long)

        # Save train/test masks
        self.filtered_hetero_data[target_node_type].train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        self.filtered_hetero_data[target_node_type].test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        self.filtered_hetero_data[target_node_type].train_mask[self.train_indices] = True
        self.filtered_hetero_data[target_node_type].test_mask[self.test_indices] = True

        # print(f"[DEBUG] Assigned labels: {labels.sum().item()} positive labels")
        # print(f"[DEBUG] Train nodes: {len(train_indices)}, Test nodes: {len(test_indices)}")

                # Print the first 10 assigned labels for the correct node type
        print(f"\n[DEBUG] Checking First 10 Assigned Labels for '{target_node_type}':")
        if target_node_type in self.filtered_hetero_data.node_types:
            for i in range(min(10, len(self.filtered_hetero_data[target_node_type].y))):
                print(f"[DEBUG] Index {i}: Label = {self.filtered_hetero_data[target_node_type].y[i].item()}")
        else:
            print(f"[DEBUG] ERROR: Node type '{target_node_type}' not found in filtered_hetero_data!")







    def get_statistics(self):
        """Computes and returns dataset statistics before and after filtering."""
        return {
            "Before Filtering": {
                "Total Node Types": len(self.nodes_by_type),
                "Total Nodes": sum(len(nodes) for nodes in self.nodes_by_type.values()),
                "Total Edge Types": len(self.edge_types),
                "Total Edges": sum(len(edges[0]) for edges in self.edge_types.values())
            },
            "After Filtering": {
                "Total Node Types": len(self.filtered_hetero_data.node_types),
                "Total Nodes": sum(len(self.nodes_by_type[node_type]) for node_type in self.filtered_hetero_data.node_types if node_type in self.nodes_by_type),
                "Total Edge Types": len(self.filtered_hetero_data.edge_types),
                "Total Edges": sum(len(self.filtered_hetero_data[edge_type].edge_index[0]) for edge_type in self.filtered_hetero_data.edge_types)
            }
        }


# Usage example
converter = RDFGraphConverter("AIFB")
hetero_data = converter.load_dataset()
print(converter.get_statistics())

print(hetero_data)
