import os
import torch
from torch_geometric.data import HeteroData
from rdflib import Graph, RDF
from collections import defaultdict
from TransformLabels import TransformLabels
from MiniDataset import MiniDataset
import pandas as pd

class RDFGraphConverter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        self.hetero_data = HeteroData()
        # self.filtered_hetero_data = HeteroData()
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
        elif self.dataset_name == "mini":
            return{"A","B","C"}
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
    
        elif self.dataset_name == "mini":
            # g = Graph()
            mini_generator = MiniDataset()
            g = mini_generator.get_graph()  # Load RDF graph from MiniDataset
            # g.parse("data/mini/raw/rdf_graph.nt", format="nt")
        else:
            raise ValueError("Unsupported dataset. Choose 'AIFB' or 'MUTAG'.")
       
        self.g = g
       
       
        # Process RDF graph
        self._extract_nodes(g)
        self._extract_features(g)
        self._assign_features()
        self._extract_edges(g)
        # self._filter_data()
        # Assign labels to Person nodes
        self._assign_labels()
        
        # Save processed data
        # self._save_processed_data()
        # for edge_type, info in self.hetero_data.edge_types.items():
        #     print(f"[DEBUG] Edge Type: {edge_type}, Unique Edge Types: {torch.unique(info.edge_type)}")
        return self.hetero_data

    def _extract_nodes(self, g):
        """Extracts nodes, assigns them unique IDs per node type, and stores a detailed mapping."""
        self.node_mapping = {}  # Maps (node_type, instance_name) → node index, full URI
        self.node_type_counters = defaultdict(int)  # Separate counters for each node type

        for subject, predicate, obj in g.triples((None, RDF.type, None)):  
            # **Fix: Extract only the last part of the URI (shorten node type)**
            node_type = obj.split("#")[-1] if "#" in obj else obj.split("/")[-1]
            

            if any(word in node_type.lower() for word in ["class", "restriction", "datatype"]):
                # print(f"[DEBUG] Skipping node type '{node_type}' for subject {subject}")
                continue

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
        # print(f"\n[DEBUG] Extracted {sum(self.node_type_counters.values())} nodes across {len(self.node_type_counters)} types.")
        # for node_type, count in self.node_type_counters.items():
        #     print(f"[DEBUG] {node_type}: {count} nodes")

        # print("\n[DEBUG] Checking node_mapping consistency for 'Person' nodes:")
        # i = 0
        # for key, value in list(self.node_mapping.items()):
        #     if key[0] == "Person" and i < 10:  # Only check Person nodes
        #         print(f"[DEBUG] Mapping - Instance: {key[1]}, URI: {value['full_uri']}, Index: {value['node_index']}")
        #         i += 1

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
        """Extracts all edges from the RDF graph and assigns correct relation types."""
        relation_type_counter = {}
        current_relation_index = 0

        for subject, predicate, obj in g:
            if subject in self.node_id_map and obj in self.node_id_map:
                subject_id = self.node_id_map[subject]
                object_id = self.node_id_map[obj]
                source_type = self.node_type_map[subject]
                target_type = self.node_type_map[obj]

                # Extract relation name
                relation = str(predicate).split("#")[-1] if "#" in str(predicate) else str(predicate).split("/")[-1]

                # Assign a unique index to each relation
                if relation not in relation_type_counter:
                    relation_type_counter[relation] = current_relation_index
                    current_relation_index += 1

                edge_type = relation_type_counter[relation]

                edge_key = (source_type, relation, target_type)
                self.edge_types[edge_key][0].append(subject_id)
                self.edge_types[edge_key][1].append(object_id)

                # Assign edge_type attribute
                edge_index = torch.tensor([self.edge_types[edge_key][0], self.edge_types[edge_key][1]], dtype=torch.long)
                edge_type_tensor = torch.full((edge_index.shape[1],), edge_type, dtype=torch.long)

                self.hetero_data[edge_key].edge_index = edge_index
                self.hetero_data[edge_key].edge_type = edge_type_tensor

        # print(f"[DEBUG] Extracted {len(relation_type_counter)} unique relation types.")
        # print(f"[DEBUG] Relation types: {relation_type_counter}")



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
        # torch.save(self.filtered_hetero_data, os.path.join(dataset_dir, "hetero_data.pt"))
        torch.save(self.hetero_data, os.path.join(dataset_dir, "hetero_data.pt"))
        # Save transformed labels
        label_transformer = TransformLabels(self.dataset_name)
        train_file = f"data/{self.dataset_name}/raw/trainingSet.tsv"
        test_file = f"data/{self.dataset_name}/raw/testSet.tsv"

        train_output = os.path.join(dataset_dir, "trainingSet_updated.csv")
        test_output = os.path.join(dataset_dir, "testSet_updated.csv")

        # Call the function with correct arguments
        label_transformer.transform_and_save_labels(train_file, test_file, train_output, test_output)

    def _assign_labels_to_aifb(self):
        """Assigns labels from both training and test sets, using node_mapping for direct lookup."""
        train_file = f"data/{self.dataset_name}/raw/trainingSet.tsv"
        test_file = f"data/{self.dataset_name}/raw/testSet.tsv"

        label_transformer = TransformLabels(self.dataset_name)
        train_df, test_df = label_transformer.assign_labels(train_file, test_file)

        target_node_type = "Person" if self.dataset_name == "aifb" else "MutagenicCompound"
        target_nodes = list(self.nodes_by_type.get(target_node_type, []))

        labels = torch.full((len(target_nodes),), -1, dtype=torch.long)  # -1 means unlabeled

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
        # self.filtered_hetero_data[target_node_type].y = labels
        self.hetero_data[target_node_type].y = labels
        
        self.train_indices = torch.tensor(train_indices, dtype=torch.long)
        self.test_indices = torch.tensor(test_indices, dtype=torch.long)

        # Save train/test masks
        # self.filtered_hetero_data[target_node_type].train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        # self.filtered_hetero_data[target_node_type].test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        # self.filtered_hetero_data[target_node_type].train_mask[self.train_indices] = True
        # self.filtered_hetero_data[target_node_type].test_mask[self.test_indices] = True
        self.hetero_data[target_node_type].train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        self.hetero_data[target_node_type].test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        self.hetero_data[target_node_type].train_mask[self.train_indices] = True
        self.hetero_data[target_node_type].test_mask[self.test_indices] = True
        # print(f"[DEBUG] Assigned labels: {labels.sum().item()} positive labels")
        # print(f"[DEBUG] Train nodes: {len(train_indices)}, Test nodes: {len(test_indices)}")

                # Print the first 10 assigned labels for the correct node type
        # print(f"\n[DEBUG] Checking First 10 Assigned Labels for '{target_node_type}':")
        # if target_node_type in self.hetero_data.node_types:
        #     for i in range(min(10, len(self.hetero_data[target_node_type].y))):
        #         print(f"[DEBUG] Index {i}: Label = {self.hetero_data[target_node_type].y[i].item()}")
        # else:
        #     print(f"[DEBUG] ERROR: Node type '{target_node_type}' not found in filtered_hetero_data!")

 
        self._save_processed_data()

    # def _assign_labels(self):
    #     """Assigns labels from training and test sets using node_mapping for AIFB/MUTAG,
    #     or loads labels directly from trainingSet.tsv and testSet.tsv for 'mini' dataset."""
        
    #     if self.dataset_name == "mini":
    #         print("[DEBUG] Assigning labels from TSV files for 'mini' dataset...")
            
    #         train_file = "data/mini/raw/trainingSet.tsv"
    #         test_file = "data/mini/raw/testSet.tsv"

    #         train_df = pd.read_csv(train_file, sep="\t")
    #         test_df = pd.read_csv(test_file, sep="\t")

    #         target_node_type = "A"  # `A` nodes contain classification labels in `mini`
    #         target_nodes = list(self.nodes_by_type.get(target_node_type, []))

    #         print(f"[DEBUG] Total nodes found for '{target_node_type}': {len(target_nodes)}")

    #         # Initialize labels (-1 means unlabeled)
    #         labels = torch.full((len(target_nodes),), -1, dtype=torch.long)

    #         # Initialize train/test masks
    #         train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
    #         test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)

    #         # Assign labels from train/test sets
    #         for dataset, df, mask in [("train", train_df, train_mask), ("test", test_df, test_mask)]:
    #             for _, row in df.iterrows():
    #                 instance_name = str(row["A_instance"]).split("/")[-1]  # Extract A1, A2, etc.
                    
    #                 if (target_node_type, instance_name) in self.node_mapping:
    #                     node_info = self.node_mapping[(target_node_type, instance_name)]
    #                     node_index = node_info["node_index"]

    #                     # Assign label from classification column
    #                     labels[node_index] = int(row["classification"])  # Convert to integer label

    #                     # Assign train/test mask
    #                     mask[node_index] = True
    #                 else:
    #                     print(f"[DEBUG] WARNING: '{instance_name}' not found in node_mapping!")

    #         # Assign labels and masks to HeteroData
    #         if target_node_type in self.hetero_data:
    #             self.hetero_data[target_node_type].y = labels
    #             self.hetero_data[target_node_type].train_mask = train_mask
    #             self.hetero_data[target_node_type].test_mask = test_mask

    #             print(f"[DEBUG] Final assigned labels for '{target_node_type}': {labels.tolist()}")
    #             print(f"[DEBUG] Train mask count: {train_mask.sum().item()}, Test mask count: {test_mask.sum().item()}")
    #         else:
    #             print(f"[DEBUG] ERROR: '{target_node_type}' not found in hetero_data!")

    #     else:
    #         print("[DEBUG] Assigning labels from TSV files for AIFB/MUTAG datasets...")

    #         train_file = f"data/{self.dataset_name}/raw/trainingSet.tsv"
    #         test_file = f"data/{self.dataset_name}/raw/testSet.tsv"

    #         label_transformer = TransformLabels(self.dataset_name)
    #         train_df, test_df = label_transformer.assign_labels(train_file, test_file)

    #         target_node_type = "Person" if self.dataset_name == "aifb" else "MutagenicCompound"
    #         target_nodes = list(self.nodes_by_type.get(target_node_type, []))

    #         print(f"[DEBUG] Total nodes found for '{target_node_type}': {len(target_nodes)}")

    #         labels = torch.full((len(target_nodes),), -1, dtype=torch.long)
    #         train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
    #         test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)

    #         for dataset, df, mask in [("train", train_df, train_mask), ("test", test_df, test_mask)]:
    #             for _, row in df.iterrows():
    #                 raw_instance = str(row["person"]).strip() if "person" in row else str(row["compound"]).strip()
    #                 instance_name = raw_instance.split("#")[-1] if "#" in raw_instance else raw_instance.split("/")[-1]

    #                 if (target_node_type, instance_name) in self.node_mapping:
    #                     node_info = self.node_mapping[(target_node_type, instance_name)]
    #                     node_index = node_info["node_index"]

    #                     labels[node_index] = row["new_label"]
    #                     mask[node_index] = True
    #                 else:
    #                     print(f"[DEBUG] WARNING: '{instance_name}' not found in node_mapping!")

    #         if target_node_type in self.hetero_data:
    #             self.hetero_data[target_node_type].y = labels
    #             self.hetero_data[target_node_type].train_mask = train_mask
    #             self.hetero_data[target_node_type].test_mask = test_mask

    #             print(f"[DEBUG] Final assigned labels for '{target_node_type}': {labels.tolist()}")
    #         else:
    #             print(f"[DEBUG] ERROR: '{target_node_type}' not found in hetero_data!")


    def _assign_labels(self):
        """Unified label assignment function that calls the appropriate method for the dataset."""
        
        # print("[INFO] Assigning labels for dataset:", self.dataset_name)

        if self.dataset_name == "mini":
            self._assign_labels_to_mini()
        elif self.dataset_name in ["aifb", "mutag"]:
            self._assign_labels_to_aifb()
        else:
            raise ValueError(f"[ERROR] Unsupported dataset: {self.dataset_name}")

        # print("[SUCCESS] Label assignment completed for dataset:", self.dataset_name)

    def get_statistics(self):
        """Computes and returns dataset statistics before and after filtering."""
        return {
            "Before Filtering": {
                "Total Node Types": len(self.nodes_by_type),
                "Total Nodes": sum(len(nodes) for nodes in self.nodes_by_type.values()),
                "Total Edge Types": len(self.edge_types),
                "Total Edges": sum(len(edges[0]) for edges in self.edge_types.values())
            }
            # "After Filtering": {
            #     "Total Node Types": len(self.filtered_hetero_data.node_types),
            #     "Total Nodes": sum(len(self.nodes_by_type[node_type]) for node_type in self.filtered_hetero_data.node_types if node_type in self.nodes_by_type),
            #     "Total Edge Types": len(self.filtered_hetero_data.edge_types),
            #     "Total Edges": sum(len(self.filtered_hetero_data[edge_type].edge_index[0]) for edge_type in self.filtered_hetero_data.edge_types)
            # }
        }


    def _assign_labels_to_mini(self):
        """Assigns classification labels to the 'mini' dataset directly from TSV files."""
        
        if self.dataset_name != "mini":
            print(f"[WARNING] Label assignment skipped: This function only works for mini dataset.")
            return

        # print("[INFO] Assigning labels directly from raw TSV files for mini dataset...")

        train_file = f"data/{self.dataset_name}/raw/trainingSet.tsv"
        test_file = f"data/{self.dataset_name}/raw/testSet.tsv"

        # Read training and test label data
        train_df = pd.read_csv(train_file, sep="\t")
        test_df = pd.read_csv(test_file, sep="\t")

        target_node_type = "A"  # A nodes contain classification labels in mini

        # Ensure 'A' exists in hetero_data before assigning labels
        if target_node_type not in self.hetero_data.node_types:
            print(f"[ERROR] {target_node_type} not found in hetero_data! Labels cannot be assigned.")
            return

        # Extract nodes from self.node_mapping
        target_nodes = {
            instance_name: node_info["node_index"]
            for (node_type, instance_name), node_info in self.node_mapping.items()
            if node_type == target_node_type
        }

        # print(f"[DEBUG] Extracted {len(target_nodes)} {target_node_type} nodes for label assignment.")

        if not target_nodes:
            print(f"[WARNING] No nodes found for {target_node_type} in hetero_data.")
            return

        # Initialize label tensor (-1 means unlabeled)
        labels = torch.full((len(target_nodes),), -1, dtype=torch.long)

        # Initialize train and test masks
        train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
        test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)

        # Assign labels using the classification column
        assigned_labels = {}  # Store assigned labels for printing

        for dataset, df, mask in [("train", train_df, train_mask), ("test", test_df, test_mask)]:
            # print(f"[INFO] Processing {dataset} dataset with {len(df)} samples...")

            for _, row in df.iterrows():
                instance_name = str(row["A_instance"]).split("/")[-1]  # Extract instance name (A1, A2, etc.)

                if instance_name in target_nodes:
                    node_index = target_nodes[instance_name]
                    labels[node_index] = int(row["classification"])  # Convert classification to integer
                    mask[node_index] = True  # Mark as train/test

                    assigned_labels[instance_name] = row["classification"]
                else:
                    print(f"[WARNING] Node {instance_name} not found in node_mapping!")

        # Assign the labels and masks to hetero_data
        self.hetero_data[target_node_type].y = labels
        self.hetero_data[target_node_type].train_mask = train_mask
        self.hetero_data[target_node_type].test_mask = test_mask

        # print(f"[SUCCESS] Labels assigned to {target_node_type}.")
        # print(f"[DEBUG] Final train mask count: {train_mask.sum().item()}, test mask count: {test_mask.sum().item()}")
        
        # Print assigned labels for verification
        # print("[INFO] Assigned Labels:")
        # for node, label in assigned_labels.items():
        #     print(f"Node {node}: Label {label}")



    def get_namespace_prefixes(self):
        """
        Extracts namespace prefixes from node_mapping and RDF edge predicates.
        Returns a tuple (class_prefix, relation_prefix) as strings.
        """
        class_prefix = None
        relation_prefix = None

        # Extract class prefix from any node
        for (_, instance_info) in self.node_mapping.items():
            full_uri = instance_info["full_uri"]
            if "#" in full_uri:
                class_prefix = full_uri.split("#")[0] + "#"
            elif "/" in full_uri:
                class_prefix = full_uri.rsplit("/", 1)[0] + "/"
            break

        # Extract real relation prefix from RDF graph
        for (source_type, relation, target_type) in self.edge_types:
            for subj, predicate, obj in self.g:
                pred_str = str(predicate)
                if pred_str.endswith(relation):
                    if "#" in pred_str:
                        relation_prefix = pred_str.rsplit("#", 1)[0] + "#"
                    elif "/" in pred_str:
                        relation_prefix = pred_str.rsplit("/", 1)[0] + "/"
                    break
            break

        return class_prefix, relation_prefix

