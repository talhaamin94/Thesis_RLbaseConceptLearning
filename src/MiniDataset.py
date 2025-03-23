# from rdflib import Graph, Literal, RDF, RDFS, URIRef, Namespace
# import os
# from pathlib import Path
# import pandas as pd
# import random

# class MiniDataset:
#     def __init__(self, file_path="./data/mini/raw/rdf_graph.nt"):
#         self.file_path = Path(file_path)
#         self.ex = Namespace("http://example.org/")
#         self.g = Graph()
#         self.colors = ["red", "blue", "green", "yellow", "purple", "orange"]

#         # Load or generate the graph
#         if self.file_path.exists():
#             self.load_graph()
#         else:
#             self.generate_graph()
#             self.save_graph()
#         # Create training & test sets
#         self.create_label_files()

#     def create_node(self, class_type, node_id, color):
#         """ Helper function to generate nodes with instance number, name, and color """
#         node_uri = URIRef(f"{class_type}{node_id}")
#         self.g.add((node_uri, RDF.type, class_type))
#         self.g.add((node_uri, self.ex.instance_number, Literal(node_id)))
#         self.g.add((node_uri, self.ex.name, Literal(f"{class_type}{node_id}")))
#         self.g.add((node_uri, self.ex.color, Literal(color)))
#         return node_uri

#     # def generate_graph(self):
#     #     # """ Generates the RDF graph if it does not exist """
#     #     # print("Generating new RDF graph...")
        
#     #     # Define classes
#     #     self.g.add((self.ex.A, RDF.type, RDFS.Class))
#     #     self.g.add((self.ex.B, RDF.type, RDFS.Class))
#     #     self.g.add((self.ex.C, RDF.type, RDFS.Class))

#     #     for i in range(1, 21):
#     #         color_A = self.colors[i % len(self.colors)]
#     #         color_B = self.colors[(i + 1) % len(self.colors)]
#     #         color_C = self.colors[(i + 2) % len(self.colors)]
            
#     #         node_B = self.create_node(self.ex.A, i, color_A)
#     #         node_A = self.create_node(self.ex.B, i, color_B)
#     #         node_C = self.create_node(self.ex.C, i, color_C)

#     #         if i <= 3:
#     #             self.g.add((node_A, self.ex.connected_to, node_C))
#     #             self.g.add((node_B, self.ex.connected_to, node_C))

#     #         node_A_uri = URIRef(f"http://example.org/A{i}")
#     #         node_B_uri = URIRef(f"http://example.org/B{i}")
#     #         self.g.add((node_A_uri, self.ex.related_to, node_B_uri))

#     #     # Add classification triples for node A
#     #     for i in range(1, 6):
#     #         node_A = URIRef(f"http://example.org/A{i}")
#     #         classification = self.classify_node_A(node_A)
#     #         self.g.add((node_A, self.ex.classification, Literal(classification)))

#     def generate_graph(self):
#         """ Generates an RDF graph with hardcoded connections for controlled experimentation. """

#         # Define classes
#         self.g.add((self.ex.A, RDF.type, RDFS.Class))
#         self.g.add((self.ex.B, RDF.type, RDFS.Class))
#         self.g.add((self.ex.C, RDF.type, RDFS.Class))

#         # Number of nodes per class
#         num_nodes = 10  

#         # Create 20 nodes for each class A, B, C
#         for i in range(1, num_nodes + 1):
#             color_A = self.colors[i % len(self.colors)]
#             color_B = self.colors[(i + 1) % len(self.colors)]
#             color_C = self.colors[(i + 2) % len(self.colors)]
            
#             node_A = self.create_node(self.ex.A, i, color_A)
#             node_B = self.create_node(self.ex.B, i, color_B)
#             node_C = self.create_node(self.ex.C, i, color_C)

#         # Hardcoded connections: Only first half of A nodes connect to C
#         for i in range(1, num_nodes // 2 + 1):  
#             node_A_uri = URIRef(f"http://example.org/A{i}")
#             node_C_uri = URIRef(f"http://example.org/C{i}")  
#             self.g.add((node_A_uri, self.ex.connected_to, node_C_uri))  # Connect first half of A to C

#         # # Hardcoded connections: Each A node connects to a predefined B node
#         # hardcoded_connections = [
#         #     (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),  
#         #     (11, 12), (13, 14), (15, 16), (17, 18), (19, 20)
#         # ]  

#         for i in range(1, num_nodes + 1, 2):  # Every alternate A connects to a B
#             node_A_uri = URIRef(f"http://example.org/A{i}")
#             node_B_uri = URIRef(f"http://example.org/B{i}")
#             self.g.add((node_A_uri, self.ex.related_to, node_B_uri))  # Hardcoded A → B connections
        
        

#         # Hardcoded B → C connections (New Addition)
#         for i in range(1, num_nodes + 1, 2):  # Every alternate B connects to a C
#             node_B_uri = URIRef(f"http://example.org/B{i}")
#             node_C_uri = URIRef(f"http://example.org/C{i}")  
#             self.g.add((node_B_uri, self.ex.connected_to, node_C_uri))  # B connects to corresponding C

#         # **Automatically classify A nodes using `classify_node_A`**
#         for i in range(1, num_nodes + 1):
#             node_A_uri = URIRef(f"http://example.org/A{i}")
#             classification = self.classify_node_A(node_A_uri)  # Dynamically assign classification
#             self.g.add((node_A_uri, self.ex.classification, Literal(classification)))



#     def classify_node_A(self, node_A):
#         """ Determines if a node of type A is connected to any node of type C """
#         connected_to_C = False
#         for _, _, obj in self.g.triples((node_A, self.ex.connected_to, None)):
#             if obj.startswith(f"{self.ex}C"):
#                 connected_to_C = True
#                 break
#         return "positive" if connected_to_C else "negative"

#     def save_graph(self):
#         """ Saves the generated graph to an NT file """
#         os.makedirs(self.file_path.parent, exist_ok=True)
#         self.g.serialize(destination=str(self.file_path), format="nt")
#         print(f"Graph serialized and saved to {self.file_path}")

#     def load_graph(self):
#         """ Loads the RDF graph from the existing file """
#         # print(f"Loading existing graph from {self.file_path}...")
#         self.g.parse(str(self.file_path), format="nt")

#     def get_graph(self):
#         """ Returns the RDF graph object """
#         return self.g



#     def create_label_files(self):
#         """Extracts classification labels from RDF graph and creates trainingSet.tsv and testSet.tsv if they do not exist."""
        
#         train_path = Path("data/mini/raw/trainingSet.tsv")
#         test_path = Path("data/mini/raw/testSet.tsv")

#         # Check if label files already exist
#         if train_path.exists() and test_path.exists():
#             # print("[INFO] Label files already exist. Skipping label generation.")
#             return  # Exit function early
        
#         # print("[INFO] Generating new training and test label files...")

#         labeled_nodes = []

#         for subj, pred, obj in self.g.triples((None, self.ex.classification, None)):
#             node_uri = str(subj)  # Full URI (e.g., "http://example.org/A1")
#             instance_name = node_uri.split("/")[-1]  # Extract last part (e.g., "A1")
#             label = 1 if str(obj).lower() == "positive" else 0  # Convert classification to 1/0
#             numerical_id = int(instance_name[1:])  # Extract numeric ID from name (e.g., "A1" → 1)

#             labeled_nodes.append((node_uri, numerical_id, label))

#         # Shuffle data
#         random.shuffle(labeled_nodes)

#         # Split: 80% train, 20% test
#         split_idx = int(len(labeled_nodes) * 0.8)
#         train_data = labeled_nodes[:split_idx]
#         test_data = labeled_nodes[split_idx:]

#         # Convert to DataFrame with correct column names
#         train_df = pd.DataFrame(train_data, columns=["A_instance", "id", "classification"])
#         test_df = pd.DataFrame(test_data, columns=["A_instance", "id", "classification"])

#         # Save to TSV files
#         os.makedirs("data/mini/raw/", exist_ok=True)
#         train_df.to_csv(train_path, sep="\t", index=False)
#         test_df.to_csv(test_path, sep="\t", index=False)

#         # print(f"[INFO] Training and test sets created:")
#         # print(f" - Training set: {len(train_df)} samples at {train_path}")
#         # print(f" - Test set: {len(test_df)} samples at {test_path}")


# # # Example usage:
# # if __name__ == "__main__":
# #     generator = RDFGraphGenerator()
# #     rdf_graph = generator.get_graph()

# #     # Print some triples
# #     for subj, pred, obj in rdf_graph:
# #         print(f"{subj} -- {pred} --> {obj}")

from rdflib import Graph, Literal, RDF, RDFS, URIRef, Namespace
import os
from pathlib import Path
import pandas as pd
import random

class MiniDataset:
    def __init__(self, file_path="./data/mini/raw/rdf_graph.nt"):
        self.file_path = Path(file_path)
        self.ns = "http://example.org/"
        self.ex = Namespace(self.ns)
        self.g = Graph()
        self.colors = ["red", "blue", "green", "yellow", "purple", "orange"]

        if self.file_path.exists():
            self.load_graph()
        else:
            self.generate_graph()
            self.save_graph()

        self.create_label_files()

    def create_node(self, class_type_uri, node_id, color):
        """Helper to create a node with full IRI and add its properties."""
        node_uri = URIRef(f"{class_type_uri}{node_id}")
        self.g.add((node_uri, RDF.type, URIRef(class_type_uri)))
        self.g.add((node_uri, URIRef(self.ns + "instance_number"), Literal(node_id)))
        self.g.add((node_uri, URIRef(self.ns + "name"), Literal(f"{class_type_uri}{node_id}")))
        self.g.add((node_uri, URIRef(self.ns + "color"), Literal(color)))
        return node_uri

    def generate_graph(self):
        """Generates RDF graph with consistent full IRIs."""
        # Define classes and properties with full URIs
        for class_name in ["A", "B", "C"]:
            self.g.add((URIRef(self.ns + class_name), RDF.type, RDFS.Class))

        for prop in ["connected_to", "related_to", "classification", "instance_number", "name", "color"]:
            self.g.add((URIRef(self.ns + prop), RDF.type, RDF.Property))

        num_nodes = 10

        for i in range(1, num_nodes + 1):
            color_A = self.colors[i % len(self.colors)]
            color_B = self.colors[(i + 1) % len(self.colors)]
            color_C = self.colors[(i + 2) % len(self.colors)]

            node_A = self.create_node(self.ns + "A", i, color_A)
            node_B = self.create_node(self.ns + "B", i, color_B)
            node_C = self.create_node(self.ns + "C", i, color_C)

        # A → C connections (first half only)
        for i in range(1, num_nodes // 2 + 1):
            self.g.add((URIRef(f"{self.ns}A{i}"), URIRef(f"{self.ns}connected_to"), URIRef(f"{self.ns}C{i}")))

        # A → B connections (odd-numbered only)
        for i in range(1, num_nodes + 1, 2):
            self.g.add((URIRef(f"{self.ns}A{i}"), URIRef(f"{self.ns}related_to"), URIRef(f"{self.ns}B{i}")))

        # B → C connections (odd-numbered only)
        for i in range(1, num_nodes + 1, 2):
            self.g.add((URIRef(f"{self.ns}B{i}"), URIRef(f"{self.ns}connected_to"), URIRef(f"{self.ns}C{i}")))

        # Assign classification labels to A nodes
        for i in range(1, num_nodes + 1):
            node_A_uri = URIRef(f"{self.ns}A{i}")
            classification = self.classify_node_A(node_A_uri)
            self.g.add((node_A_uri, URIRef(self.ns + "classification"), Literal(classification)))

    def classify_node_A(self, node_A):
        """Classify A node as positive if it's connected to a C node."""
        for _, _, obj in self.g.triples((node_A, URIRef(self.ns + "connected_to"), None)):
            if str(obj).startswith(f"{self.ns}C"):
                return "positive"
        return "negative"

    def save_graph(self):
        """Serialize RDF graph to .nt format."""
        os.makedirs(self.file_path.parent, exist_ok=True)
        self.g.serialize(destination=str(self.file_path), format="nt")
        print(f"[MiniDataset] Graph saved to {self.file_path}")

    def load_graph(self):
        """Load RDF graph from existing file."""
        self.g.parse(str(self.file_path), format="nt")

    def get_graph(self):
        """Return RDFLib Graph object."""
        return self.g

    def create_label_files(self):
        """Generate training and test TSVs from classification labels."""
        train_path = Path("data/mini/raw/trainingSet.tsv")
        test_path = Path("data/mini/raw/testSet.tsv")

        if train_path.exists() and test_path.exists():
            return  # Skip if already created

        labeled_nodes = []

        for subj, _, obj in self.g.triples((None, URIRef(self.ns + "classification"), None)):
            instance_name = str(subj).split("/")[-1]  # A1, A2, ...
            label = 1 if str(obj).lower() == "positive" else 0
            numerical_id = int(instance_name[1:])  # e.g. A3 → 3

            labeled_nodes.append((str(subj), numerical_id, label))

        random.shuffle(labeled_nodes)
        split = int(0.8 * len(labeled_nodes))
        train_data = labeled_nodes[:split]
        test_data = labeled_nodes[split:]

        train_df = pd.DataFrame(train_data, columns=["A_instance", "id", "classification"])
        test_df = pd.DataFrame(test_data, columns=["A_instance", "id", "classification"])

        os.makedirs(train_path.parent, exist_ok=True)
        train_df.to_csv(train_path, sep="\t", index=False)
        test_df.to_csv(test_path, sep="\t", index=False)
        print("[MiniDataset] Train/test label files created.")
