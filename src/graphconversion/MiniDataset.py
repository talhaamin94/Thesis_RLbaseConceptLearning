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
