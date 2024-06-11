import torch_geometric.transforms as transforms
from torch_geometric.data import HeteroData
from torch_geometric.datasets import *
from rdflib import Graph
from collections import defaultdict
from torch_geometric.data import HeteroData
import torch
from collections import Counter
from torch_geometric.utils import index_sort, k_hop_subgraph


# entity_prefix = 'http://www.aifb.uni-karlsruhe.de/'
# relation_prefix = 'http://swrc.ontoware.org/'











# # Load RDF graph from file
# g = Graph()
# g.parse("thesis/AIFB/aifb_fixed_complete.nt", format="nt")  # Replace "aifb.owl" with your RDF file

# # Initialize HeteroData object
# hetero_data = HeteroData()

# # Extract entities and relationships
# # entities = set()
# # relationships = defaultdict(list)

# freq = Counter(g.predicates())

# relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
# subjects = set(g.subjects())
# objects = set(g.objects())
# nodes = list(subjects.union(objects))


# # For reproducibility, ensure a fixed order
# subjects = sorted(subjects)
# objects = sorted(objects)
# nodes = sorted(nodes)


# N = len(nodes)
# R = 2 * len(relations)

# relations_dict = {rel: i for i, rel in enumerate(relations)}
# nodes_dict = {str(node): i for i, node in enumerate(nodes)}

# edges = []
# for s, p, o in g.triples((None, None, None)):
#     src, dst = nodes_dict[str(s)], nodes_dict[str(o)]
#     rel = relations_dict[p]
#     edges.append([src, dst, 2 * rel])
#     edges.append([dst, src, 2 * rel + 1])

# edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
# _, perm = index_sort(N * R * edge[0] + R * edge[1] + edge[2])
# edge = edge[:, perm]

# edge_index, edge_type = edge[:2], edge[2]


# #rename edges such that they are valid Python identifiers (to avoid warnings down the road)

# edge_type_names = []

# for e in edge_type:
#     tmp = tuple(['v', 'r' + str(e.numpy()), 'v'])
#     edge_type_names.append(tmp)


# print(nodes[8182])

# # Create a dictionary to hold node classes
# node_classes = {"node": list(range(len(nodes)))}

# # Assign the node classes to the HeteroData object
# hetero_data.node_stores["class"] = node_classes

# # Print node classes
# print("Node classes:", hetero_data.node_stores["class"])






# for subject, predicate, obj in g:
#     subject = str(subject)
#     predicate = str(predicate)
#     obj = str(obj)
#     entities.add(subject)
#     entities.add(obj)
#     relationships[predicate].append((subject, obj))

# # Map entities to nodes
# entity2idx = {entity: i for i, entity in enumerate(entities)}
# for entity in entities:
#     hetero_data[entity] = {"idx": entity2idx[entity]}  # Add entity node to HeteroData

# # Map relationships to edges
# for rel_type, rel_list in relationships.items():
#     src_indices, dst_indices = zip(*[(entity2idx[src], entity2idx[dst]) for src, dst in rel_list])
#     # hetero_data.edge_index[rel_type] = (torch.tensor(src_indices), torch.tensor(dst_indices))
#     # Define relation between entities and assign edge index
#     hetero_data[rel_list[0][0], rel_type, rel_list[0][1]].edge_index = (torch.tensor(src_indices), torch.tensor(dst_indices))

# # Print HeteroData object
# print(hetero_data.node_stores)

#------------------------------------------------------------------------------------------    


# import networkx as nx
# import torch
# from torch_geometric.data import HeteroData
# from rdflib import URIRef, Graph, Namespace
# from rdflib.namespace import RDF,FOAF, RDFS, OWL


# Ex = Namespace("http://www.aifb.uni-karlsruhe.de/")
# person = URIRef("http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL")
# # Define the URI pattern to match specific types
# uri_pattern = "http://swrc.ontoware.org/ontology#"
# subclasses = set()





# # Step 1: Load and parse the N3 file
# g = Graph()
# g.parse("data/aifb/raw/aifb_stripped.nt", format="nt")

# # Dictionary to map subclasses to their parent classes
# subclass_map = {}

# # Identify all subclass relationships
# for subclass, _, parent_class in g.triples((None, RDFS.subClassOf, None)):
#     subclass_map[subclass] = parent_class

# # Function to find the topmost parent class
# def find_topmost_class(cls):
#     while cls in subclass_map:
#         cls = subclass_map[cls]
#     return cls



# # Dictionary to store nodes by their types
# nodes_by_type = {}

# # Identify subclasses
# for s, p, o in g.triples((None, RDFS.subClassOf, None)):
#     subclasses.add(s)

# # Iterate over all triples in the graph
# for subject, predicate, obj in g:
#     if predicate == RDF.type:
#         if obj.startswith(uri_pattern):
#             if obj not in nodes_by_type:
#                 nodes_by_type[obj] = []
#             nodes_by_type[obj].append(subject)


# # Dictionary to store nodes by their types
# nodes_by_type = {}

# # Extract the list of persons
# persons = nodes_by_type.get(person, [])

# # Print the nodes of each type
# for node_type, nodes in nodes_by_type.items():
#     print(f"Type: {node_type}")
#     # for node in nodes:
#     #     print(f"  Node: {node}")

# # Print the persons
# print("\nPersons:")
# for per in persons:
#     print(per)





import networkx as nx
import torch
from torch_geometric.data import HeteroData
from rdflib import URIRef, Graph, Namespace
from rdflib.namespace import RDF, FOAF, RDFS, OWL


Ex = Namespace("http://www.aifb.uni-karlsruhe.de/")
#person = URIRef("http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL")
# Define the URI pattern to match specific types
uri_pattern = "http://swrc.ontoware.org/ontology#"
subclasses = set()



# Step 1: Load and parse the N3 file
g = Graph()
g.parse("data/aifb/raw/aifb_stripped.nt", format="nt")

# Dictionary to map subclasses to their parent classes
subclass_map = {}

# Identify all subclass relationships
# for subclass, _, parent_class in g.triples((None, RDFS.subClassOf, None)):
#     subclass_map[subclass] = parent_class

# # Function to find the topmost parent class
# def find_topmost_class(cls):
#     while cls in subclass_map:
#         cls = subclass_map[cls]
#     return cls

# # Dictionary to store nodes by their types
# nodes_by_type = {}

# # Iterate over all triples in the graph
# for subject, predicate, obj in g:
#     if predicate == RDF.type:
#         if obj.startswith(uri_pattern):
#             if obj not in nodes_by_type:
#                 nodes_by_type[obj] = []
#             nodes_by_type[obj].append(subject)

# # Extract the list of persons
# persons = nodes_by_type.get(person, [])




# Dictionary to map middle parts of URIs to node types
uri_to_node_type = {
    "Personen": "Person",
    "Publikationen": "Publication",
    "Forschungsgebiete" : "ResearchGroup",
    "Projekte" : "Project",
    "ExternerAutor" : "ExternalAuthor"
    # Add other mappings as needed
}

# Dictionary to store nodes by their types
nodes_by_type = {node_type: set() for node_type in uri_to_node_type.values()}  # Use sets to ensure uniqueness

# Step 2: Filter instances matching the URI pattern and organize them
for subject, predicate, obj in g:
    if str(subject).startswith(str(Ex)):
        # Extract the middle part of the URI to use as node type
        parts = str(subject).split("/")
        if len(parts) > 4:  # Ensure there is a middle part to check
            middle_part = parts[-3]
            if middle_part == "Publikationen":
                next_part = parts[-2]
                if next_part == "viewExternerAutorOWL":
                    node_type = "ExternalAuthor"
                else:
                    node_type = "Publication"
            elif middle_part in uri_to_node_type:
                node_type = uri_to_node_type[middle_part]
            nodes_by_type[node_type].add(subject)


# Convert sets to lists for HeteroData
nodes_by_type = {node_type: list(nodes) for node_type, nodes in nodes_by_type.items()}

# Initialize HeteroData
hetero_data = HeteroData()

# Step 3: Convert nodes to HeteroData
for node_type, nodes in nodes_by_type.items():
    # Add nodes to HeteroData with dummy features
    hetero_data[node_type].x = torch.tensor([1.0] * len(nodes)).unsqueeze(1)


print(hetero_data.node_types)
for i in hetero_data.node_types:
    print(hetero_data[i].num_nodes)


