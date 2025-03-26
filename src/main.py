from GNNTrainer import GNNTrainer
from RDFGraphConverter import RDFGraphConverter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from MiniDataset import MiniDataset
from TransETrainer import TransETrainer
from RLTrainer import RLTrainer
import time
from Evaluator import Evaluator
from owlapy.class_expression import OWLClass, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf
from owlapy.owl_property import OWLObjectProperty
from owlapy.iri import IRI
from owlapy.render import DLSyntaxObjectRenderer

# Load dataset
# converter = RDFGraphConverter("AIFB")
# hetero_data = converter.load_dataset()
# print(converter.get_statistics())
# print(hetero_data)
# gnn = GNNTrainer(hetero_data, "Person")
# gnn.run_training()

def inspect_node_connections(data, node_type, node_local_index):
    if node_type not in data.node_types:
        print(f"[ERROR] Node type '{node_type}' does not exist in the graph.")
        return

    num_nodes = data[node_type].x.size(0)
    if node_local_index < 0 or node_local_index >= num_nodes:
        print(f"[ERROR] Node index {node_local_index} is out of range for node type '{node_type}' (max index: {num_nodes-1}).")
        return

    print(f"[INFO] Node ({node_type}, {node_local_index}) exists. Checking for connections...\n")
    found = False

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if edge_type[0] == node_type and src == node_local_index:
                print(f"Outgoing: ({node_type}, {node_local_index}) --[{edge_type[1]}]--> ({edge_type[2]}, {dst})")
                found = True
            elif edge_type[2] == node_type and dst == node_local_index:
                print(f"Incoming: ({edge_type[0]}, {src}) --[{edge_type[1]}]--> ({node_type}, {node_local_index})")
                found = True

    if not found:
        print(f"[INFO] No connections found for ({node_type}, {node_local_index}).")


# converter = RDFGraphConverter("mini")
# hetero_data = converter.load_dataset()
# # print(converter.get_statistics())
# # print(hetero_data)

# gnn = GNNTrainer(hetero_data, "A", hidden_dim=64, learning_rate=0.01,dropout_rate=0.5, wd = 1e-3)
# # gnn.run_training(30)

# gnn.load_model()
# transE = TransETrainer(hetero_data,128,100,1024,0.002)
# transE.train()


# gnn = HGNNTrainer(hetero_data, "A", hidden_dim=32, learning_rate=0.001, dropout_rate=0.5, wd=1e-3)
# gnn.train_model(epochs=50)
# test_acc = gnn.test_model()
# print(f"Test Accuracy: {test_acc:.4f}")
# converter = RDFGraphConverter("")







# gnn.evaluate_test_set()
# print(len(gnn.get_positive_nodes())+len(gnn.get_negative_nodes()))
# print(gnn.get_negative_nodes())


# # Define hyperparameter grid
# learning_rates = [0.005, 0.01, 0.1]
# hidden_dims = [16, 32, 64]
# epochs_list = [15, 25, 40]
# dropout_rates = [0.2, 0.3, 0.4]

# # Store results
# experiment_results = []

# # Placeholder for hetero_data (Assuming it should be loaded before running this)
# # hetero_data = Load your heterogeneous graph data here

# # Check if GNNTrainer is defined before proceeding
# try:
#     GNNTrainer
# except NameError:
#     print("Error: GNNTrainer is not defined. Please ensure the class is loaded before running this script.")
#     raise

# # Iterate over all hyperparameter combinations
# for lr in learning_rates:
#     for hidden_dim in hidden_dims:
#         for epochs in epochs_list:
#             for dropout_rate in dropout_rates:
#                 print(f"\nTraining with LR={lr}, Hidden Dim={hidden_dim}, Epochs={epochs}, Dropout={dropout_rate}")

#                 # Initialize and train the model
#                 gnn_trainer = GNNTrainer(
#                     hetero_data, 
#                     node_type="Person",
#                     hidden_dim=hidden_dim, 
#                     learning_rate=lr,
#                     dropout_rate=dropout_rate
#                 )
                
#                 gnn_trainer.run_training(epochs=epochs)

#                 # Evaluate the model
#                 train_acc, test_acc = gnn_trainer.test()

#                 # Store results
#                 experiment_results.append({
#                     "Learning Rate": lr,
#                     "Hidden Dim": hidden_dim,
#                     "Epochs": epochs,
#                     "Dropout Rate": dropout_rate,
#                     "Train Accuracy": train_acc,
#                     "Test Accuracy": test_acc
#                 })

# # Convert results to a DataFrame for visualization
# experiment_results_df = pd.DataFrame(experiment_results)

# # Save results for further analysis
# save_dir = "results"
# os.makedirs(save_dir, exist_ok=True)
# results_save_path = os.path.join(save_dir, "hyperparameter_results.csv")
# experiment_results_df.to_csv(results_save_path, index=False)

# # Display results
# print("\nHyperparameter Tuning Results:")
# print(experiment_results_df)

# print(f"\nHyperparameter tuning results saved to: {results_save_path}")

# Train and evaluate the GNN model
node_type = "Person"
dataset = "aifb"
# node_type = "A"
# dataset = "mini"
converter = RDFGraphConverter(dataset)
hetero_data = converter.load_dataset()
class_prefix, relation_prefix = converter.get_namespace_prefixes()

trainer = TransETrainer(hetero_data,num_epochs=2000)
gnn_trainer = GNNTrainer(hetero_data, node_type=node_type)
gnn_trainer.run_training()
trainer.train()


rlagent = RLTrainer(hetero_data,node_type,class_prefix,relation_prefix,gnn_trainer,trainer,50)
rlagent.train()
rlagent.test()
