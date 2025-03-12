from GNNTrainer import GNNTrainer
from DataConversion import RDFGraphConverter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from Graph_to_Homogeneous import Graph_to_Homogeneous



# Load dataset
converter = RDFGraphConverter("AIFB")
hetero_data = converter.load_dataset()
from GNNTrainer import GNNTrainer
from DataConversion import RDFGraphConverter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from Graph_to_Homogeneous import Graph_to_Homogeneous

# Load dataset
converter = RDFGraphConverter("AIFB")
hetero_data = converter.load_dataset()

# Re-import necessary libraries
import pandas as pd
import os

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



gnn = GNNTrainer(hetero_data, "Person")
gnn.run_training()
# gnn.evaluate_test_set()
# print(len(gnn.get_positive_nodes())+len(gnn.get_negative_nodes()))
# print(gnn.get_negative_nodes())
