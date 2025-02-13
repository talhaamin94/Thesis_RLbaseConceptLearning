from GNNTrainer import GNNTrainer
from DataConversion import RDFGraphConverter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os




# Load dataset
converter = RDFGraphConverter("AIFB")
hetero_data = converter.load_dataset()

# # Define hyperparameter grid
# learning_rates = [0.001, 0.01, 0.1]
# hidden_dims = [16, 32, 64]
# epochs_list = [10, 20, 50]

# # Store results
# best_f1 = 0
# best_hyperparams = None
# results = []

# # Iterate through all combinations of hyperparameters
# for lr, hidden_dim, epochs in itertools.product(learning_rates, hidden_dims, epochs_list):
#     print(f"\nTraining with LR={lr}, Hidden Dim={hidden_dim}, Epochs={epochs}")
    
#     # Initialize trainer with different hyperparameters
#     gnn_trainer = GNNTrainer(
#         hetero_data, 
#         node_type="Person",  
#         hidden_dim=hidden_dim, 
#         learning_rate=lr
#     )

#     # Train model
#     gnn_trainer.train(epochs=epochs)

#     # Evaluate on the test set
#     _, _, _, f1_score_test = gnn_trainer.evaluate_test_set()

#     # Store results
#     results.append((lr, hidden_dim, epochs, f1_score_test))

#     # Update best hyperparameters
#     if f1_score_test > best_f1:
#         best_f1 = f1_score_test
#         best_hyperparams = (lr, hidden_dim, epochs)

# # Convert results to a NumPy array for easier plotting
# results_array = np.array(results, dtype=object)

# # Reshape results into a 3D array (learning_rates x hidden_dims x epochs)
# f1_scores = results_array[:, 3].astype(float).reshape(
#     len(learning_rates), len(hidden_dims), len(epochs_list)
# )

# # Plot the results
# fig, ax = plt.subplots(1, len(epochs_list), figsize=(15, 5))

# for i, epoch in enumerate(epochs_list):
#     im = ax[i].imshow(f1_scores[:, :, i], cmap="viridis", aspect="auto")
#     ax[i].set_xticks(np.arange(len(hidden_dims)))
#     ax[i].set_xticklabels(hidden_dims)
#     ax[i].set_yticks(np.arange(len(learning_rates)))
#     ax[i].set_yticklabels(learning_rates)
#     ax[i].set_xlabel("Hidden Dim")
#     ax[i].set_title(f"Epochs: {epoch}")

# ax[0].set_ylabel("Learning Rate")
# fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.05)

# # Define the save path
# save_dir = "results"
# save_path = os.path.join(save_dir, "hyperparameter_tuning.png")

# # Ensure the directory exists
# os.makedirs(save_dir, exist_ok=True)

# # Save the figure
# plt.savefig(save_path, dpi=300, bbox_inches="tight")

# print(f"Figure saved successfully at {save_path}")

# plt.show()



# # Print the best hyperparameter combination
# print("\nBest Hyperparameters:")
# print(f"Learning Rate: {best_hyperparams[0]}")
# print(f"Hidden Dimension: {best_hyperparams[1]}")
# print(f"Epochs: {best_hyperparams[2]}")
# print(f"Best F1 Score: {best_f1:.4f}")

gnn = GNNTrainer(hetero_data, "Person")
gnn.train()
gnn.evaluate_test_set()