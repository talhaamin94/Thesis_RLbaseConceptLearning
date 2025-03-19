import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import HeteroData
from TransETrainer import TransETrainer
from RLGraphEnv import RLGraphEnv
from PolicyNetwork import PolicyNetwork
from GNNTrainer import GNNTrainer

# ==================== Logical Expression Generator ====================
def paths_to_logical_expressions(paths, node_mapping):
    logical_expressions = []
    for path in paths:
        if len(path) < 2:
            continue
        owl_classes = [f"∃ connectedTo.{node_mapping[n][0]}{node_mapping[n][1]}" for n in path if n in node_mapping]
        logical_expressions.append(" AND ".join(owl_classes))
    return logical_expressions

# ==================== Train RL Agent ====================
def train_rl_agent_with_owl(hetero_data, num_episodes=100):
    trainer = TransETrainer(hetero_data)
    trainer.train()

    env = RLGraphEnv(hetero_data, trainer)

    state_dim = trainer.transe_model.entity_embeddings.weight.shape[1]
    policy_net = PolicyNetwork(state_dim, len(trainer.node_mapping))

    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    gnn_trainer = GNNTrainer(hetero_data, node_type='A')
    gnn_trainer.train()
    positive_nodes = gnn_trainer.get_positive_nodes()

    if not positive_nodes:
        positive_nodes = list(trainer.global_to_node.keys())

    print("Starting RL with positive nodes:", positive_nodes)

    for episode in range(num_episodes):
        start_node = np.random.choice(positive_nodes)
        state = env.reset(start_node)

        episode_states, episode_actions, rewards = [], [], []

        for _ in range(2):
            neighbors = env.get_neighbors(state)

            if not neighbors:
                action = np.random.choice(list(trainer.global_to_node.keys()))
            else:
                state_emb, _ = env.get_node_embedding(state)
                state_emb = torch.tensor(state_emb, dtype=torch.float32)

                action_probs = policy_net.forward(state_emb).detach().numpy().flatten()

                # Debugging Prints
                # print(f"State {state}: Action probabilities shape: {action_probs.shape}")

                neighbor_indices = [n for n in neighbors if 0 <= n < len(action_probs)]
                # print(f"State {state}: Filtered neighbor indices: {neighbor_indices}")

                if neighbor_indices:
                    valid_probs = action_probs[neighbor_indices]

                    if valid_probs.sum() > 0:
                        valid_probs /= valid_probs.sum()
                        action = np.random.choice(neighbor_indices, p=valid_probs)
                    else:
                        action = np.random.choice(neighbor_indices)
                else:
                    action = np.random.choice(neighbors)

            episode_states.append(state_emb)
            episode_actions.append(action)

            state, done = env.step(action)
            if done:
                break

        if episode_states:
            action_batch = torch.tensor(episode_actions, dtype=torch.long)
            state_batch = torch.stack(episode_states)

            retrieved_nodes = set(env.path)
            reward = (len(set(trainer.node_mapping.keys()) & retrieved_nodes) - 
                      len(retrieved_nodes - set(trainer.node_mapping.keys()))) / len(trainer.node_mapping)
            rewards.append(reward)

            returns = torch.tensor([sum(rewards[i:]) for i in range(len(rewards))], dtype=torch.float32)
            policy_net.update(state_batch, action_batch, returns)

        print(f"Episode {episode + 1}/{num_episodes}: Path Taken = {env.path}, Reward = {reward}")

    logical_expressions = paths_to_logical_expressions(env.tracked_paths, trainer.node_mapping)
    return logical_expressions

# ==================== Run RL Training ====================
logical_expressions = train_rl_agent_with_owl(hetero_data)

for expr in logical_expressions:
    print(expr)
