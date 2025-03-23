import os
import json
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt

from Evaluator import Evaluator
from TransETrainer import TransETrainer
from GNNTrainer import GNNTrainer
from RLGraphEnv import RLGraphEnv
from PolicyNetwork import PolicyNetwork
from owlapy.class_expression import OWLClass, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf
from owlapy.owl_property import OWLObjectProperty
from owlapy.iri import IRI
from owlapy.render import ManchesterOWLSyntaxOWLObjectRenderer

class RLTrainer:
    def __init__(self, hetero_data, node_type,class_prefix, relation_prefix,num_episodes=100, roll_out=3):
        self.class_prefix = class_prefix
        self.relation_prefix = relation_prefix
        self.hetero_data = hetero_data
        self.node_type = node_type
        self.num_episodes = num_episodes
        self.roll_out = roll_out  # New: Configurable rollout value
        self.trainer = TransETrainer(hetero_data)
        self.gnn_trainer = GNNTrainer(hetero_data, node_type=node_type)

        self.trainer.train()
        self.gnn_trainer.run_training()

        self.env = RLGraphEnv(hetero_data, self.trainer)

        self.state_dim = self.trainer.transe_model.entity_embeddings.weight.shape[1] * 2
        self.policy_net = PolicyNetwork(self.state_dim, len(self.trainer.node_mapping))
        self.renderer = ManchesterOWLSyntaxOWLObjectRenderer()

        # self.positive_nodes = self.gnn_trainer.get_positive_nodes()
        self.positive_nodes = {
            self.trainer.node_mapping[(self.node_type, local_id)]
            for local_id in self.gnn_trainer.get_positive_nodes()
        }
        print(self.positive_nodes)
        if not self.positive_nodes:
            self.positive_nodes = list(self.trainer.global_to_node.keys())

        self.episode_rewards = []
        self.run_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "episodes": []
        }
    def calculate_logical_expression(self):
        expressions = []

        for path in self.env.tracked_paths:
            if len(path) < 1:
                continue

            parts = []
            for src, edge_type, dst, src_local, dst_local in path:
                relation = edge_type[1]           # e.g., "related_to"
                target_type = dst_local[0]        # e.g., "C"

                prop_iri = IRI.create(self.relation_prefix + relation)
                class_iri = IRI.create(self.class_prefix + target_type)

                prop = OWLObjectProperty(prop_iri)
                target = OWLClass(class_iri)
                parts.append(OWLObjectSomeValuesFrom(prop, target))

            if parts:
                expression = OWLObjectIntersectionOf(parts) if len(parts) > 1 else parts[0]
                expressions.append(expression)

        return expressions


    def calculate_reward(self, typed_path):
        """
        Computes the reward for the current episode's typed_path.
        """
        labeled_nodeset = set(self.gnn_trainer.get_positive_nodes()) | set(self.gnn_trainer.get_negative_nodes())
        evaluator = Evaluator(self.hetero_data, labeled_nodeset=labeled_nodeset)

        logical_expr = self.build_logical_expression(typed_path)  # <- Use only the current path

        if logical_expr is None:
            return 0.0

        precision, recall, accuracy = evaluator.explanation_accuracy(
            set(self.gnn_trainer.get_positive_nodes()), logical_expr
        )
        rendered_expr = self.renderer.render(logical_expr)
        print("Typed path used for reward:", typed_path)
        print("Generated expression:", rendered_expr)
        return accuracy


    def train(self):
        print("Starting RL with positive nodes:", self.positive_nodes)

        for episode in range(self.num_episodes):
            # Pick a valid starting node that has neighbors
            while True:
                start_node = np.random.choice(list(self.positive_nodes))
                neighbors = self.env.get_neighbors(start_node)
                if neighbors:
                    break

            self.env.reset(start_node)

            episode_states, episode_actions, rewards = [], [], []

            for _ in range(self.roll_out):
                current_state = self.env.get_state_embedding()
                neighbors = self.env.get_neighbors(self.env.current_node)

                if not neighbors:
                    print(f"Dead-end reached at node {self.env.current_node}. Ending episode early.")
                    break

                neighbor_nodes = [n for n, _ in neighbors]
                action_probs = self.policy_net.forward(current_state).detach().numpy().flatten()
                neighbor_indices = [n for n in neighbor_nodes if 0 <= n < len(action_probs)]

                if neighbor_indices:
                    valid_probs = action_probs[neighbor_indices]
                    if valid_probs.sum() > 0:
                        valid_probs /= valid_probs.sum()
                        chosen_node = np.random.choice(neighbor_indices, p=valid_probs)
                    else:
                        chosen_node = np.random.choice(neighbor_indices)
                else:
                    chosen_node = np.random.choice(neighbor_nodes)

                # Get edge type for the chosen neighbor
                action_node, action_edge = next((n, et) for n, et in neighbors if n == chosen_node)

                episode_states.append(current_state)
                episode_actions.append(action_node)

                next_state, _ = self.env.step(action_node, action_edge)  # Let rollout length decide end

            # After rollout ends, compute reward and update
            if episode_states:
                action_batch = torch.tensor(episode_actions, dtype=torch.long)
                state_batch = torch.stack(episode_states)

                reward = self.calculate_reward(self.env.typed_path)
                rewards.append(reward)

                returns = torch.tensor([sum(rewards[i:]) for i in range(len(rewards))], dtype=torch.float32)
                self.policy_net.update(state_batch, action_batch, returns)

                self.episode_rewards.append(reward)
                self.run_log["episodes"].append({
                    "typed_path": self.env.typed_path.copy(),
                    "episode": episode + 1,
                    "path": self.env.path.copy(),
                    "reward": reward
                })

                # Add final typed_path to tracked_paths
                self.env.tracked_paths.append(self.env.typed_path.copy())

            print(f"Episode {episode + 1}/{self.num_episodes}: Path Taken = {self.env.path}, Reward = {reward}")

        # Save logical expressions
        self.get_logical_expressions()
        self.save_logical_expressions()




    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label='Reward per Episode')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("RL Agent Rewards Over Episodes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_run_log(self, output_dir="logs", filename_prefix="rl_run"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = self.run_log["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(self.run_log, f, indent=4)

        print(f"Run log saved to {filepath}")

    def get_logical_expressions(self):
        expressions = self.calculate_logical_expression()
        print("\nGenerated Logical Expressions (Rendered):")
        for index, expr in enumerate(expressions):
            readable = self.renderer.render(expr)
            print(f"Expression {index + 1}: {readable}")
        return expressions

    def save_logical_expressions(self, output_dir="logs", filename_prefix="logical_expressions"):
        import os
        import json
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        expressions = self.get_logical_expressions()

        # Convert OWL expressions to strings
        stringified_expressions = [str(expr) for expr in expressions]

        with open(filepath, "w") as file:
            json.dump(stringified_expressions, file, indent=4)

        print(f"Logical expressions saved to {filepath}")

    def build_logical_expression(self, typed_path):
        """
        Builds a single OWL logical expression from a single typed path.
        """
        if not typed_path:
            return None

        parts = []
        for _, edge_type, _, _, dst_local in typed_path:
            relation = edge_type[1]
            target_type = dst_local[0]

            prop_iri = IRI.create(self.relation_prefix + relation)
            class_iri = IRI.create(self.class_prefix + target_type)

            prop = OWLObjectProperty(prop_iri)
            target = OWLClass(class_iri)

            parts.append(OWLObjectSomeValuesFrom(prop, target))

        if not parts:
            return None

        return OWLObjectIntersectionOf(parts) if len(parts) > 1 else parts[0]
