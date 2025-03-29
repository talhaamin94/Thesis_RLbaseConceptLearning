import os
import json
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt

from utils.Evaluator import Evaluator

from .RLGraphEnv import RLGraphEnv
from .PolicyNetwork import PolicyNetwork
from owlapy.class_expression import OWLClass, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf
from owlapy.owl_property import OWLObjectProperty
from owlapy.iri import IRI
from owlapy.render import DLSyntaxObjectRenderer



class RLTrainer:
    def __init__(self, hetero_data, node_type, class_prefix, relation_prefix, gnn_trainer, transe_trainer, num_episodes=100, roll_out=2):
        self.class_prefix = class_prefix
        self.relation_prefix = relation_prefix
        self.hetero_data = hetero_data
        self.node_type = node_type
        self.num_episodes = num_episodes
        self.roll_out = roll_out  # New: Configurable rollout value

        self.gnn_trainer = gnn_trainer
        self.trainer = transe_trainer
        self.env = RLGraphEnv(hetero_data, self.trainer)

        self.state_dim = self.trainer.transe_model.entity_embeddings.weight.shape[1] * 2
        self.policy_net = PolicyNetwork(self.state_dim, len(self.trainer.node_mapping))
        self.renderer = DLSyntaxObjectRenderer()

        # self.positive_nodes = self.gnn_trainer.get_positive_nodes()

        # print(self.positive_nodes)
        # if not self.positive_nodes:
        #     self.positive_nodes = list(self.trainer.global_to_node.keys())
        self.local_positive_nodes = set(self.gnn_trainer.get_positive_nodes())
        self.global_positive_nodes = {
            self.trainer.node_mapping[(self.node_type, local_id)]
            for local_id in self.local_positive_nodes
        }
       
        self.episode_rewards = []
        self.run_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "episodes": []
        }


    def calculate_reward(self, typed_path):
        labeled_nodes = set(self.gnn_trainer.get_positive_nodes()) | set(self.gnn_trainer.get_negative_nodes())
        evaluator = Evaluator(self.hetero_data, labeled_nodeset=labeled_nodes)
        
        logical_expr = self.build_logical_expression(typed_path)
        if logical_expr is None:
            print("No logical expression generated.")
            return 0.0, None

        # Evaluate and normalize expr_nodes
        raw_expr_nodes = evaluator._eval_formula(logical_expr)
        expr_nodes = {(int(idx), ntype) for idx, ntype in raw_expr_nodes}  # <-- Normalize np.int64 to int

        positive_nodes = {(int(idx), self.node_type) for idx in self.local_positive_nodes}
        negative_nodes = {(int(idx), self.node_type) for idx in self.gnn_trainer.get_negative_nodes()}

        # Continue as usual
        true_positives = expr_nodes & positive_nodes
        true_negatives = negative_nodes - expr_nodes
        false_positives = expr_nodes & negative_nodes
        false_negatives = positive_nodes - expr_nodes

        reward = round((len(true_positives) + len(true_negatives)) / len(labeled_nodes),4)

        # Debug Output
        rendered_expr = self.renderer.render(logical_expr)
        # print(f"Expression matched nodes: {len(expr_nodes)} -> {expr_nodes}")
        # print(f"Positive nodes: {len(positive_nodes)} -> {positive_nodes}")
        # print(f"Negative nodes: {len(negative_nodes)} -> {negative_nodes}")
        # print(f"True Positives (TP): {len(true_positives)}")
        # print(f"True Negatives (TN): {len(true_negatives)}")
        # print(f"False Positives (FP): {len(false_positives)}")
        # print(f"False Negatives (FN): {len(false_negatives)}")
        # print(f"Total Labeled Nodes: {len(labeled_nodes)}")
        # print(f"Reward: {reward:.4f}")
        # print(f"------------------------\n")

        return reward, logical_expr


    def train(self):
        # print("Starting RL with positive nodes:", self.positive_nodes)

        for episode in range(self.num_episodes):
            # Pick a valid starting node that has neighbors
            while True:
                start_node = int(np.random.choice(list(self.global_positive_nodes)))

                
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

                self.env.step(action_node, action_edge)

            # After rollout ends, compute reward and update
            if episode_states:
                action_batch = torch.tensor(episode_actions, dtype=torch.long)
                state_batch = torch.stack(episode_states)

                reward, expr = self.calculate_reward(self.env.typed_path)

                rewards.append(reward)

                returns = torch.tensor([sum(rewards[i:]) for i in range(len(rewards))], dtype=torch.float32)
                self.policy_net.update(state_batch, action_batch, returns)

                self.episode_rewards.append(reward)
                self.run_log["episodes"].append({
                    # "typed_path": self.env.typed_path.copy(),
                    "episode": episode + 1,
                    # "path": self.env.path.copy(),
                    "reward": reward
                })

                # Add final typed_path to tracked_paths
                self.env.tracked_paths.append(self.env.typed_path.copy())

            readable_expr = self.renderer.render(expr) if expr else "None"
            # print(f"Episode {episode + 1}/{self.num_episodes}: Expression = {readable_expr}, Reward = {reward}")

        # Save logical expressions
        # self.get_logical_expressions()
        # self.save_logical_expressions()




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
        Builds a nested OWL logical expression from a single typed path.
        """
        if not typed_path:
            # print("[DEBUG] Empty typed_path — returning None")
            return None

        # print("[DEBUG] Building nested logical expression from typed path:")
        
        # Start from the last node and build nested expressions backwards
        expr = None
        for step in reversed(typed_path):
            src, edge_type, dst, src_local, dst_local = step

            relation = edge_type[1]
            target_type = dst_local[0]

            prop_iri = self.relation_prefix + relation
            class_iri = self.class_prefix + target_type

            # print(f" - Relation: {relation} -> IRI: {prop_iri}")
            # print(f" - Target Type: {target_type} -> IRI: {class_iri}")

            prop = OWLObjectProperty(IRI.create(prop_iri))
            target_class = OWLClass(IRI.create(class_iri))

            if expr is None:
                expr = OWLObjectSomeValuesFrom(prop, target_class)
            else:
                # Nest the previous expression inside the current one
                expr = OWLObjectSomeValuesFrom(prop, OWLObjectIntersectionOf([target_class, expr]))

            # print(f"   Current Nested Expression: {self.renderer.render(expr)}")

        # print(f"[DEBUG] Final Nested Expression: {self.renderer.render(expr)}")
        return expr

    def test(self, num_tests=10):
        print("\n--- Testing Learned Policy ---")
        best_reward = -1
        best_expression = None
        best_path = None
        generated_paths = []  # ⬅️ Store typed paths here
        all_expressions = []

        for test_num in range(num_tests):
            while True:
                start_node = int(np.random.choice(list(self.global_positive_nodes)))
                neighbors = self.env.get_neighbors(start_node)
                if neighbors:
                    break

            self.env.reset(start_node)

            for _ in range(self.roll_out):
                current_state = self.env.get_state_embedding()
                neighbors = self.env.get_neighbors(self.env.current_node)

                if not neighbors:
                    break

                action_probs = self.policy_net.forward(current_state).detach().numpy().flatten()
                neighbor_nodes = [n for n, _ in neighbors]
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

                action_node, action_edge = next((n, et) for n, et in neighbors if n == chosen_node)
                self.env.step(action_node, action_edge)

            reward, expr = self.calculate_reward(self.env.typed_path)
            readable_expr = self.renderer.render(expr) if expr else "None"
            # print(f"Test {test_num + 1}/{num_tests}: Expression = {readable_expr}, Reward = {reward:.4f}")

            all_expressions.append(expr)
            generated_paths.append(self.env.typed_path.copy())

            if reward > best_reward:
                best_reward = reward
                best_expression = expr
                best_path = self.env.typed_path.copy()

        final_expr = self.renderer.render(best_expression) if best_expression else "None"
        # print(f"\n=== Best Expression During Testing ===\n{final_expr}\nReward = {best_reward:.4f}")

        return best_expression, best_reward, best_path, generated_paths

