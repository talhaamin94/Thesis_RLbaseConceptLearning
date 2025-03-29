import random
from owlapy.class_expression import OWLObjectSomeValuesFrom, OWLObjectIntersectionOf, OWLClass
from owlapy.owl_property import OWLObjectProperty
from owlapy.iri import IRI
import torch

class HeteroDataWalker:
    def __init__(self, hetero_data, node_type, node_mapping, global_to_node, class_prefix, relation_prefix, gnn_trainer):
        self.data = hetero_data
        self.node_type = node_type
        self.node_mapping = node_mapping
        self.global_to_node = global_to_node
        self.class_prefix = class_prefix
        self.relation_prefix = relation_prefix
        self.gnn_trainer = gnn_trainer

    def _get_valid_start_nodes(self):
        return [i for i in range(self.data[self.node_type].num_nodes) if self.data[self.node_type].train_mask[i]]

    def _sample_edge(self, node_type):
        valid_edges = [et for et in self.data.edge_types if et[0] == node_type]
        return random.choice(valid_edges) if valid_edges else None

    def _get_neighbors(self, src_idx, edge_type):
        edge_index = self.data[edge_type].edge_index
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        return [dst.item() for i, dst in enumerate(dst_nodes) if src_nodes[i].item() == src_idx]

    def _get_node_types(self, node):
        # dummy fallback: assumes only one type available - override as needed
        return [node[0]]

    def _to_logical(self, path):
        parts = []
        for _, edge_type, _, _, dst_local in path:
            role = edge_type[1]
            target_type = dst_local[0]

            prop = OWLObjectProperty(IRI.create(self.relation_prefix + role))
            target = OWLClass(IRI.create(self.class_prefix + target_type))
            parts.append(OWLObjectSomeValuesFrom(prop, target))
        return OWLObjectIntersectionOf(parts) if len(parts) > 1 else (parts[0] if parts else None)

    def walk(self, from_positive=False, biased=False, num_walks=10, max_len=3):
        walks = []
        attempts = 0
        max_total_attempts = num_walks * 20
        positive_nodes = self.gnn_trainer.get_positive_nodes()

        while len(walks) < num_walks and attempts < max_total_attempts:
            attempts += 1

            if from_positive and positive_nodes:
                if biased:
                    # Biased: compute type frequencies
                    type_counts = {}
                    for node in positive_nodes:
                        types = self._get_node_types((self.node_type, node))
                        for t in types:
                            if t != 'Thing':
                                type_counts[t] = type_counts.get(t, 0) + 1
                    total = sum(type_counts.values())
                    type_probs = {t: c / total for t, c in type_counts.items()}

                    valid_start_found = False
                    for _ in range(10):
                        start_local = random.choice(positive_nodes)
                        types = self._get_node_types((self.node_type, start_local))
                        if not types:
                            continue
                        sampled_type = random.choices(
                            population=types,
                            weights=[type_probs.get(t, 0.01) for t in types],
                            k=1
                        )[0]
                        has_edges = any(
                            self._get_neighbors(start_local, et)
                            for et in self.data.edge_types
                            if et[0] == self.node_type
                        )
                        if has_edges:
                            valid_start_found = True
                            break
                    if not valid_start_found:
                        continue
                else:
                    valid_start_found = False
                    for _ in range(10):
                        start_local = random.choice(positive_nodes)
                        has_edges = any(
                            self._get_neighbors(start_local, et)
                            for et in self.data.edge_types
                            if et[0] == self.node_type
                        )
                        if has_edges:
                            valid_start_found = True
                            break
                    if not valid_start_found:
                        continue
            else:
                start_local = random.randint(0, self.data[self.node_type].num_nodes - 1)

            src_type = self.node_type
            src_id = start_local
            path = []

            for _ in range(max_len):
                edge_type = self._sample_edge(src_type)
                if not edge_type:
                    break

                neighbors = self._get_neighbors(src_id, edge_type)
                if not neighbors:
                    break

                if biased:
                    scored = [(nid, len(self._get_neighbors(nid, edge_type))) for nid in neighbors]
                    total = sum(score for _, score in scored)
                    weights = [score / total if total > 0 else 1 / len(scored) for _, score in scored]
                    dst_id = random.choices([nid for nid, _ in scored], weights=weights, k=1)[0]
                else:
                    dst_id = random.choice(neighbors)

                dst_type = edge_type[2]
                dst_node = (dst_type, dst_id)

                path.append(((src_type, src_id), edge_type, (dst_type, dst_id), None, dst_node))

                src_type, src_id = dst_type, dst_id

            if path:
                walks.append(path)

        if len(walks) < num_walks:
            print(f"[WARN] Only {len(walks)} / {num_walks} walks were generated after {attempts} attempts.")

        return walks

    def convert_paths_to_expressions(self, paths):
        return [self._to_logical(p) for p in paths if p]

    def evaluate_walks(self, typed_paths, rl_trainer, label=""):
        best_expr = None
        best_reward = -1
        best_path = None
        all_results = []

        print(f"\n--- Evaluating {len(typed_paths)} {label} walks ---")

        for i, path in enumerate(typed_paths):
            reward, expr = rl_trainer.calculate_reward(path)
            readable = rl_trainer.renderer.render(expr) if expr else "None"

            # print(f"[{label}] Walk {i+1}: Reward = {reward:.4f}, Expression = {readable}")

            all_results.append({
                "index": i,
                "reward": reward,
                "expression": readable
            })

            if reward > best_reward:
                best_reward = reward
                best_expr = expr
                best_path = path

        if best_expr:
            print(f"\n Best {label} Expression:\n{rl_trainer.renderer.render(best_expr)}\nReward = {best_reward:.4f}")
        else:
            print(f"\n No valid expression found for {label} walks.")

        return best_expr, best_reward, best_path, all_results
