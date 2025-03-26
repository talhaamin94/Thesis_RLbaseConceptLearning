import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define TransE Model
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        # self.relation_scaling = nn.Parameter(torch.ones(num_relations))  # Learnable scaling per relation

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation):
        head_emb = self.entity_embeddings(head)
        # relation_emb = self.relation_embeddings(relation) * self.relation_scaling[relation].unsqueeze(1)  # Apply scaling
        relation_emb = self.relation_embeddings(relation)
        return head_emb + relation_emb


    def score(self, head, relation, tail):
        h_r = self.forward(head, relation)
        t_emb = self.entity_embeddings(tail)

        return -torch.norm(h_r - t_emb, p=1, dim=1)  # Use L1 norm, but remove in-function normalization



class TransETrainer:
    def __init__(self, hetero_data, embedding_dim=128, num_epochs=2000, batch_size=1024, lr=0.002):
        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.node_embeddings_path = os.path.join(self.data_dir, "node_embeddings.csv")
        self.relation_embeddings_path = os.path.join(self.data_dir, "relation_embeddings.csv")
        self.edge_embeddings_path = os.path.join(self.data_dir, "edge_embeddings.csv")
        self.metrics_path = os.path.join(self.data_dir, "transe_metrics.txt")
        self.hetero_data = hetero_data
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.unique_nodes, self.unique_edges, self.node_to_index, self.relation_to_index = self._extract_graph_info()
        self.num_entities = len(self.unique_nodes)
        self.num_relations = len(self.hetero_data.edge_types)
        self.transe_model = TransE(self.num_entities, self.num_relations, self.embedding_dim)
        self.optimizer = optim.Adam(self.transe_model.parameters(), lr=self.lr, weight_decay=1e-4)  # L3 regularization
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.loss_fn = nn.MarginRankingLoss(margin=3.0)
    
    def _extract_graph_info(self):
        """Extracts unique nodes, edges, and their mappings."""
        unique_nodes = set()
        self.node_mapping = {}  # (node_type, local_index) -> global_index
        self.global_to_node = {}  # global_index -> (node_type, local_index)
        global_index = 0  
        person_nodes = []
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].x.shape[0] if hasattr(self.hetero_data[node_type], "x") else 0
            for local_idx in range(num_nodes):
                self.node_mapping[(node_type, local_idx)] = global_index  # Forward mapping
                self.global_to_node[global_index] = (node_type, local_idx)  # Reverse mapping
                unique_nodes.add(global_index)
                global_index += 1  # Increment global index
        unique_nodes = sorted(list(unique_nodes))  # Ensure sorted list
        node_to_index = {node: i for i, node in enumerate(unique_nodes)}
        # print(f"Person nodes: {person_nodes}")
        unique_edges = []
        for edge_type in self.hetero_data.edge_types:
            edge_index = self.hetero_data[edge_type].edge_index

            for i in range(edge_index.shape[1]):
                src_local = edge_index[0, i].item()
                tgt_local = edge_index[1, i].item()

                if (edge_type[0], src_local) in self.node_mapping and (edge_type[2], tgt_local) in self.node_mapping:
                    src_global = self.node_mapping[(edge_type[0], src_local)]
                    tgt_global = self.node_mapping[(edge_type[2], tgt_local)]
                    unique_edges.append((src_global, edge_type, tgt_global))  # Store full tuple

        relation_to_index = {rel: i for i, rel in enumerate(self.hetero_data.edge_types)}  # Store full tuple

        return unique_nodes, unique_edges, node_to_index, relation_to_index

    def get_embedding(self, global_index):
        """Retrieve TransE embedding and original node name for a given global index."""
        if global_index not in self.global_to_node:
            raise ValueError(f"Global index {global_index} not found in mapping.")

        node_type, local_index = self.global_to_node[global_index]
        embedding = self.transe_model.entity_embeddings(torch.tensor([global_index])).detach().numpy()
        
        return embedding, f"{node_type}{local_index}"

    def _save_embeddings(self):
        """Saves node, relation, and edge embeddings to CSV files."""
        print("Saving embeddings to disk...")

        node_indices = torch.arange(self.num_entities)
        node_embeddings = self.transe_model.entity_embeddings(node_indices).detach().numpy()
        node_df = pd.DataFrame(node_embeddings, index=self.unique_nodes)
        node_df.to_csv(self.node_embeddings_path)

        relation_indices = torch.arange(self.num_relations)
        relation_embeddings = self.transe_model.relation_embeddings(relation_indices).detach().numpy()
        relation_df = pd.DataFrame(relation_embeddings, index=self.hetero_data.edge_types)
        relation_df.to_csv(self.relation_embeddings_path)

        edge_embeddings = []
        edge_index_list = []
        for (head, relation, tail) in self.unique_edges:
            head_idx = torch.tensor(self.node_to_index[head])
            relation_idx = torch.tensor(self.relation_to_index[relation])
            tail_idx = torch.tensor(self.node_to_index[tail])

            head_emb = self.transe_model.entity_embeddings(head_idx)
            relation_emb = self.transe_model.relation_embeddings(relation_idx)
            tail_emb = self.transe_model.entity_embeddings(tail_idx)

            edge_embedding = (head_emb + relation_emb - tail_emb).detach().numpy()
            edge_embeddings.append(edge_embedding)
            edge_index_list.append(f"{head}-{relation}-{tail}")

        edge_df = pd.DataFrame(edge_embeddings, index=edge_index_list)
        edge_df.to_csv(self.edge_embeddings_path)

        print(f"Embeddings saved:\n - Nodes: {self.node_embeddings_path}\n - Relations: {self.relation_embeddings_path}\n - Edges: {self.edge_embeddings_path}")
    
    def _evaluate(self):
        self.normalize_embeddings()
        self.transe_model.eval()
        ranks = []

        for (head, relation, tail) in self.unique_edges:
            head_idx = torch.tensor([self.node_to_index[head]])
            relation_idx = torch.tensor([self.relation_to_index[relation]])
            tail_idx = torch.tensor([self.node_to_index[tail]])

            correct_score = self.transe_model.score(head_idx, relation_idx, tail_idx).item()
            all_tail_indices = torch.arange(self.num_entities)
            all_scores = self.transe_model.score(head_idx.repeat(self.num_entities), relation_idx.repeat(self.num_entities), all_tail_indices)

            sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
            rank = (sorted_indices == tail_idx).nonzero(as_tuple=True)

            if len(rank[0]) > 0:  # Check if tensor inside tuple is non-empty
                ranks.append(rank[0][0].item() + 1)  #  Extract first tensor and get rank value
            else:
                ranks.append(self.num_entities)  # Worst case: rank at the end


        MR = np.mean(ranks)
        MRR = np.mean([1.0 / r for r in ranks])
        Hits_1 = np.mean([1 if r <= 1 else 0 for r in ranks])
        Hits_3 = np.mean([1 if r <= 3 else 0 for r in ranks])
        Hits_10 = np.mean([1 if r <= 10 else 0 for r in ranks])

        with open(self.metrics_path, "w") as f:
            f.write(f"Mean Rank (MR): {MR:.2f}\n")
            f.write(f"Mean Reciprocal Rank (MRR): {MRR:.4f}\n")
            f.write(f"Hits@1: {Hits_1:.4f}\n")
            f.write(f"Hits@3: {Hits_3:.4f}\n")
            f.write(f"Hits@10: {Hits_10:.4f}\n")


        print(f"Evaluation complete. MRR: {MRR:.4f}, Hits@1: {Hits_1:.4f}, Hits@10: {Hits_10:.4f}")

        return MRR

    def train(self):
        print("Training TransE")

        train_edges = torch.tensor([(self.node_to_index[h], self.relation_to_index[r], self.node_to_index[t]) 
                                    for (h, r, t) in self.unique_edges])

        for epoch in range(self.num_epochs):  
            self.transe_model.train()
            self.optimizer.zero_grad()

            # Select batch samples
            idx = torch.randint(0, train_edges.shape[0], (self.batch_size,))
            pos_triplets = train_edges[idx]
            neg_triplets = self.generate_negative_samples(pos_triplets)  # Better negative sampling

            # Compute positive & negative scores
            pos_scores = self.transe_model.score(pos_triplets[:, 0], pos_triplets[:, 1], pos_triplets[:, 2])
            neg_scores = self.transe_model.score(neg_triplets[:, 0], neg_triplets[:, 1], neg_triplets[:, 2])

            # Compute loss
            # With this standard margin ranking loss:
            loss = self.loss_fn(pos_scores, neg_scores, torch.ones_like(pos_scores))
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.transe_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # self.normalize_embeddings()
            # Reduce learning rate if needed
            # self.scheduler.step(loss.item())  # Ensure loss is converted to float

            # Print loss every 10 epochs  (low overhead)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")



        # Final MRR evaluation after 500 epochs 
        final_mrr = self._evaluate()
        print(f"Final MRR after training: {final_mrr:.4f}")


    
    
    def generate_negative_samples(self, train_edges):
        neg_triplets = train_edges.clone()
        corrupt_head = torch.rand(train_edges.shape[0]) > 0.5  # 50% chance to corrupt head or tail

        # Corrupt head or tail accordingly
        neg_triplets[corrupt_head, 0] = torch.randint(0, self.num_entities, (corrupt_head.sum(),))
        neg_triplets[~corrupt_head, 2] = torch.randint(0, self.num_entities, ((~corrupt_head).sum(),))
        
        return neg_triplets

    # def generate_adversarial_negatives(self, train_edges, alpha=0.5):
    #     neg_triplets = train_edges.clone()
    #     corrupt_head = torch.rand(train_edges.shape[0]) > 0.5  # 50% corrupt head, 50% corrupt tail

    #     for i in range(len(neg_triplets)):
    #         if corrupt_head[i]:  
    #             neg_triplets[i, 0] = torch.randint(0, self.num_entities, (1,))
    #         else:
    #             neg_triplets[i, 2] = torch.randint(0, self.num_entities, (1,))

    #     neg_scores = self.transe_model.score(neg_triplets[:, 0], neg_triplets[:, 1], neg_triplets[:, 2])
        
    #     # Softmax weighting to emphasize harder negatives
    #     neg_weights = torch.softmax(-alpha * neg_scores, dim=0)
    #     selected_neg_idx = torch.multinomial(neg_weights, len(neg_triplets), replacement=True)

    #     return neg_triplets[selected_neg_idx]

    def normalize_embeddings(self):
        with torch.no_grad():
            self.transe_model.entity_embeddings.weight /= torch.norm(
                self.transe_model.entity_embeddings.weight, p=2, dim=1, keepdim=True
            )
    # def generate_negative_samples(self, train_edges):
    #     neg_triplets = train_edges.clone()
    #     corrupt_head = torch.rand(train_edges.shape[0]) > 0.5  # 50% chance to corrupt head or tail

    #     for i in range(len(neg_triplets)):
    #         if corrupt_head[i]:  
    #             neg_triplets[i, 0] = torch.randint(0, self.num_entities, (1,))
    #         else:
    #             neg_triplets[i, 2] = torch.randint(0, self.num_entities, (1,))

    #         # Ensure negatives are actually incorrect
    #         while tuple(neg_triplets[i].tolist()) in self.unique_edges:
    #             if corrupt_head[i]:
    #                 neg_triplets[i, 0] = torch.randint(0, self.num_entities, (1,))
    #             else:
    #                 neg_triplets[i, 2] = torch.randint(0, self.num_entities, (1,))

    #     return neg_triplets
