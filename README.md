# Concept-Based Explanations for Graph Neural Networks:
## Integrating Evolutionary Algorithms and Reinforcement Learning

This repository presents the implementation for a Master’s thesis project that integrates **Reinforcement Learning (RL)** and **Evolutionary Algorithms** to generate interpretable explanations for predictions made by **Graph Neural Networks (GNNs)**.

Instead of relying solely on black-box attributions, this project focuses on **generating logical class expressions** (e.g., OWL DL formulas) that explain why certain nodes belong to a predicted class. These expressions are learned by an RL agent through pathfinding over a knowledge graph trained with TransE and evaluated via a GNN.

---

## Key Features

- GNNs trained using **FASTRGCN** on RDF-based knowledge graphs.
- Uses **TransE** embeddings for entity and relation representations.
- RL agent based on **REINFORCE** that generates OWL logical expressions from paths.
- Supports **multiple datasets**: AIFB, MUTAG, and a synthetic MINI dataset.
- Comparison of RL-based explanation paths vs. random and biased walk initializations.
- Explanations rendered in **DL Syntax** via OWLAPI Python bindings (`owlapy`).

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/talhaamin94/Thesis_RLbaseConceptLearning.git
```
### 2. Install Packages

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.10+ and CUDA if running on GPU.

## Running experiments
```bash
python run.py --dataset [AIFB|MUTAG|MINI] --num_walks 10 --walk_len 2
```
### Arguments

- `--dataset`: Choose between `AIFB`, `MUTAG`, or `MINI`.
- `--num_walks`: Number of random/baseline/RL walks to compare.
- `--walk_len`: Maximum walk length for RL agent and baselines.

### Examples
Train and evaluate on AIFB
```bash
python run.py --dataset AIFB --num_walks 10 --walk_len 2
```
Train on AIFB without experiments:
```bash
python run.py --dataset AIFB

```

## Dataset Details

You do **not** need to manually download any datasets.

- **AIFB** and **MUTAG** are automatically downloaded when you run the project the first time, using [`torch_geometric.datasets.Entities`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Entities.html).
- The `.nt.gz` RDF files are extracted and prepared internally.
- **MINI** is a synthetic RDF dataset generated using a custom graph generator with labeled examples.

---

## Dataset References

- **AIFB**:  
  [RDF Benchmarking for Machine Learning](https://link.springer.com/chapter/10.1007/978-3-319-46547-0_4)

- **MUTAG**:  
  [Toxicity Prediction Using Graph Kernels](https://link.springer.com/chapter/10.1007/3-540-44673-7_34)

## Output

Results of walk-based explanation evaluations are saved in:

results/{DATASET}_initialization_comparison.json

Each result file includes:

- Best logical expression
- Final reward
- Average reward
- Full list of tested paths




