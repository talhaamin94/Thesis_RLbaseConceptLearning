import argparse
import os
import json
from src.graphconversion.RDFGraphConverter import RDFGraphConverter
from src.TransETrainer import TransETrainer
from src.gnn.GNNTrainer import GNNTrainer
from src.rlagent.RLTrainer import RLTrainer
from src.Hetero_Data_Walker import HeteroDataWalker
from utils.prepare_data import prepare_dataset






def summarize_results(results, renderer, best_expr, best_reward):
    return {
        "best_expression": renderer.render(best_expr) if best_expr else None,
        "best_reward": best_reward,
        "average_reward": round(sum(r["reward"] for r in results) / len(results), 4) if results else 0.0,
        "all_results": results
    }







def main():
    parser = argparse.ArgumentParser(description="Run RL-based graph explanation and baseline comparison.")
    parser.add_argument("--dataset", type=str, required=True, choices=["AIFB", "MUTAG", "MINI"], help="Dataset name")
    parser.add_argument("--num_walks", type=int, default=0, help="Number of RL/baseline walks to evaluate (if > 0)")
    parser.add_argument("--walk_len", type=int, default=2, help="Length of each walk (RL and baseline)")
    args = parser.parse_args()
    dataset = args.dataset.upper()
    
    # Ensure data exists or is generated
    prepare_dataset(args.dataset)
    
    print(f"Loading dataset: {dataset}")

    converter = RDFGraphConverter(dataset)
    hetero_data = converter.load_dataset()
    print(converter.get_statistics())
    class_prefix, relation_prefix = converter.get_namespace_prefixes()
    # Set the target node type and hidden dimension based on dataset
    if dataset == "MINI":
        node_type = "A"
        hidden_dim = 32
        dropout_rate = 0.3
    elif dataset == "AIFB":
        node_type = "Person"
        hidden_dim = 16
        dropout_rate = 0.3
    elif dataset == "MUTAG":
        node_type = "Compound"
        hidden_dim = 16
        dropout_rate = 0.2
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    transe_trainer = TransETrainer(hetero_data, num_epochs=2000)
    gnn_trainer = GNNTrainer(hetero_data, node_type=node_type, hidden_dim = hidden_dim,dropout_rate=dropout_rate)
    stats= gnn_trainer.run_training()

    print("Best Model Stats:", stats)
    metrics = transe_trainer.train()
    if metrics != False:
        print("Final TransE Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    if args.walk_len > 2:
        num_episodes = 50 * (args.walk_len - 2)
    else:
        num_episodes = 50  # or some default value

    rl_trainer = RLTrainer(
        hetero_data=hetero_data,
        node_type=node_type,
        class_prefix=class_prefix,
        relation_prefix=relation_prefix,
        gnn_trainer=gnn_trainer,
        transe_trainer=transe_trainer,
        num_episodes=num_episodes,
        roll_out=args.walk_len  # RL rollout length = walk length
    )

    rl_trainer.train()

    if args.num_walks > 0:
        print(f"\n--- Evaluating RL and baseline walks (count = {args.num_walks}) ---")

        walker = HeteroDataWalker(
            hetero_data=hetero_data,
            node_type=node_type,
            node_mapping=transe_trainer.node_mapping,
            global_to_node=transe_trainer.global_to_node,
            class_prefix=class_prefix,
            relation_prefix=relation_prefix,
            gnn_trainer=gnn_trainer
        )

        # === Evaluate All RL Paths ===
        # Run RL policy test and evaluate all test-time paths
        rl_expr, rl_reward, rl_path, rl_test_results = rl_trainer.test(num_tests=args.num_walks)
        best_rl_expr, best_rl_reward, _, rl_all_results = walker.evaluate_walks(rl_test_results, rl_trainer, label="RL")



        # === Random Walks (any start) ===
        random_paths = walker.walk(from_positive=False, biased=False, num_walks=args.num_walks, max_len=args.walk_len)
        best_random_expr, best_random_reward, _, random_results = walker.evaluate_walks(random_paths, rl_trainer, label="Random")

        # === Random Walks from Positive Nodes ===
        random_from_positive_paths = walker.walk(from_positive=True, biased=False, num_walks=args.num_walks, max_len=args.walk_len)
        best_random_pos_expr, best_random_pos_reward, _, random_pos_results = walker.evaluate_walks(random_from_positive_paths, rl_trainer, label="RandomFromPositive")

        # === Biased Walks from Positive Nodes ===
        biased_paths = walker.walk(from_positive=True, biased=True, num_walks=args.num_walks, max_len=args.walk_len)
        best_biased_expr, best_biased_reward, _, biased_results = walker.evaluate_walks(biased_paths, rl_trainer, label="Biased")

        # === Save results ===
        os.makedirs("results", exist_ok=True)
        output_path = f"results/{dataset}_initialization_comparison.json"

        with open(output_path, "w") as f:
            json.dump({
                "RL": summarize_results(rl_all_results, rl_trainer.renderer, best_rl_expr, best_rl_reward),
                "Random": summarize_results(random_results, rl_trainer.renderer, best_random_expr, best_random_reward),
                "RandomFromPositive": summarize_results(random_pos_results, rl_trainer.renderer, best_random_pos_expr, best_random_pos_reward),
                "Biased": summarize_results(biased_results, rl_trainer.renderer, best_biased_expr, best_biased_reward)
            }, f, indent=4)

        print(f"\n Saved evaluation results to {output_path}")
    else:
        print("\nSkipping walk evaluation. Use --num_walks to enable RL and baseline testing.")

if __name__ == "__main__":
    main()
