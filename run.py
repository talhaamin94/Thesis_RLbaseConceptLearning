# run.py
import argparse
from src.RDFGraphConverter import RDFGraphConverter
from src.RLTrainer import RLTrainer  # example: adapt based on actual use

def main():
    parser = argparse.ArgumentParser(description="Run RL-based graph explanation.")
    parser.add_argument("--dataset", type=str, required=True, choices=["AIFB", "MUTAG"], help="Dataset name")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    converter = RDFGraphConverter(args.dataset)
    hetero_data = converter.load_dataset()

    stats = converter.get_statistics()
    print("Dataset Statistics:", stats)

    # Example training
    trainer = RLTrainer(hetero_data)
    trainer.train()

if __name__ == "__main__":
    main()
