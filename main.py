import argparse

from src.data_generation.main_generator import generate_dataset
from src.training.train import train
from src.evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="AutoOpticalDiagnostics pipeline controller")
    parser.add_argument("--generate", action="store_true", help="Run synthetic data generation")
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation phase")
    parser.add_argument("--run_all", action="store_true", help="Execute the full pipeline (default)")
    args = parser.parse_args()

    if args.run_all or not any([args.generate, args.train, args.evaluate]):
        args.generate = args.train = args.evaluate = True

    if args.generate:
        generate_dataset()
    if args.train:
        train()
    if args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()