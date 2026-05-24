from tokenizer.tokenizer import Tokenizer
from data.data_loader import load_wikitext, load_tinystories
import argparse


def load_dataset(dataset: str):

    if dataset == "wiki":
        return load_wikitext()

    elif dataset == "stories":
        return load_tinystories()

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train_tokenizer(text, save_path, num_merges):

    tokenizer = Tokenizer()

    print("Training tokenizer...")
    tokenizer.train(text, num_merges=num_merges)

    print("Saving tokenizer...")
    tokenizer.save(save_path)

    print(f"Training complete. Saved to {save_path}")


def train_from_dataset(dataset, save_path, num_merges):

    text = load_dataset(dataset)
    train_tokenizer(text, save_path, num_merges)


def train_from_file(file_path, save_path, num_merges):

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    train_tokenizer(text, save_path, num_merges)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["wiki", "stories"])
    parser.add_argument("--file", type=str, help="Train from custom file")
    parser.add_argument("--merges", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="saved")

    args = parser.parse_args()

    if args.file:
        train_from_file(args.file, args.save_path, args.merges)

    elif args.dataset:
        train_from_dataset(args.dataset, args.save_path, args.merges)

    else:
        print("You must provide either --dataset or --file")