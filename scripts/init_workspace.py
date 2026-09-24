"""Create ignored runtime directories, relative to this repository."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "data/raw", "data/processed", "data/priors",
        "checkpoints", "outputs", "logs",
    ):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {path}")


if __name__ == "__main__":
    main()
