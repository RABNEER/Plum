"""Export the trusted local training checkpoint for the Python package."""

import json
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parent


def main():
    checkpoint = torch.load(PACKAGE.parent / "referee.pt", weights_only=True, map_location="cpu")
    weights = {name: tensor.tolist() for name, tensor in checkpoint["sbm"].items()}
    target = PACKAGE / "src" / "plum_sbm" / "weights.json"
    target.write_text(json.dumps(weights, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
