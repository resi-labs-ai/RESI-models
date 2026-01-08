"""Compute full SHA-256 hash for a model file."""

import hashlib
import sys
from pathlib import Path


def compute_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()  # Full 64 chars


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/compute_hash.py <file_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(compute_hash(path))
