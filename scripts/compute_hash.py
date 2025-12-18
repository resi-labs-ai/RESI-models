"""Compute 8-char SHA-1 hash for a model file."""

import hashlib
import sys
from pathlib import Path


def compute_hash(file_path: Path) -> str:
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()[:8]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/compute_hash.py <file_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(compute_hash(path))
