"""Disk cache management for ONNX models."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from .models import CachedModel, CachedModelMetadata

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.onnx"
METADATA_FILENAME = "metadata.json"


class ModelCache:
    """
    Manage disk cache of downloaded ONNX models.

    Cache structure:
        cache_dir/
        ├── {hotkey_1}/
        │   ├── model.onnx
        │   └── metadata.json
        ├── {hotkey_2}/
        │   └── ...

    Only stores hash in metadata for cache invalidation.
    Full commitment data comes from chain via Pylon.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize model cache.

        Args:
            cache_dir: Directory to store cached models
        """
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelCache ready at {self._cache_dir}")

    def get(self, hotkey: str) -> CachedModel | None:
        """
        Get cached model if exists and valid.

        Args:
            hotkey: Miner's hotkey

        Returns:
            CachedModel if found and valid, None otherwise
        """
        hotkey_dir = self._cache_dir / hotkey
        model_path = hotkey_dir / MODEL_FILENAME
        metadata_path = hotkey_dir / METADATA_FILENAME

        if not model_path.exists() or not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                metadata = CachedModelMetadata.from_dict(json.load(f))
            return CachedModel(path=model_path, metadata=metadata)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Corrupted metadata for {hotkey}: {e}")
            return None

    def is_valid(self, hotkey: str, expected_hash: str) -> bool:
        """
        Check if cached model matches expected hash.

        Args:
            hotkey: Miner's hotkey
            expected_hash: Hash from chain commitment

        Returns:
            True if cached and hash matches
        """
        cached = self.get(hotkey)
        if cached is None:
            return False
        return cached.metadata.hash == expected_hash

    def put(
        self,
        hotkey: str,
        temp_model_path: Path,
        model_hash: str,
        size_bytes: int,
    ) -> Path:
        """
        Store model in cache (atomic move from temp).

        Args:
            hotkey: Miner's hotkey
            temp_model_path: Path to downloaded model in temp location
            model_hash: Verified hash of the model
            size_bytes: Size of model file

        Returns:
            Path to cached model
        """
        hotkey_dir = self._cache_dir / hotkey
        hotkey_dir.mkdir(parents=True, exist_ok=True)

        model_path = hotkey_dir / MODEL_FILENAME
        metadata_path = hotkey_dir / METADATA_FILENAME

        # Atomic move from temp to cache
        shutil.move(str(temp_model_path), str(model_path))

        # Write metadata
        metadata = CachedModelMetadata(
            hash=model_hash,
            size_bytes=size_bytes,
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f)

        logger.info(f"Cached model for {hotkey} ({size_bytes} bytes)")
        return model_path

    def remove(self, hotkey: str) -> bool:
        """
        Remove model from cache.

        Args:
            hotkey: Miner's hotkey

        Returns:
            True if removed, False if not found
        """
        hotkey_dir = self._cache_dir / hotkey
        if hotkey_dir.exists():
            shutil.rmtree(hotkey_dir)
            logger.info(f"Removed cached model for {hotkey}")
            return True
        return False

    def get_all_hotkeys(self) -> list[str]:
        """List all cached hotkeys."""
        return [
            d.name
            for d in self._cache_dir.iterdir()
            if d.is_dir() and (d / MODEL_FILENAME).exists()
        ]

    def get_total_size_bytes(self) -> int:
        """Get total cache size in bytes."""
        total = 0
        for hotkey in self.get_all_hotkeys():
            model_path = self._cache_dir / hotkey / MODEL_FILENAME
            if model_path.exists():
                total += model_path.stat().st_size
        return total

    def get_free_disk_space(self) -> int:
        """Get available disk space in bytes."""
        return shutil.disk_usage(self._cache_dir).free

    def cleanup_corrupted(self) -> list[str]:
        """
        Scan cache and remove corrupted entries.

        Called on validator startup.
        An entry is corrupted if:
        - metadata.json is missing or invalid
        - model.onnx is missing

        Returns:
            List of removed hotkeys
        """
        removed = []
        for hotkey_dir in self._cache_dir.iterdir():
            if not hotkey_dir.is_dir():
                continue

            hotkey = hotkey_dir.name
            model_path = hotkey_dir / MODEL_FILENAME
            metadata_path = hotkey_dir / METADATA_FILENAME

            is_corrupted = False

            if not model_path.exists():
                logger.warning(f"Missing model.onnx for {hotkey}")
                is_corrupted = True
            elif not metadata_path.exists():
                logger.warning(f"Missing metadata.json for {hotkey}")
                is_corrupted = True
            else:
                # Try to parse metadata
                try:
                    with open(metadata_path) as f:
                        CachedModelMetadata.from_dict(json.load(f))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid metadata for {hotkey}: {e}")
                    is_corrupted = True

            if is_corrupted:
                shutil.rmtree(hotkey_dir)
                removed.append(hotkey)
                logger.info(f"Removed corrupted cache entry for {hotkey}")

        if removed:
            logger.info(f"Cleanup removed {len(removed)} corrupted entries")

        return removed

    def cleanup_stale(self, active_hotkeys: set[str]) -> list[str]:
        """
        Remove cache entries for hotkeys no longer on chain.

        Called periodically to prevent unbounded cache growth when
        miners deregister or commitments are removed.

        Args:
            active_hotkeys: Set of hotkeys with current commitments

        Returns:
            List of removed hotkeys
        """
        removed = []
        for hotkey in self.get_all_hotkeys():
            if hotkey not in active_hotkeys:
                self.remove(hotkey)
                removed.append(hotkey)
                logger.info(f"Removed stale cache entry for {hotkey}")

        if removed:
            logger.info(f"Cleanup removed {len(removed)} stale entries")

        return removed
