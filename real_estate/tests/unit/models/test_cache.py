"""Unit tests for ModelCache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from real_estate.models import ModelCache


class TestGet:
    """Tests for ModelCache.get method."""

    def test_returns_none_when_not_cached(self, cache: ModelCache) -> None:
        """Returns None when hotkey is not in cache."""
        result = cache.get("nonexistent_hotkey")
        assert result is None

    def test_returns_cached_model_when_exists(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns CachedModel when model and metadata exist."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Create model file
        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        # Create metadata
        metadata_path = hotkey_dir / "metadata.json"
        metadata = {"hash": "abc12345", "size_bytes": 13}
        metadata_path.write_text(json.dumps(metadata))

        result = cache.get(hotkey)

        assert result is not None
        assert result.path == model_path
        assert result.metadata.hash == "abc12345"
        assert result.metadata.size_bytes == 13

    def test_returns_none_when_metadata_corrupted(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns None when metadata JSON is corrupted."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Create model file
        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        # Create corrupted metadata
        metadata_path = hotkey_dir / "metadata.json"
        metadata_path.write_text("not valid json {")

        result = cache.get(hotkey)
        assert result is None

    def test_returns_none_when_model_file_missing(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns None when model.onnx is missing."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Only create metadata, no model file
        metadata_path = hotkey_dir / "metadata.json"
        metadata = {"hash": "abc12345", "size_bytes": 13}
        metadata_path.write_text(json.dumps(metadata))

        result = cache.get(hotkey)
        assert result is None

    def test_returns_none_when_metadata_missing_required_fields(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns None when metadata is missing required fields."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Create model file
        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        # Create metadata missing 'hash' field
        metadata_path = hotkey_dir / "metadata.json"
        metadata = {"size_bytes": 13}  # missing 'hash'
        metadata_path.write_text(json.dumps(metadata))

        result = cache.get(hotkey)
        assert result is None


class TestIsValid:
    """Tests for ModelCache.is_valid method."""

    def test_returns_false_when_not_cached(self, cache: ModelCache) -> None:
        """Returns False when hotkey is not in cache."""
        result = cache.is_valid("nonexistent_hotkey", "any_hash")
        assert result is False

    def test_returns_true_when_hash_matches(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns True when cached hash matches expected hash."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        metadata_path = hotkey_dir / "metadata.json"
        metadata = {"hash": "abc12345", "size_bytes": 13}
        metadata_path.write_text(json.dumps(metadata))

        result = cache.is_valid(hotkey, "abc12345")
        assert result is True

    def test_returns_false_when_hash_differs(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns False when cached hash differs from expected."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        metadata_path = hotkey_dir / "metadata.json"
        metadata = {"hash": "abc12345", "size_bytes": 13}
        metadata_path.write_text(json.dumps(metadata))

        result = cache.is_valid(hotkey, "different_hash")
        assert result is False


class TestPut:
    """Tests for ModelCache.put method."""

    def test_stores_model_and_metadata(
        self, cache: ModelCache, temp_cache_dir: Path, tmp_path: Path
    ) -> None:
        """Stores model file and creates metadata."""
        hotkey = "test_hotkey"

        # Create temp model file
        temp_model = tmp_path / "temp_model.onnx"
        temp_model.write_bytes(b"model content here")

        cached_path = cache.put(
            hotkey=hotkey,
            temp_model_path=temp_model,
            model_hash="abc12345",
            size_bytes=18,
        )

        # Verify model was stored
        assert cached_path.exists()
        assert cached_path == temp_cache_dir / hotkey / "model.onnx"
        assert cached_path.read_bytes() == b"model content here"

        # Verify metadata was created
        metadata_path = temp_cache_dir / hotkey / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["hash"] == "abc12345"
        assert metadata["size_bytes"] == 18

    def test_atomic_move_from_temp(
        self, cache: ModelCache, tmp_path: Path
    ) -> None:
        """Verifies temp file is moved (not copied)."""
        hotkey = "test_hotkey"

        temp_model = tmp_path / "temp_model.onnx"
        temp_model.write_bytes(b"model content")

        cache.put(
            hotkey=hotkey,
            temp_model_path=temp_model,
            model_hash="abc12345",
            size_bytes=13,
        )

        # Temp file should no longer exist
        assert not temp_model.exists()

    def test_overwrites_existing_cache(
        self, cache: ModelCache, temp_cache_dir: Path, tmp_path: Path
    ) -> None:
        """Overwrites existing cached model."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Create existing cached model
        existing_model = hotkey_dir / "model.onnx"
        existing_model.write_bytes(b"old content")

        # Put new model
        temp_model = tmp_path / "temp_model.onnx"
        temp_model.write_bytes(b"new content")

        cache.put(
            hotkey=hotkey,
            temp_model_path=temp_model,
            model_hash="new_hash",
            size_bytes=11,
        )

        # Verify new content
        cached_model = hotkey_dir / "model.onnx"
        assert cached_model.read_bytes() == b"new content"

        metadata_path = hotkey_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        assert metadata["hash"] == "new_hash"


class TestRemove:
    """Tests for ModelCache.remove method."""

    def test_removes_cached_model(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Removes cached model directory."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        metadata_path = hotkey_dir / "metadata.json"
        metadata_path.write_text('{"hash": "abc", "size_bytes": 13}')

        result = cache.remove(hotkey)

        assert result is True
        assert not hotkey_dir.exists()

    def test_returns_false_when_not_found(self, cache: ModelCache) -> None:
        """Returns False when hotkey not in cache."""
        result = cache.remove("nonexistent_hotkey")
        assert result is False


class TestCleanupCorrupted:
    """Tests for ModelCache.cleanup_corrupted method."""

    def test_removes_entries_with_missing_model(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Removes entries where model.onnx is missing."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Only create metadata, no model
        metadata_path = hotkey_dir / "metadata.json"
        metadata_path.write_text('{"hash": "abc", "size_bytes": 13}')

        removed = cache.cleanup_corrupted()

        assert hotkey in removed
        assert not hotkey_dir.exists()

    def test_removes_entries_with_missing_metadata(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Removes entries where metadata.json is missing."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        # Only create model, no metadata
        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        removed = cache.cleanup_corrupted()

        assert hotkey in removed
        assert not hotkey_dir.exists()

    def test_removes_entries_with_invalid_metadata_json(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Removes entries where metadata.json is invalid JSON."""
        hotkey = "test_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        metadata_path = hotkey_dir / "metadata.json"
        metadata_path.write_text("not valid json")

        removed = cache.cleanup_corrupted()

        assert hotkey in removed
        assert not hotkey_dir.exists()

    def test_keeps_valid_entries(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Keeps valid cache entries."""
        hotkey = "valid_hotkey"
        hotkey_dir = temp_cache_dir / hotkey
        hotkey_dir.mkdir()

        model_path = hotkey_dir / "model.onnx"
        model_path.write_bytes(b"model content")

        metadata_path = hotkey_dir / "metadata.json"
        metadata_path.write_text('{"hash": "abc12345", "size_bytes": 13}')

        removed = cache.cleanup_corrupted()

        assert hotkey not in removed
        assert hotkey_dir.exists()


class TestGetFreeDiskSpace:
    """Tests for ModelCache.get_free_disk_space method."""

    def test_returns_disk_usage(self, cache: ModelCache) -> None:
        """Returns free disk space from shutil.disk_usage."""
        mock_usage = type("DiskUsage", (), {"free": 500_000_000})()

        with patch("shutil.disk_usage", return_value=mock_usage):
            result = cache.get_free_disk_space()

        assert result == 500_000_000


class TestGetAllHotkeys:
    """Tests for ModelCache.get_all_hotkeys method."""

    def test_returns_empty_when_no_cached_models(self, cache: ModelCache) -> None:
        """Returns empty list when cache is empty."""
        result = cache.get_all_hotkeys()
        assert result == []

    def test_returns_all_cached_hotkeys(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns list of all cached hotkeys."""
        # Create two cached models
        for hotkey in ["hotkey1", "hotkey2"]:
            hotkey_dir = temp_cache_dir / hotkey
            hotkey_dir.mkdir()
            (hotkey_dir / "model.onnx").write_bytes(b"content")

        result = cache.get_all_hotkeys()

        assert set(result) == {"hotkey1", "hotkey2"}

    def test_excludes_incomplete_entries(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Excludes entries without model.onnx."""
        # Valid entry
        valid_dir = temp_cache_dir / "valid_hotkey"
        valid_dir.mkdir()
        (valid_dir / "model.onnx").write_bytes(b"content")

        # Invalid entry (missing model)
        invalid_dir = temp_cache_dir / "invalid_hotkey"
        invalid_dir.mkdir()
        (invalid_dir / "metadata.json").write_text("{}")

        result = cache.get_all_hotkeys()

        assert result == ["valid_hotkey"]


class TestGetTotalSizeBytes:
    """Tests for ModelCache.get_total_size_bytes method."""

    def test_returns_zero_when_empty(self, cache: ModelCache) -> None:
        """Returns 0 when cache is empty."""
        result = cache.get_total_size_bytes()
        assert result == 0

    def test_returns_sum_of_model_sizes(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Returns total size of all cached models."""
        # Create models with different sizes
        for hotkey, content in [("h1", b"small"), ("h2", b"medium content here")]:
            hotkey_dir = temp_cache_dir / hotkey
            hotkey_dir.mkdir()
            (hotkey_dir / "model.onnx").write_bytes(content)

        result = cache.get_total_size_bytes()

        # "small" = 5 bytes, "medium content here" = 19 bytes
        assert result == 24


class TestCleanupStale:
    """Tests for ModelCache.cleanup_stale method."""

    def test_removes_hotkeys_not_in_active_set(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Removes cached models for hotkeys no longer on chain."""
        # Create cached models for 3 hotkeys
        for hotkey in ["active1", "active2", "stale_hotkey"]:
            hotkey_dir = temp_cache_dir / hotkey
            hotkey_dir.mkdir()
            (hotkey_dir / "model.onnx").write_bytes(b"content")

        # Only 2 are still active on chain
        active_hotkeys = {"active1", "active2"}
        removed = cache.cleanup_stale(active_hotkeys)

        assert removed == ["stale_hotkey"]
        assert not (temp_cache_dir / "stale_hotkey").exists()
        assert (temp_cache_dir / "active1").exists()
        assert (temp_cache_dir / "active2").exists()

    def test_keeps_all_when_all_active(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Keeps all cached models when all hotkeys are active."""
        for hotkey in ["h1", "h2"]:
            hotkey_dir = temp_cache_dir / hotkey
            hotkey_dir.mkdir()
            (hotkey_dir / "model.onnx").write_bytes(b"content")

        active_hotkeys = {"h1", "h2"}
        removed = cache.cleanup_stale(active_hotkeys)

        assert removed == []
        assert (temp_cache_dir / "h1").exists()
        assert (temp_cache_dir / "h2").exists()

    def test_removes_all_when_none_active(
        self, cache: ModelCache, temp_cache_dir: Path
    ) -> None:
        """Removes all cached models when no hotkeys are active."""
        for hotkey in ["old1", "old2"]:
            hotkey_dir = temp_cache_dir / hotkey
            hotkey_dir.mkdir()
            (hotkey_dir / "model.onnx").write_bytes(b"content")

        active_hotkeys: set[str] = set()
        removed = cache.cleanup_stale(active_hotkeys)

        assert set(removed) == {"old1", "old2"}
        assert not (temp_cache_dir / "old1").exists()
        assert not (temp_cache_dir / "old2").exists()

    def test_returns_empty_when_cache_empty(self, cache: ModelCache) -> None:
        """Returns empty list when cache is empty."""
        active_hotkeys = {"h1", "h2"}
        removed = cache.cleanup_stale(active_hotkeys)

        assert removed == []
