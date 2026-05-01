"""Tests for image bundle feature support."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from real_estate.data import (
    IMAGES_FEATURE_NAME,
    TabularEncoder,
    FeatureConfig,
    FeatureConfigError,
    ImageBlockConfig,
    parse_feature_config,
)
from real_estate.data.config_encoder import IMAGE_BLOCK_V1_DIM, MAX_IMAGES_PER_PROPERTY_V1
from real_estate.data.image_bundle import (
    DecodedImageBundle,
    ImageBundleManifest,
    decode_for_model,
    parse_manifest,
    verify_bundle,
)

REQUIRED = ["living_area_sqft", "latitude", "longitude", "bedrooms", "bathrooms"]
MINIMAL_FEATURES = REQUIRED + [
    "lot_size_sqft",
    "year_built",
    "has_pool",
    "has_garage",
    "stories",
]


def _make_raw(features=None, version="1.0"):
    return {"version": version, "features": features or MINIMAL_FEATURES}


# =====================================================================
# FeatureConfig + property_images feature
# =====================================================================


class TestPropertyImagesFeature:
    def test_no_images(self) -> None:
        fc = parse_feature_config(_make_raw())
        assert fc.image_block is None

    def test_with_property_images(self) -> None:
        fc = parse_feature_config(_make_raw(features=MINIMAL_FEATURES + [IMAGES_FEATURE_NAME]))
        assert fc.image_block is not None
        assert fc.image_block.dim == IMAGE_BLOCK_V1_DIM
        assert fc.image_block.max_images_per_property == MAX_IMAGES_PER_PROPERTY_V1

    def test_property_images_not_counted_in_feature_bounds(self) -> None:
        """property_images shouldn't count toward the 10-79 numeric/boolean limit."""
        fc = parse_feature_config(_make_raw(features=MINIMAL_FEATURES + [IMAGES_FEATURE_NAME]))
        assert len(fc.features) == 11  # 10 numeric + property_images
        assert fc.image_block is not None

    def test_encoder_excludes_property_images_from_array(self) -> None:
        """TabularEncoder should only encode numeric/boolean features."""
        fc = parse_feature_config(_make_raw(features=MINIMAL_FEATURES + [IMAGES_FEATURE_NAME]))
        encoder = TabularEncoder(fc)
        props = [{"living_area_sqft": 2000, "latitude": 40.0, "longitude": -74.0,
                  "bedrooms": 3, "bathrooms": 2}]
        arr = encoder.encode(props)
        # Should have 10 columns (not 11) — property_images excluded
        assert arr.shape == (1, 10)

    def test_property_images_in_features_list(self) -> None:
        """property_images should appear in the features tuple."""
        fc = parse_feature_config(_make_raw(features=MINIMAL_FEATURES + [IMAGES_FEATURE_NAME]))
        assert IMAGES_FEATURE_NAME in fc.features


# =====================================================================
# Image bundle zip fixtures
# =====================================================================


def _create_test_image(width=224, height=224, seed=42) -> bytes:
    """Create a minimal JPEG image of the given size."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _compute_decoded_sha256(images: dict[str, list[bytes]], resolution=(224, 224)):
    """Compute decoded_sha256 matching the bundle spec."""
    from PIL import Image

    h, w = resolution
    hasher = hashlib.sha256()
    for prop_id in sorted(images.keys()):
        for img_bytes in images[prop_id]:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            assert img.size == (w, h)
            hasher.update(np.asarray(img, dtype=np.uint8).tobytes())
    return hasher.hexdigest()


def _create_test_bundle(tmp_path: Path, properties: dict[str, int] | None = None) -> Path:
    """Create a test image bundle zip.

    Args:
        tmp_path: pytest tmp_path
        properties: mapping of property_id -> number of images
    """
    if properties is None:
        properties = {"prop_a": 2, "prop_b": 1, "prop_c": 3}

    # Generate images
    images: dict[str, list[bytes]] = {}
    filenames_map: dict[str, list[str]] = {}
    seed = 0
    for prop_id, n_imgs in properties.items():
        images[prop_id] = []
        filenames_map[prop_id] = []
        for i in range(n_imgs):
            img_bytes = _create_test_image(seed=seed)
            filename = f"{prop_id}_{i}.jpg"
            images[prop_id].append(img_bytes)
            filenames_map[prop_id].append(filename)
            seed += 1

    decoded_sha256 = _compute_decoded_sha256(images)

    manifest = {
        "encoder_version": "1",
        "resolution": [224, 224],
        "format": "jpeg_q85_rgb",
        "decoded_sha256": decoded_sha256,
        "properties": filenames_map,
    }

    zip_path = tmp_path / "test_images.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for prop_id, img_list in images.items():
            for img_bytes, filename in zip(img_list, filenames_map[prop_id]):
                zf.writestr(filename, img_bytes)

    return zip_path


# =====================================================================
# parse_manifest tests
# =====================================================================


class TestParseManifest:
    def test_valid_bundle(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)
        manifest = parse_manifest(zip_path)
        assert manifest.encoder_version == "1"
        assert manifest.resolution == (224, 224)
        assert manifest.format == "jpeg_q85_rgb"
        assert "prop_a" in manifest.properties
        assert len(manifest.properties["prop_a"]) == 2

    def test_missing_manifest(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("dummy.txt", "hello")

        with pytest.raises(ValueError, match="missing manifest.json"):
            parse_manifest(zip_path)

    def test_malformed_manifest(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"only_one_field": True}))

        with pytest.raises(ValueError, match="missing fields"):
            parse_manifest(zip_path)

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        manifest = {
            "encoder_version": "1",
            "resolution": [224, 224],
            "format": "jpeg_q85_rgb",
            "decoded_sha256": "abc123",
            "properties": {
                "evil_prop": ["../../etc/passwd"],
            },
        }
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        with pytest.raises(ValueError, match="Unsafe filename"):
            parse_manifest(zip_path)

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        manifest = {
            "encoder_version": "1",
            "resolution": [224, 224],
            "format": "jpeg_q85_rgb",
            "decoded_sha256": "abc123",
            "properties": {
                "evil_prop": ["/etc/shadow"],
            },
        }
        zip_path = tmp_path / "evil2.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        with pytest.raises(ValueError, match="Unsafe filename"):
            parse_manifest(zip_path)


# =====================================================================
# verify_bundle tests
# =====================================================================


class TestVerifyBundle:
    def test_valid_bundle_passes(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)
        manifest = verify_bundle(zip_path)
        assert isinstance(manifest, ImageBundleManifest)

    def test_corrupted_sha256_fails(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)

        # Tamper with the sha256 in manifest
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest_data = json.loads(zf.read("manifest.json"))
            all_names = zf.namelist()

        manifest_data["decoded_sha256"] = "deadbeef" * 8

        tampered_path = tmp_path / "tampered.zip"
        with zipfile.ZipFile(zip_path, "r") as zf_in:
            with zipfile.ZipFile(tampered_path, "w") as zf_out:
                for name in all_names:
                    if name == "manifest.json":
                        zf_out.writestr(name, json.dumps(manifest_data))
                    else:
                        zf_out.writestr(name, zf_in.read(name))

        with pytest.raises(ValueError, match="decoded_sha256 mismatch"):
            verify_bundle(tampered_path)


# =====================================================================
# decode_for_model tests
# =====================================================================


class TestDecodeForModel:
    def test_decode_matching_properties(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)
        manifest = parse_manifest(zip_path)

        result = decode_for_model(
            zip_path=zip_path,
            manifest=manifest,
            property_ids=["prop_a", "prop_b", "prop_c"],
            max_images_per_property=3,
        )

        assert isinstance(result, DecodedImageBundle)
        assert result.images.shape == (3, 3, 3, 224, 224)
        assert result.image_counts.shape == (3,)
        assert result.images.dtype == np.uint8

        # prop_a has 2 images, prop_b has 1, prop_c has 3
        assert result.image_counts.tolist() == [2, 1, 3]

    def test_decode_missing_property_gets_zeros(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)
        manifest = parse_manifest(zip_path)

        result = decode_for_model(
            zip_path=zip_path,
            manifest=manifest,
            property_ids=["prop_a", "unknown_prop"],
            max_images_per_property=2,
        )

        assert result.images.shape == (2, 2, 3, 224, 224)
        # unknown_prop gets zero images
        assert result.image_counts[1] == 0
        assert result.images[1].sum() == 0

    def test_decode_truncates_to_max_images(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)
        manifest = parse_manifest(zip_path)

        # prop_c has 3 images but we only want 1
        result = decode_for_model(
            zip_path=zip_path,
            manifest=manifest,
            property_ids=["prop_c"],
            max_images_per_property=1,
        )

        assert result.images.shape == (1, 1, 3, 224, 224)
        assert result.image_counts[0] == 1

    def test_decode_nonzero_pixels(self, tmp_path: Path) -> None:
        zip_path = _create_test_bundle(tmp_path)
        manifest = parse_manifest(zip_path)

        result = decode_for_model(
            zip_path=zip_path,
            manifest=manifest,
            property_ids=["prop_a"],
            max_images_per_property=2,
        )

        # Decoded images should have actual pixel data (not all zeros)
        assert result.images[0, 0].sum() > 0
