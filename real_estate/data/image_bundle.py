"""Image bundle decoder — unzip, decode JPEGs, build padded tensors.

Memory strategy:
  - The zip stays on disk; we never load all images into RAM at once.
  - `verify_bundle()` streams through images for sha256 without keeping pixels.
  - `decode_for_model()` builds one model's tensor just before Docker inference,
    then the caller writes it to the workspace and discards it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageBundleManifest:
    """Parsed manifest.json from an image bundle zip."""

    encoder_version: str
    resolution: tuple[int, int]
    format: str
    decoded_sha256: str
    properties: dict[str, list[str]]


@dataclass(frozen=True)
class DecodedImageBundle:
    """Decoded image tensors ready for model inference.

    Attributes:
        images: uint8 array of shape (N_properties, max_images, C, H, W).
        image_counts: int32 array of shape (N_properties,).
            Number of real images per property (slots beyond this are zero-padded).
        property_ids: ordered list of property IDs matching dim 0.
    """

    images: np.ndarray
    image_counts: np.ndarray
    property_ids: list[str]


def _validate_filename(filename: str) -> None:
    """Reject filenames with path traversal or absolute paths."""
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise ValueError(f"Unsafe filename in manifest: {filename!r}")


def parse_manifest(zip_path: Path) -> ImageBundleManifest:
    """Parse manifest.json from an image bundle zip.

    Raises:
        ValueError: If manifest is missing or malformed.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError("Image bundle zip missing manifest.json")

        with zf.open("manifest.json") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("manifest.json must be a JSON object")

    required = {"encoder_version", "resolution", "format", "decoded_sha256", "properties"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"manifest.json missing fields: {sorted(missing)}")

    resolution = data["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("manifest.resolution must be [H, W]")

    properties = data["properties"]
    if not isinstance(properties, dict):
        raise ValueError("manifest.properties must be an object")

    # Validate all filenames against path traversal
    for prop_id, filenames in properties.items():
        if not isinstance(filenames, list):
            raise ValueError(f"manifest.properties[{prop_id!r}] must be a list")
        for fn in filenames:
            _validate_filename(fn)

    return ImageBundleManifest(
        encoder_version=str(data["encoder_version"]),
        resolution=(int(resolution[0]), int(resolution[1])),
        format=data["format"],
        decoded_sha256=data["decoded_sha256"],
        properties=properties,
    )


def verify_bundle(zip_path: Path) -> ImageBundleManifest:
    """Parse manifest and verify decoded_sha256. Streams images — no big allocation.

    Call once after download, before any model evaluation.

    Returns:
        Verified manifest.

    Raises:
        ValueError: If manifest is invalid or sha256 mismatch.
    """
    manifest = parse_manifest(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        _verify_decoded_sha256(zf, manifest)

    return manifest


def decode_for_model(
    zip_path: Path,
    manifest: ImageBundleManifest,
    property_ids: list[str],
    max_images_per_property: int,
) -> DecodedImageBundle:
    """Decode images for a single model's inference run.

    Called per-model, right before writing to the Docker workspace.
    Only properties listed in property_ids are decoded; the rest are skipped.

    Args:
        zip_path: Path to the image bundle zip file.
        manifest: Pre-parsed and verified manifest (from verify_bundle).
        property_ids: Ordered property IDs from the validation set.
        max_images_per_property: From the model's ImageBlockConfig.

    Returns:
        DecodedImageBundle with padded image tensors and per-property counts.
    """
    from PIL import Image

    h, w = manifest.resolution
    c = 3  # Always RGB

    n_props = len(property_ids)
    images = np.zeros((n_props, max_images_per_property, c, h, w), dtype=np.uint8)
    image_counts = np.zeros(n_props, dtype=np.int32)

    prop_to_idx = {pid: i for i, pid in enumerate(property_ids)}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for prop_id, filenames in manifest.properties.items():
            if prop_id not in prop_to_idx:
                continue

            idx = prop_to_idx[prop_id]
            count = 0

            for img_idx, filename in enumerate(filenames[:max_images_per_property]):
                try:
                    with zf.open(filename) as f:
                        img = Image.open(f).convert("RGB")
                        if img.size != (w, h):
                            img = img.resize((w, h), Image.LANCZOS)
                        # HWC -> CHW
                        pixels = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
                        images[idx, img_idx] = pixels
                        count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to decode image {filename} for property {prop_id}: {e}"
                    )

            image_counts[idx] = count

    total = int(image_counts.sum())
    logger.info(
        f"Decoded image bundle for model: {total} images across {n_props} properties "
        f"(max {max_images_per_property}/prop, "
        f"tensor {images.nbytes / 1024 / 1024:.0f} MB)"
    )

    return DecodedImageBundle(images=images, image_counts=image_counts, property_ids=property_ids)


def _verify_decoded_sha256(zf: zipfile.ZipFile, manifest: ImageBundleManifest) -> None:
    """Verify decoded pixels match the manifest's decoded_sha256.

    Streams one image at a time through the hasher — no big allocation.

    Raises:
        ValueError: On mismatch.
    """
    from PIL import Image

    h, w = manifest.resolution
    hasher = hashlib.sha256()

    for prop_id in sorted(manifest.properties.keys()):
        for filename in manifest.properties[prop_id]:
            with zf.open(filename) as f:
                img = Image.open(f).convert("RGB")
                if img.size != (w, h):
                    raise ValueError(
                        f"Image {filename} has wrong size {img.size}, expected ({w}, {h})"
                    )
                hasher.update(np.asarray(img, dtype=np.uint8).tobytes())

    actual = hasher.hexdigest()
    if actual != manifest.decoded_sha256:
        raise ValueError(
            f"decoded_sha256 mismatch: manifest={manifest.decoded_sha256}, "
            f"actual={actual}"
        )

    logger.debug("Image bundle decoded_sha256 verified OK")
