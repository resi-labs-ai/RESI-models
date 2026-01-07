"""Pioneer detector for finding earliest committers in duplicate groups."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import DuplicateGroup

if TYPE_CHECKING:
    from ..chain.models import ChainModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class PioneerDetectionResult:
    """Result of pioneer detection with tracking of skipped hotkeys."""

    pioneer_hotkeys: frozenset[str]
    """Hotkeys that were first to submit within their duplicate group."""

    copier_hotkeys: frozenset[str]
    """Hotkeys that copied another model (submitted after pioneer)."""

    skipped_hotkeys: list[str]
    """Hotkeys that were skipped due to missing chain metadata."""


class PioneerDetector:
    """
    Determines the pioneer (earliest committer) in each duplicate group.

    Pioneer = model with lowest block_number in the group.
    Non-pioneers are considered "copiers" who submitted identical predictions
    after the pioneer.

    Hotkeys without chain metadata are skipped (logged, not failed).

    Usage:
        detector = PioneerDetector()
        result = detector.detect_pioneers(groups, metadata)
        # result.pioneer_hotkeys = frozenset of first submitters
        # result.copier_hotkeys = frozenset of copiers (to be zero-scored)
        # result.skipped_hotkeys = hotkeys without metadata
    """

    def detect_pioneers(
        self,
        groups: list[DuplicateGroup],
        metadata: dict[str, ChainModelMetadata],
    ) -> PioneerDetectionResult:
        """
        Identify pioneers and copiers in each duplicate group.

        Hotkeys without metadata are skipped and logged, not failed.
        Groups that end up with < 2 hotkeys after filtering are skipped.

        Args:
            groups: List of duplicate groups from PredictionGrouper
            metadata: Chain metadata for models, keyed by hotkey.
                     Hotkeys missing from this dict are skipped.

        Returns:
            PioneerDetectionResult with:
            - pioneer_hotkeys: hotkeys that submitted first
            - copier_hotkeys: hotkeys that copied (submitted after pioneer)
            - skipped_hotkeys: hotkeys without metadata

        Example:
            groups = [DuplicateGroup(hotkeys=("A", "B", "C"))]
            metadata = {
                "A": ChainModelMetadata(hotkey="A", block_number=1000, ...),
                "B": ChainModelMetadata(hotkey="B", block_number=900, ...),
                # C is missing - will be skipped
            }
            result = detector.detect_pioneers(groups, metadata)
            # result.pioneer_hotkeys = frozenset({"B"})
            # result.copier_hotkeys = frozenset({"A"})
            # result.skipped_hotkeys = ["C"]
        """
        # Find hotkeys without metadata
        skipped_hotkeys = self._find_hotkeys_with_missing_metadata(groups, metadata)
        if skipped_hotkeys:
            logger.warning(
                f"Skipping {len(skipped_hotkeys)} hotkeys without chain metadata: "
                f"{', '.join(skipped_hotkeys[:5])}{'...' if len(skipped_hotkeys) > 5 else ''}"
            )

        pioneers: set[str] = set()
        copiers: set[str] = set()

        for group in groups:
            # Filter to hotkeys with metadata
            hotkeys_with_metadata = [hk for hk in group.hotkeys if hk in metadata]

            # Skip if < 2 hotkeys remain (can't determine pioneer/copier)
            if len(hotkeys_with_metadata) < 2:
                continue

            # Find pioneer (lowest block number)
            pioneer_hotkey = self._find_pioneer_in_group(
                hotkeys_with_metadata, metadata
            )

            # Categorize hotkeys
            pioneers.add(pioneer_hotkey)
            for hotkey in hotkeys_with_metadata:
                if hotkey != pioneer_hotkey:
                    copiers.add(hotkey)

        return PioneerDetectionResult(
            pioneer_hotkeys=frozenset(pioneers),
            copier_hotkeys=frozenset(copiers),
            skipped_hotkeys=skipped_hotkeys,
        )

    def _find_hotkeys_with_missing_metadata(
        self,
        groups: list[DuplicateGroup],
        metadata: dict[str, ChainModelMetadata],
    ) -> list[str]:
        """Find hotkeys in groups that don't have metadata."""
        all_hotkeys = {hk for group in groups for hk in group.hotkeys}
        missing = [hk for hk in all_hotkeys if hk not in metadata]
        return sorted(missing)

    def _find_pioneer_in_group(
        self,
        hotkeys: list[str],
        metadata: dict[str, ChainModelMetadata],
    ) -> str:
        """
        Find the pioneer (earliest committer) in a group.

        Returns hotkey with lowest block_number.
        """
        return min(hotkeys, key=lambda hk: metadata[hk].block_number)
