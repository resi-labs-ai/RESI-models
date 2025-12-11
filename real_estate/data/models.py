"""Data models for validation datasets."""

from dataclasses import dataclass
from typing import Any

# Fields are validated by FeatureEncoder against feature_config.yaml
PropertyData = dict[str, Any]


@dataclass
class ValidationDataset:
    """
    Validation dataset with ground truth sale prices.

    Ground truth is extracted from the 'price' field of each property.
    Field validation happens in FeatureEncoder.
    """

    properties: list[PropertyData]

    def __len__(self) -> int:
        return len(self.properties)

    def __repr__(self) -> str:
        return f"ValidationDataset({len(self.properties)} properties)"

    @property
    def ground_truth(self) -> list[float]:
        """Extract ground truth (price) from properties."""
        return [float(p["price"]) for p in self.properties]
