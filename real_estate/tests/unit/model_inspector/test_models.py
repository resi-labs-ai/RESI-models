"""Tests for model_inspector models."""

from real_estate.model_inspector.models import (
    InspectionBatchResult,
    ModelInspectionResult,
    RejectionReason,
)


class TestInspectionBatchResult:
    def test_rejected_hotkeys_computed(self) -> None:
        """rejected_hotkeys aggregates across results correctly."""
        results = (
            ModelInspectionResult(
                hotkey="bad1",
                has_lookup_pattern=True,
                has_unused_initializers=False,
                price_like_values=0,
                total_params=0,
                rejection_reason=RejectionReason.LOOKUP_PATTERN,
            ),
            ModelInspectionResult(
                hotkey="good",
                has_lookup_pattern=False,
                has_unused_initializers=False,
                price_like_values=0,
                total_params=0,
            ),
        )
        result = InspectionBatchResult(results=results)
        assert result.is_rejected("bad1")
        assert not result.is_rejected("good")

