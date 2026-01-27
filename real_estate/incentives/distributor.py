"""Incentive distribution with winner-takes-most mechanism."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import DistributorConfig, IncentiveWeights

if TYPE_CHECKING:
    from ..evaluation.models import EvaluationResult

logger = logging.getLogger(__name__)


class IncentiveDistributor:
    """
    Distribute incentive weights: 99% to winner, 1% to non-cheaters.

    Three-tier system:
    1. Winner: Gets winner_share (default 99%) of emissions
    2. Non-cheaters (non-winners): Share non_winner_share (default 1%) proportionally
    3. Cheaters: Get 0%

    Usage:
        distributor = IncentiveDistributor()
        weights = distributor.calculate_weights(
            results=evaluation_results,
            winner_hotkey="5Faa...",
            cheater_hotkeys={"5Bbb...", "5Ccc..."}
        )
    """

    def __init__(self, config: DistributorConfig | None = None):
        """
        Initialize distributor.

        Args:
            config: Configuration for weight distribution. Uses defaults if None.
        """
        self._config = config or DistributorConfig()

    @property
    def winner_share(self) -> float:
        """Share allocated to winner."""
        return self._config.winner_share

    @property
    def non_winner_share(self) -> float:
        """Share distributed among non-winners."""
        return self._config.non_winner_share

    def calculate_weights(
        self,
        results: list[EvaluationResult],
        winner_hotkey: str,
        winner_score: float,
        cheater_hotkeys: frozenset[str] | set[str] | None = None,
    ) -> IncentiveWeights:
        """
        Calculate weights for all miners.

        Args:
            results: All evaluation results
            winner_hotkey: Hotkey of the selected winner
            winner_score: Score of the winning model
            cheater_hotkeys: Set of hotkeys identified as cheaters (copiers)

        Returns:
            IncentiveWeights with weight distribution
        """
        cheaters = cheater_hotkeys or frozenset()

        # Filter to successful, non-cheater results (excluding winner)
        non_winner_results = [
            r
            for r in results
            if r.success and r.hotkey != winner_hotkey and r.hotkey not in cheaters
        ]

        # Build weight distribution
        weights: dict[str, float] = {}

        # Winner gets majority share
        weights[winner_hotkey] = self._config.winner_share

        # Non-winners share remaining proportionally by score
        if non_winner_results:
            total_score = sum(r.score for r in non_winner_results)
            for result in non_winner_results:
                share = (
                    (result.score / total_score) * self._config.non_winner_share
                    if total_score > 0
                    else 0.0
                )
                weights[result.hotkey] = share

        # Cheaters explicitly get 0
        for cheater in cheaters:
            weights[cheater] = 0.0

        logger.info(
            f"Distributed weights: winner={winner_hotkey} ({self._config.winner_share:.1%}), "
            f"non_winners={len(non_winner_results)} ({self._config.non_winner_share:.1%}), "
            f"cheaters={len(cheaters)} (0%)"
        )

        # Log detailed weight distribution
        logger.debug("Weight distribution (all miners):")
        for hotkey, weight in sorted(weights.items(), key=lambda x: -x[1]):
            logger.debug(f"  {hotkey}: {weight:.6f}")

        return IncentiveWeights(
            weights=weights,
            winner_hotkey=winner_hotkey,
            winner_score=winner_score,
        )
