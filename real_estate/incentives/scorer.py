"""Winner selection with threshold mechanism."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .errors import NoValidModelsError
from .models import WinnerCandidate, WinnerSelectionConfig, WinnerSelectionResult

if TYPE_CHECKING:
    from ..chain.models import ChainModelMetadata
    from ..evaluation.models import EvaluationResult

logger = logging.getLogger(__name__)


class WinnerSelector:
    """
    Selects winner using threshold + commit time mechanism.

    Instead of pure "highest score wins", this uses a threshold approach:
    1. Define winner set: All models within threshold of the best score
    2. Elect winner by commit time: Within the winner set, earliest commit wins
    3. Reward innovation: If you improve by more than threshold, you win

    Usage:
        selector = WinnerSelector()
        result = selector.select_winner(evaluation_results, chain_metadata)
        print(f"Winner: {result.winner_hotkey}")
    """

    def __init__(self, config: WinnerSelectionConfig | None = None):
        """
        Initialize winner selector.

        Args:
            config: Configuration for winner selection. Uses defaults if None.
        """
        self._config = config or WinnerSelectionConfig()

    @property
    def threshold(self) -> float:
        """Score threshold for winner set."""
        return self._config.score_threshold

    def select_winner(
        self,
        results: list[EvaluationResult],
        metadata: dict[str, ChainModelMetadata],
    ) -> WinnerSelectionResult:
        """
        Select winner using threshold + commit time mechanism.

        Args:
            results: Evaluation results with scores
            metadata: Chain metadata with commit blocks (keyed by hotkey)

        Returns:
            WinnerSelectionResult with winner and candidate information

        Raises:
            NoValidModelsError: If no valid models to evaluate
        """
        # Filter to successful results only
        valid_results = [r for r in results if r.success]

        if not valid_results:
            raise NoValidModelsError("No successful evaluation results")

        # Sort by score descending
        sorted_results = sorted(valid_results, key=lambda r: r.score, reverse=True)
        best_score = sorted_results[0].score

        # Define winner set (within threshold of best)
        threshold_cutoff = best_score - self._config.score_threshold
        winner_set_results = [r for r in sorted_results if r.score >= threshold_cutoff]

        # Build candidates with metadata
        # All evaluated models must have metadata (they came from chain commitments)
        candidates: list[WinnerCandidate] = []
        for result in winner_set_results:
            if result.hotkey not in metadata:
                raise ValueError(
                    f"Missing chain metadata for hotkey {result.hotkey}. "
                    f"This indicates a bug - evaluated models must have metadata."
                )

            meta = metadata[result.hotkey]
            candidates.append(
                WinnerCandidate(
                    hotkey=result.hotkey,
                    score=result.score,
                    block_number=meta.block_number,
                )
            )

        # Within winner set, pick earliest commit
        winner = min(candidates, key=lambda c: (c.block_number, c.hotkey))

        logger.info(
            f"Selected winner {winner.hotkey} "
            f"(score={winner.score:.4f}, block={winner.block_number}) "
            f"from {len(candidates)} candidates"
        )

        return WinnerSelectionResult(
            winner_hotkey=winner.hotkey,
            winner_score=winner.score,
            winner_block=winner.block_number,
            candidates=tuple(candidates),
            best_score=best_score,
            threshold=self._config.score_threshold,
        )
