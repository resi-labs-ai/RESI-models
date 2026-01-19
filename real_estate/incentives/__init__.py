"""
Incentives module for calculating scores and distributing weights.

This module handles:
- Winner selection using threshold + commit time mechanism
- Incentive distribution (99% winner, 1% non-cheaters, 0% cheaters)

Main components:
- WinnerSelector: Selects winner among models with similar scores
- IncentiveDistributor: Calculates weight distribution for chain

Usage:
    from real_estate.incentives import WinnerSelector, IncentiveDistributor

    # Select winner
    selector = WinnerSelector(score_threshold=0.005)
    winner_result = selector.select_winner(evaluation_results, chain_metadata)

    # Calculate weights
    distributor = IncentiveDistributor()
    weights = distributor.calculate_weights(
        results=evaluation_results,
        winner_hotkey=winner_result.winner_hotkey,
        winner_score=winner_result.winner_score,
        cheater_hotkeys=copier_hotkeys,
    )

    # Set weights on chain
    await chain_client.set_weights(weights.weights)
"""

from .distributor import IncentiveDistributor
from .errors import IncentiveError, NoValidModelsError
from .models import (
    DistributorConfig,
    IncentiveWeights,
    WinnerCandidate,
    WinnerSelectionResult,
)
from .scorer import WinnerSelector

__all__ = [
    # Main components
    "WinnerSelector",
    "IncentiveDistributor",
    # Configuration
    "DistributorConfig",
    # Result models
    "WinnerSelectionResult",
    "WinnerCandidate",
    "IncentiveWeights",
    # Errors
    "IncentiveError",
    "NoValidModelsError",
]
