"""Tests for randomness data models."""

from real_estate.randomness import RandomnessConfig, SeedResult


class TestRandomnessConfig:
    def test_defaults(self) -> None:
        config = RandomnessConfig()
        assert config.commit_hours_before_eval == 4.0
        assert config.blocks_until_reveal == 360
        assert config.seed_modulus == 2**32

    def test_frozen(self) -> None:
        config = RandomnessConfig()
        try:
            config.blocks_until_reveal = 100  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass


class TestSeedResult:
    def test_creation(self) -> None:
        result = SeedResult(
            seed=42,
            num_reveals=3,
            validator_hotkeys=frozenset({"a", "b", "c"}),
        )
        assert result.seed == 42
        assert result.num_reveals == 3
        assert len(result.validator_hotkeys) == 3

    def test_frozen(self) -> None:
        result = SeedResult(seed=1, num_reveals=1, validator_hotkeys=frozenset())
        try:
            result.seed = 99  # type: ignore[misc]
            assert False, "Should raise FrozenInstanceError"
        except AttributeError:
            pass
