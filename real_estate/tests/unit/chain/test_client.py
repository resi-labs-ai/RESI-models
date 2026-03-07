"""Tests for ChainClient — specifically the RevealedCommitments merge logic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from real_estate.chain.client import ChainClient, PylonConfig, _extract_netuid
from real_estate.chain.models import ChainModelMetadata


def _hex_encode(data: dict) -> str:
    """Encode dict as hex string (matching chain commitment format)."""
    return "0x" + json.dumps(data, separators=(",", ":")).encode().hex()


@pytest.fixture
def pylon_config():
    return PylonConfig(url="http://localhost:8000", token="test", identity="validator-46")


class TestExtractNetuid:
    def test_extracts_from_validator_identity(self):
        assert _extract_netuid("validator-46") == 46

    def test_extracts_from_identity_with_prefix(self):
        assert _extract_netuid("my-validator-99") == 99

    def test_falls_back_to_46(self):
        assert _extract_netuid("no-number-here") == 46

    def test_extracts_first_number(self):
        assert _extract_netuid("v2-subnet-46") == 2


class TestGetRevealedCommitments:
    def test_returns_empty_when_no_subtensor(self, pylon_config):
        client = ChainClient(pylon_config, subtensor=None)
        result = client._get_revealed_commitments()
        assert result == {}

    def test_parses_revealed_commitments(self, pylon_config):
        mock_subtensor = MagicMock()
        mock_subtensor.get_all_revealed_commitments.return_value = {
            "hotkey_alice": (
                (7000000, '{"h":"abc123","r":"alice/model"}'),
            ),
            "hotkey_bob": (
                (6900000, '{"h":"def456","r":"bob/model-v1"}'),
                (7100000, '{"h":"ghi789","r":"bob/model-v2"}'),
            ),
        }

        client = ChainClient(pylon_config, subtensor=mock_subtensor)
        result = client._get_revealed_commitments()

        assert len(result) == 2
        assert result["hotkey_alice"].hf_repo_id == "alice/model"
        assert result["hotkey_alice"].model_hash == "abc123"
        assert result["hotkey_alice"].block_number == 7000000
        # Bob should use latest entry (v2)
        assert result["hotkey_bob"].hf_repo_id == "bob/model-v2"
        assert result["hotkey_bob"].model_hash == "ghi789"
        assert result["hotkey_bob"].block_number == 7100000

    def test_skips_invalid_json(self, pylon_config):
        mock_subtensor = MagicMock()
        mock_subtensor.get_all_revealed_commitments.return_value = {
            "hotkey_good": ((7000000, '{"h":"abc","r":"good/model"}'),),
            "hotkey_bad": ((7000000, "not-json"),),
        }

        client = ChainClient(pylon_config, subtensor=mock_subtensor)
        result = client._get_revealed_commitments()

        assert len(result) == 1
        assert "hotkey_good" in result
        assert "hotkey_bad" not in result

    def test_returns_empty_on_exception(self, pylon_config):
        mock_subtensor = MagicMock()
        mock_subtensor.get_all_revealed_commitments.side_effect = Exception("chain error")

        client = ChainClient(pylon_config, subtensor=mock_subtensor)
        result = client._get_revealed_commitments()

        assert result == {}


class TestGetAllCommitmentsMerge:
    """Tests that get_all_commitments merges CommitmentOf + RevealedCommitments."""

    @pytest.mark.asyncio
    async def test_merges_revealed_with_pylon(self, pylon_config):
        """Miners only in RevealedCommitments should be included."""
        mock_subtensor = MagicMock()
        mock_subtensor.get_all_revealed_commitments.return_value = {
            "hotkey_alice": ((7000000, '{"h":"aaa","r":"alice/model"}'),),
            "hotkey_charlie": ((7000000, '{"h":"ccc","r":"charlie/model"}'),),
        }

        client = ChainClient(pylon_config, subtensor=mock_subtensor)

        # Mock pylon client — returns only alice and bob
        mock_pylon = AsyncMock()
        mock_response = MagicMock()
        mock_response.commitments = {
            "hotkey_alice": _hex_encode({"h": "aaa_new", "r": "alice/model-v2"}),
            "hotkey_bob": _hex_encode({"h": "bbb", "r": "bob/model"}),
        }
        mock_pylon.identity.get_commitments = AsyncMock(return_value=mock_response)
        client._client = mock_pylon

        result = await client.get_all_commitments()

        hotkeys = {m.hotkey for m in result}
        assert hotkeys == {"hotkey_alice", "hotkey_bob", "hotkey_charlie"}

        # Alice should come from Pylon (CommitmentOf takes priority)
        alice = next(m for m in result if m.hotkey == "hotkey_alice")
        assert alice.hf_repo_id == "alice/model-v2"

        # Charlie should come from RevealedCommitments
        charlie = next(m for m in result if m.hotkey == "hotkey_charlie")
        assert charlie.hf_repo_id == "charlie/model"

    @pytest.mark.asyncio
    async def test_works_without_subtensor(self, pylon_config):
        """Without subtensor, only Pylon results are returned."""
        client = ChainClient(pylon_config, subtensor=None)

        mock_pylon = AsyncMock()
        mock_response = MagicMock()
        mock_response.commitments = {
            "hotkey_bob": _hex_encode({"h": "bbb", "r": "bob/model"}),
        }
        mock_pylon.identity.get_commitments = AsyncMock(return_value=mock_response)
        client._client = mock_pylon

        result = await client.get_all_commitments()

        assert len(result) == 1
        assert result[0].hotkey == "hotkey_bob"

    @pytest.mark.asyncio
    async def test_revealed_failure_does_not_break_pylon(self, pylon_config):
        """If RevealedCommitments query fails, Pylon results still work."""
        mock_subtensor = MagicMock()
        mock_subtensor.get_all_revealed_commitments.side_effect = Exception("boom")

        client = ChainClient(pylon_config, subtensor=mock_subtensor)

        mock_pylon = AsyncMock()
        mock_response = MagicMock()
        mock_response.commitments = {
            "hotkey_bob": _hex_encode({"h": "bbb", "r": "bob/model"}),
        }
        mock_pylon.identity.get_commitments = AsyncMock(return_value=mock_response)
        client._client = mock_pylon

        result = await client.get_all_commitments()

        assert len(result) == 1
        assert result[0].hotkey == "hotkey_bob"
