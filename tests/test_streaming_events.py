"""Unit tests for streaming event emission.

Tests that content block events are emitted correctly,
particularly for multi-block responses.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest  # type: ignore[import-not-found]

from amplifier_module_provider_claude import ClaudeProvider


class MockCoordinator:
    """Mock coordinator that captures emitted events."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.hooks = self

    async def emit(self, event: str, data: dict):
        self.events.append((event, data))


@pytest.fixture
def provider():
    """Create a provider with mock coordinator."""
    coordinator = MockCoordinator()
    return ClaudeProvider(config={}, coordinator=coordinator)


@pytest.fixture
def coordinator(provider):
    """Get the mock coordinator from provider."""
    return provider.coordinator


class TestStreamingDeltaEvents:
    """Test that streaming delta events contain correct content."""

    @pytest.mark.asyncio
    async def test_single_text_block_delta_contains_block_text(
        self, provider, coordinator
    ):
        """Single text block: delta should contain that block's text."""
        # Simulate CLI output with single text block
        cli_output = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "Hello world"}]},
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "result": "Hello world",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "session_id": "test-session",
                }
            ),
        ]

        # Run with mocked subprocess
        await self._run_with_mocked_cli(provider, cli_output)

        # Find the content_block:delta events
        delta_events = [
            (event, data)
            for event, data in coordinator.events
            if event == "content_block:delta"
        ]

        assert len(delta_events) == 1, (
            f"Expected 1 delta event, got {len(delta_events)}"
        )
        _, delta_data = delta_events[0]
        assert delta_data["delta"]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_delta_contains_only_current_block(
        self, provider, coordinator
    ):
        """Multiple text blocks: each delta should contain only that block's text, not accumulated.

        This is the critical test for the streaming delta bug.
        When Claude CLI returns multiple text blocks, each CONTENT_BLOCK_DELTA
        should emit only the current block's text, not the accumulated text.
        """
        # Simulate CLI output with multiple text blocks (like Claude sometimes does)
        cli_output = [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "First block"},
                            {"type": "text", "text": "Second block"},
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "result": "First block\nSecond block",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "session_id": "test-session",
                }
            ),
        ]

        await self._run_with_mocked_cli(provider, cli_output)

        # Find the content_block:delta events
        delta_events = [
            (event, data)
            for event, data in coordinator.events
            if event == "content_block:delta"
        ]

        assert len(delta_events) == 2, (
            f"Expected 2 delta events, got {len(delta_events)}"
        )

        # CRITICAL: Each delta should contain ONLY its block's text
        # Not accumulated text from previous blocks
        _, first_delta = delta_events[0]
        _, second_delta = delta_events[1]

        # First delta should be "First block" only
        assert first_delta["delta"]["text"] == "First block", (
            f"First delta should be 'First block', got '{first_delta['delta']['text']}'"
        )

        # Second delta should be "Second block" only, NOT "First block\nSecond block"
        assert second_delta["delta"]["text"] == "Second block", (
            f"Second delta should be 'Second block', got '{second_delta['delta']['text']}'. "
            "This indicates the streaming delta bug - accumulated text is being emitted."
        )

    async def _run_with_mocked_cli(self, provider, cli_output_lines: list[str]):
        """Execute _execute_cli with mocked subprocess."""
        # Create mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0

        # Mock stdin
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdin.close = MagicMock()
        mock_process.stdin.wait_closed = AsyncMock()

        # Mock stdout as async iterator
        async def mock_stdout_iter():
            for line in cli_output_lines:
                yield (line + "\n").encode("utf-8")

        mock_process.stdout = mock_stdout_iter()

        # Mock stderr
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")

        # Mock wait
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            cmd = ["claude", "-p", "--model", "sonnet"]
            await provider._execute_cli(cmd, "test prompt")
