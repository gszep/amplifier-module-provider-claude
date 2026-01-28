"""Unit tests for streaming event emission.

Tests that the provider emits canonical Amplifier content block events
(content_block:start, content_block:delta, content_block:end,
thinking:delta, thinking:final) during the SDK query loop for real-time display.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest  # type: ignore[import-not-found]

from amplifier_module_provider_claude import ClaudeProvider


# ---------------------------------------------------------------------------
# Mock coordinator
# ---------------------------------------------------------------------------

class MockCoordinator:
    """Mock coordinator that captures emitted events."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []
        self.hooks = self

    async def emit(self, event: str, data: dict):
        self.events.append((event, data))


# ---------------------------------------------------------------------------
# Mock SDK types – used for isinstance checks inside _execute_sdk_query
# ---------------------------------------------------------------------------

class MockSDKTextBlock:
    def __init__(self, text: str):
        self.text = text


class MockSDKThinkingBlock:
    def __init__(self, thinking: str, signature: str = "sig_mock"):
        self.thinking = thinking
        self.signature = signature


class MockSDKAssistantMessage:
    def __init__(self, content, error=None):
        self.content = content
        self.error = error


class MockSDKResultMessage:
    def __init__(
        self,
        result=None,
        session_id="sess_mock",
        duration_ms=100,
        total_cost_usd=0.01,
        num_turns=1,
        usage=None,
    ):
        self.result = result
        self.session_id = session_id
        self.duration_ms = duration_ms
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.usage = usage or {"input_tokens": 10, "output_tokens": 20}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdk_query(messages):
    """Return an async-generator factory that yields *messages* in order."""

    async def _mock(**kwargs):
        for msg in messages:
            yield msg

    return _mock


@contextmanager
def mock_sdk(messages):
    """Patch SDK types and query so _execute_sdk_query runs against mocks."""
    with (
        patch(
            "amplifier_module_provider_claude.sdk_query",
            _make_sdk_query(messages),
        ),
        patch(
            "amplifier_module_provider_claude.SDKAssistantMessage",
            MockSDKAssistantMessage,
        ),
        patch(
            "amplifier_module_provider_claude.SDKResultMessage",
            MockSDKResultMessage,
        ),
        patch(
            "amplifier_module_provider_claude.SDKTextBlock",
            MockSDKTextBlock,
        ),
        patch(
            "amplifier_module_provider_claude.SDKThinkingBlock",
            MockSDKThinkingBlock,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    coordinator = MockCoordinator()
    return ClaudeProvider(config={}, coordinator=coordinator)


@pytest.fixture
def coordinator(provider):
    return provider.coordinator


# ---------------------------------------------------------------------------
# Tests – text streaming
# ---------------------------------------------------------------------------

class TestTextStreamingEvents:
    """Canonical content_block events for text blocks."""

    @pytest.mark.asyncio
    async def test_text_block_emits_start_delta_end(self, provider, coordinator):
        """A single text block should produce start → delta → end."""
        msgs = [
            MockSDKAssistantMessage(content=[MockSDKTextBlock("Hello")]),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        names = [e[0] for e in coordinator.events]
        assert "content_block:start" in names
        assert "content_block:delta" in names
        assert "content_block:end" in names

        start_idx = names.index("content_block:start")
        delta_idx = names.index("content_block:delta")
        end_idx = names.index("content_block:end")
        assert start_idx < delta_idx < end_idx

    @pytest.mark.asyncio
    async def test_text_delta_payload(self, provider, coordinator):
        """content_block:delta for text must carry type and text."""
        msgs = [
            MockSDKAssistantMessage(content=[MockSDKTextBlock("world")]),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        deltas = [d for n, d in coordinator.events if n == "content_block:delta"]
        assert len(deltas) == 1
        assert deltas[0]["type"] == "text"
        assert deltas[0]["text"] == "world"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_emit_one_start(self, provider, coordinator):
        """Multiple partial text blocks should open only one content_block:start."""
        msgs = [
            MockSDKAssistantMessage(content=[MockSDKTextBlock("chunk1")]),
            MockSDKAssistantMessage(content=[MockSDKTextBlock("chunk2")]),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        starts = [d for n, d in coordinator.events
                  if n == "content_block:start" and d.get("type") == "text"]
        deltas = [d for n, d in coordinator.events
                  if n == "content_block:delta" and d.get("type") == "text"]
        ends = [d for n, d in coordinator.events
                if n == "content_block:end" and d.get("type") == "text"]

        assert len(starts) == 1, "Only one start per logical text block"
        assert len(deltas) == 2, "One delta per partial message"
        assert len(ends) == 1, "Only one end per logical text block"


# ---------------------------------------------------------------------------
# Tests – thinking streaming
# ---------------------------------------------------------------------------

class TestThinkingStreamingEvents:
    """Canonical thinking events for thinking blocks."""

    @pytest.mark.asyncio
    async def test_thinking_block_emits_start_delta_final_end(
        self, provider, coordinator
    ):
        """A thinking block should produce start → thinking:delta → thinking:final → end."""
        msgs = [
            MockSDKAssistantMessage(
                content=[MockSDKThinkingBlock("Let me reason")]
            ),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        names = [e[0] for e in coordinator.events]
        assert "content_block:start" in names
        assert "thinking:delta" in names
        assert "thinking:final" in names
        assert "content_block:end" in names

        # Ordering: start before delta, delta before final, final before end
        start_i = names.index("content_block:start")
        delta_i = names.index("thinking:delta")
        final_i = names.index("thinking:final")
        end_i = [i for i, n in enumerate(names) if n == "content_block:end"
                 and coordinator.events[i][1].get("type") == "thinking"][0]
        assert start_i < delta_i < final_i < end_i

    @pytest.mark.asyncio
    async def test_thinking_delta_payload(self, provider, coordinator):
        """thinking:delta must carry the thinking text."""
        msgs = [
            MockSDKAssistantMessage(
                content=[MockSDKThinkingBlock("deep thought")]
            ),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        deltas = [d for n, d in coordinator.events if n == "thinking:delta"]
        assert len(deltas) == 1
        assert deltas[0]["text"] == "deep thought"

    @pytest.mark.asyncio
    async def test_thinking_final_contains_full_text(self, provider, coordinator):
        """thinking:final must contain the complete thinking text."""
        msgs = [
            MockSDKAssistantMessage(
                content=[MockSDKThinkingBlock("complete reasoning")]
            ),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        finals = [d for n, d in coordinator.events if n == "thinking:final"]
        assert len(finals) == 1
        assert finals[0]["text"] == "complete reasoning"


# ---------------------------------------------------------------------------
# Tests – mixed content
# ---------------------------------------------------------------------------

class TestMixedStreamingEvents:
    """Thinking + text in the same response."""

    @pytest.mark.asyncio
    async def test_thinking_and_text_produce_both_event_types(
        self, provider, coordinator
    ):
        """Both thinking and text events should appear when both block types are present."""
        msgs = [
            MockSDKAssistantMessage(
                content=[
                    MockSDKThinkingBlock("reasoning"),
                    MockSDKTextBlock("answer"),
                ]
            ),
            MockSDKResultMessage(),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        names = [e[0] for e in coordinator.events]
        assert "thinking:delta" in names
        assert "thinking:final" in names

        text_deltas = [
            d for n, d in coordinator.events
            if n == "content_block:delta" and d.get("type") == "text"
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0]["text"] == "answer"

    @pytest.mark.asyncio
    async def test_no_content_emits_no_block_events(self, provider, coordinator):
        """Empty content should produce zero content_block / thinking events."""
        msgs = [
            MockSDKAssistantMessage(content=[]),
            MockSDKResultMessage(result="fallback"),
        ]

        with mock_sdk(msgs):
            await provider._execute_sdk_query("prompt", "sonnet", None)

        block_events = [
            e for e in coordinator.events
            if e[0].startswith("content_block:") or e[0].startswith("thinking:")
        ]
        assert len(block_events) == 0


# ---------------------------------------------------------------------------
# Tests – SDKResultMessage overwrite fix
# ---------------------------------------------------------------------------

class TestResultMessageOverwrite:
    """Accumulated text from partial messages must not be replaced by msg.result."""

    @pytest.mark.asyncio
    async def test_accumulated_text_preserved_over_result(self, provider, coordinator):
        """When partial messages yielded text, msg.result must not overwrite it."""
        raw = (
            "Here is my response\n\n"
            '<tool_use>\n{"tool": "search", "id": "call_1", "input": {"q": "test"}}\n</tool_use>'
        )
        msgs = [
            MockSDKAssistantMessage(content=[MockSDKTextBlock(raw)]),
            # msg.result is a cleaned version that strips tool_use blocks
            MockSDKResultMessage(result="Here is my response"),
        ]

        with mock_sdk(msgs):
            result = await provider._execute_sdk_query("prompt", "sonnet", None)

        assert result["text"] == raw
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_result_used_as_fallback_when_no_partial_text(
        self, provider, coordinator
    ):
        """msg.result should be used when no text arrived from partial messages."""
        msgs = [
            MockSDKAssistantMessage(content=[]),
            MockSDKResultMessage(result="fallback response"),
        ]

        with mock_sdk(msgs):
            result = await provider._execute_sdk_query("prompt", "sonnet", None)

        assert result["text"] == "fallback response"
