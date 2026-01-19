"""Unit tests for tool result validation and repair.

Tests that the provider correctly detects missing tool results
and creates synthetic error messages to prevent protocol violations.
"""

import pytest  # type: ignore[import-not-found]

from amplifier_core.message_models import Message, ToolCallBlock  # type: ignore

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


class TestToolResultValidation:
    """Test tool result detection and validation."""

    def test_find_missing_tool_results_detects_unpaired_calls(self, provider):
        """Tool calls without corresponding results should be detected.

        When an assistant message contains tool_use blocks but there's no
        tool message with matching tool_call_id, the validation should
        detect this and return the missing pair information.
        """
        messages = [
            Message(role="user", content="What time is it?"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call_123",
                        name="get_time",
                        input={"timezone": "UTC"},
                    )
                ],
            ),
            # NO tool result message - this is the bug scenario
            Message(role="user", content="Still waiting..."),
        ]

        missing = provider._find_missing_tool_results(messages)

        assert len(missing) == 1, f"Expected 1 missing result, got {len(missing)}"
        msg_idx, call_id, tool_name, tool_args = missing[0]
        assert call_id == "call_123"
        assert tool_name == "get_time"
        assert tool_args == {"timezone": "UTC"}

    def test_find_missing_tool_results_returns_empty_when_paired(self, provider):
        """Tool calls with matching results should not be flagged."""
        messages = [
            Message(role="user", content="What time is it?"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call_123",
                        name="get_time",
                        input={"timezone": "UTC"},
                    )
                ],
            ),
            Message(
                role="tool",
                content="2024-01-15 10:30:00 UTC",
                tool_call_id="call_123",
                name="get_time",
            ),
            Message(role="assistant", content="The time is 10:30 AM UTC."),
        ]

        missing = provider._find_missing_tool_results(messages)

        assert len(missing) == 0, f"Expected no missing results, got {len(missing)}"

    def test_find_missing_tool_results_handles_multiple_calls(self, provider):
        """Multiple tool calls should each be checked for results."""
        messages = [
            Message(role="user", content="Get time and weather"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call_time",
                        name="get_time",
                        input={},
                    ),
                    ToolCallBlock(
                        id="call_weather",
                        name="get_weather",
                        input={"city": "NYC"},
                    ),
                ],
            ),
            # Only one result provided - weather is missing
            Message(
                role="tool",
                content="10:30 AM",
                tool_call_id="call_time",
                name="get_time",
            ),
            Message(role="user", content="What about weather?"),
        ]

        missing = provider._find_missing_tool_results(messages)

        assert len(missing) == 1, f"Expected 1 missing result, got {len(missing)}"
        _, call_id, tool_name, _ = missing[0]
        assert call_id == "call_weather"
        assert tool_name == "get_weather"

    def test_repaired_ids_not_detected_again(self, provider):
        """Tool IDs that were already repaired should not be re-detected.

        This prevents infinite loops where the same missing result
        is detected and repaired repeatedly.
        """
        # Pre-mark an ID as repaired
        provider._repaired_tool_ids.add("call_123")

        messages = [
            Message(role="user", content="What time is it?"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(
                        id="call_123",
                        name="get_time",
                        input={},
                    )
                ],
            ),
            # No tool result, but ID is already in repaired set
        ]

        missing = provider._find_missing_tool_results(messages)

        assert len(missing) == 0, "Repaired IDs should not be re-detected as missing"


class TestSyntheticResultCreation:
    """Test synthetic error result creation."""

    def test_create_synthetic_result_has_correct_structure(self, provider):
        """Synthetic results should be valid Message objects."""
        result = provider._create_synthetic_result(
            call_id="call_abc", tool_name="my_tool"
        )

        assert result.role == "tool"
        assert result.tool_call_id == "call_abc"
        assert result.name == "my_tool"
        assert "SYSTEM ERROR" in result.content
        assert "missing" in result.content.lower()

    def test_create_synthetic_result_includes_tool_info(self, provider):
        """Synthetic results should include tool identification for debugging."""
        result = provider._create_synthetic_result(
            call_id="call_xyz", tool_name="search_web"
        )

        assert "search_web" in result.content
        assert "call_xyz" in result.content
