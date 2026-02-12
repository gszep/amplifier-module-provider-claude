"""Tests for tool-name validation in _parse_tool_blocks_from_text.

Covers the fix for phantom tool calls (first_tool/second_tool) caused by
echoed TOOL_USE_REMINDER examples being parsed as real tool invocations.
"""

import json

from amplifier_module_provider_claude import ClaudeProvider


def _make_provider(**kwargs) -> ClaudeProvider:
    return ClaudeProvider(**kwargs)


# =============================================================================
# Tool-name validation
# =============================================================================


def test_valid_tool_names_are_kept():
    """Tool blocks with names in available_tool_names pass validation."""
    provider = _make_provider()
    provider._available_tool_names = {"edit_file", "read_file", "grep"}

    text = '[tool]: {"type": "tool_use", "name": "edit_file", "id": "ef01", "input": {"path": "a.py"}}'
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 1
    assert blocks[0].name == "edit_file"
    assert blocks[0].id == "ef01"


def test_unknown_tool_names_are_filtered():
    """Tool blocks with names NOT in available_tool_names are rejected."""
    provider = _make_provider()
    provider._available_tool_names = {"edit_file", "read_file", "grep"}

    text = '[tool]: {"type": "tool_use", "name": "first_tool", "id": "tl4xcu5", "input": {"param1": "value1", "param2": 42}}'
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 0  # filtered out


def test_mixed_valid_and_invalid_tools():
    """Only valid tool names are kept; invalid ones are filtered."""
    provider = _make_provider()
    provider._available_tool_names = {"edit_file", "read_file"}

    text = (
        '[tool]: {"type": "tool_use", "name": "first_tool", "id": "tl4xcu5", "input": {"param1": "value1"}}\n'
        '[tool]: {"type": "tool_use", "name": "second_tool", "id": "tl4t214", "input": {"param1": "valueX"}}\n'
        '[tool]: {"type": "tool_use", "name": "edit_file", "id": "ef05sys", "input": {"path": "a.py"}}'
    )
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 1
    assert blocks[0].name == "edit_file"
    assert blocks[0].id == "ef05sys"


def test_text_cleaned_regardless_of_validation():
    """ALL parsed [tool]: spans are stripped from text, even filtered ones."""
    provider = _make_provider()
    provider._available_tool_names = {"edit_file"}

    text = (
        "Some text before\n"
        '[tool]: {"type": "tool_use", "name": "first_tool", "id": "tl4xcu5", "input": {"p": 1}}\n'
        "Some text between\n"
        '[tool]: {"type": "tool_use", "name": "edit_file", "id": "ef01", "input": {"path": "a.py"}}\n'
        "Some text after"
    )
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 1  # only edit_file
    # Both [tool]: spans should be removed from text
    assert "[tool]:" not in cleaned
    assert "Some text before" in cleaned
    assert "Some text between" in cleaned
    assert "Some text after" in cleaned


def test_no_available_tools_skips_validation():
    """When _available_tool_names is empty, no filtering is applied."""
    provider = _make_provider()
    provider._available_tool_names = set()  # empty = no filtering

    text = '[tool]: {"type": "tool_use", "name": "anything", "id": "x1", "input": {"a": 1}}'
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 1
    assert blocks[0].name == "anything"


# =============================================================================
# Regression: exact reproduction of the d285ed4d phantom tool scenario
# =============================================================================


def test_phantom_tool_call_scenario():
    """Exact reproduction of the d285ed4d session bug.

    Claude received only the TOOL_USE_REMINDER and echoed the examples
    (first_tool, second_tool) alongside a legitimate edit_file call.
    The parser should keep edit_file and reject the examples.
    """
    provider = _make_provider()
    provider._available_tool_names = {
        "edit_file",
        "read_file",
        "grep",
        "glob",
        "bash",
        "write_file",
        "delegate",
        "todo",
        "python_check",
        "LSP",
        "web_search",
        "web_fetch",
    }

    # Simulated Claude response text from the incident
    first_tool = json.dumps(
        {
            "type": "tool_use",
            "name": "first_tool",
            "id": "tl4xcu5",
            "input": {"param1": "value1", "param2": 42},
        }
    )
    second_tool = json.dumps(
        {
            "type": "tool_use",
            "name": "second_tool",
            "id": "tl4t214",
            "input": {"param1": "valueX"},
        }
    )
    edit_file = json.dumps(
        {
            "type": "tool_use",
            "name": "edit_file",
            "id": "ef05sys",
            "input": {
                "file_path": "/home/user/claude.py",
                "old_string": "old code",
                "new_string": "new code",
            },
        }
    )

    text = (
        '[tool]: {"id": "rf04cla", "output": {"file_path": "claude.py", "content": "file contents"}}\n'
        '[tool]: {"id": "pc01fmt", "output": {"success": true, "clean": true}}\n\n'
        '<system-reminder source="hooks-status-context">environment info</system-reminder>\n\n'
        '<system-reminder source="hooks-tools-reminder">\n'
        "You have access to tools.\n"
        "<example>\n"
        f"[tool]: {first_tool}\n"
        f"[tool]: {second_tool}\n"
        "</example>\n"
        "</system-reminder>\n\n"
        "Now I can see the exact string in claude.py...\n\n"
        f"[tool]: {edit_file}"
    )

    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    # Only edit_file should survive validation
    assert len(blocks) == 1
    assert blocks[0].name == "edit_file"
    assert blocks[0].id == "ef05sys"

    # The hallucinated tool results (rf04cla, pc01fmt) don't have "type": "tool_use"
    # so they fail AnthropicToolUseBlock validation and aren't parsed at all.
    # The example tools (first_tool, second_tool) ARE parsed but filtered.
    # The text should be cleaned of all [tool]: spans.
    assert "first_tool" not in cleaned or "[tool]:" not in cleaned


# =============================================================================
# Existing behavior preserved: fenced code blocks are skipped
# =============================================================================


def test_fenced_code_blocks_still_skipped():
    """Tool blocks inside fenced code should not be parsed."""
    provider = _make_provider()
    provider._available_tool_names = {"edit_file"}

    text = (
        "Here's an example:\n"
        "```\n"
        '[tool]: {"type": "tool_use", "name": "edit_file", "id": "x1", "input": {"a": 1}}\n'
        "```\n"
        "That was the example."
    )
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 0  # inside fenced code, not parsed
    assert "[tool]:" in cleaned  # preserved in text


def test_inline_code_blocks_still_skipped():
    """Tool blocks inside inline code should not be parsed."""
    provider = _make_provider()
    provider._available_tool_names = {"edit_file"}

    text = 'Use `[tool]: {"type": "tool_use"}` syntax to call tools.'
    blocks, cleaned = provider._parse_tool_blocks_from_text(text)

    assert len(blocks) == 0
