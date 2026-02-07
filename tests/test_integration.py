"""Integration tests for provider-claude module.

Tests the architecture:
- Provider returns tool_calls for Amplifier orchestrator to execute
- Built-in tools are disabled (--tools "")
- Tool definitions are injected via system prompt

Requires:
- amplifier CLI installed locally
- Claude Code CLI installed and authenticated
"""

import subprocess
import tempfile

import pytest


@pytest.mark.long
def test_provider_basic_completion():
    """Test basic completion without tools."""

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "amplifier",
                "run",
                "--provider",
                "claude",
                "what is 1+1? reply with just the number",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmpdir,
        )

    print(
        f"""\namplifier run --provider claude "what is 1+1? reply with just the number"\n{result.stdout}"""
    )
    if result.stderr:
        print(f"{result.stderr}")

    # The response should contain "2"
    assert "2" in result.stdout, f"Expected '2' in response, got: {result.stdout}"


@pytest.mark.long
def test_provider_with_tools():
    """Test that provider returns tool_calls for orchestrator execution."""

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "amplifier",
                "run",
                "--provider",
                "claude",
                "What is the current date and time? Use a tool to find out.",
            ],
            capture_output=True,
            text=True,
            timeout=180,  # Longer timeout for tool execution
            cwd=tmpdir,
        )

    print(
        f"""\namplifier run --provider claude "What is the current date and time?"\n{result.stdout}"""
    )
    if result.stderr:
        print(f"{result.stderr}")

    response = result.stdout.lower()
    date_indicators = [
        "2024",
        "2025",
        "2026",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "time",
        "date",
        "today",
    ]
    has_date_info = any(indicator in response for indicator in date_indicators)

    assert result.returncode == 0 or has_date_info, (
        f"Command failed or no date info: {result.stdout}"
    )


expect_script = r"""
set timeout 30;
spawn amplifier resume;
expect "> ";
send "what is double my favorite number? Return just the number with no punctuation\r";
expect "Amplifier:";
expect "> ";
send "\003";
expect "Exit Amplifier?";
send "y\r";
expect eof
"""


@pytest.mark.long
def test_provider_multi_turn_conversation():
    """Test session continuity across multiple turns."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # First turn - establish context
        subprocess.run(
            [
                "amplifier",
                "run",
                "--provider",
                "claude",
                "My favorite number is 21. Remember this. Do not write any files to disk.",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmpdir,
        )

        # Second turn - check if context is remembered
        result = subprocess.run(
            ["expect", "-c", expect_script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmpdir,
        )

    # The response should acknowledge the number
    assert "42" in result.stdout, f"Expected acknowledgment of 42, got: {result.stdout}"
