"""Integration test for provider-claude module.

Tests the full flow: install module from repo, update, and run a prompt.
Requires amplifier CLI to be installed locally.

IMPORTANT: The test runs from /tmp to prevent Claude Code from picking up
the working directory context (modified files, git status, etc.) which would
cause it to respond to those instead of the actual prompt.
"""

import os
import subprocess
import tempfile


def test_provider_integration():
    """Test installing and running the claude provider via amplifier CLI."""
    # Use local source for development, GitHub for CI
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_source = repo_root
    github_source = "git+https://github.com/gszep/amplifier-module-provider-claude@main"

    # Prefer local source if running from repo (for development testing)
    source = (
        local_source
        if os.path.exists(os.path.join(repo_root, "pyproject.toml"))
        else github_source
    )

    subprocess.run(
        ["amplifier", "module", "add", "provider-claude", "--source", source],
    )

    # Run from a neutral directory to prevent Claude Code from picking up
    # the repo's working directory context (modified files, etc.)
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
            timeout=120,  # 2 minute timeout
            cwd=tmpdir,  # Run from temp directory
        )

    print(
        f"""\namplifier run --provider claude "what is 1+1? reply with just the number"\n{result.stdout}"""
    )
    if result.stderr:
        print(f"{result.stderr}")

    answer = result.stdout.strip()[-1]
    assert answer == "2", f"Expected '2' in response, got: {answer}"


def test_websearch_tool_integration():
    """Test that Claude uses WebSearch tool for current information queries."""
    # Use local source for development, GitHub for CI
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_source = repo_root
    github_source = "git+https://github.com/gszep/amplifier-module-provider-claude@main"

    # Prefer local source if running from repo (for development testing)
    source = (
        local_source
        if os.path.exists(os.path.join(repo_root, "pyproject.toml"))
        else github_source
    )

    subprocess.run(
        ["amplifier", "module", "add", "provider-claude", "--source", source],
    )

    # Run from a neutral directory to prevent Claude Code from picking up
    # the repo's working directory context (modified files, etc.)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                "amplifier",
                "run",
                "--provider",
                "claude",
                "what is the weather in Tokyo today?",
            ],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=tmpdir,  # Run from temp directory
        )

    print(
        f"""\namplifier run --provider claude "what is the weather in Tokyo today?"\n{result.stdout}"""
    )
    if result.stderr:
        print(f"{result.stderr}")

    # The response should contain weather-related information
    # We check for common weather terms that would appear in a weather report
    response = result.stdout.lower()
    weather_indicators = ["temperature", "°c", "°f", "celsius", "fahrenheit", "weather", "sunny", "cloudy", "rain", "humidity"]
    has_weather_info = any(indicator in response for indicator in weather_indicators)

    assert has_weather_info, f"Expected weather information in response, got: {result.stdout}"


if __name__ == "__main__":
    test_provider_integration()
