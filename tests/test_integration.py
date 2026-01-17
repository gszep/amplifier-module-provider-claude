"""Integration test for provider-claude module.

Tests the full flow: install module from repo, update, and run a prompt.
Requires amplifier CLI to be installed locally.
"""

import subprocess


def test_provider_integration():
    """Test installing and running the claude provider via amplifier CLI."""
    repo_url = "git+https://github.com/gszep/amplifier-module-provider-claude@main"

    subprocess.run(
        ["amplifier", "module", "add", "provider-claude", "--source", repo_url],
    )
    subprocess.run(
        ["amplifier", "module", "update", "provider-claude"],
    )

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
    )

    print(
        f"""\namplifier run --provider claude "what is 1+1? reply with just the number"\n{result.stdout}"""
    )
    if result.stderr:
        print(f"{result.stderr}")

    answer = result.stdout.strip()[-1]
    assert answer == "2", f"Expected '2' in response, got: {answer}"


if __name__ == "__main__":
    test_provider_integration()
