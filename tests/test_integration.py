"""Integration test for provider-claude module.

Tests the full flow: install module from repo, update, and run a prompt.
Requires amplifier CLI to be installed locally.
"""

import subprocess


def test_provider_integration():
    """Test installing and running the claude provider via amplifier CLI."""
    repo_url = "git+https://github.com/gszep/amplifier-module-provider-claude@main"

    # Step 1: Add/update the module from the repo
    print("\n=== Adding provider-claude module ===")
    result = subprocess.run(
        ["amplifier", "module", "add", "provider-claude", "--source", repo_url],
        capture_output=True,
        text=True,
    )
    # May fail if already added, that's OK
    print(f"Add result: {result.returncode}")
    if result.stdout:
        print(f"stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"stderr: {result.stderr[:500]}")

    # Step 2: Update the module to get latest changes
    print("\n=== Updating provider-claude module ===")
    result = subprocess.run(
        ["amplifier", "module", "update", "provider-claude"],
        capture_output=True,
        text=True,
    )
    print(f"Update result: {result.returncode}")
    if result.stdout:
        print(f"stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"stderr: {result.stderr[:500]}")

    # Step 3: Run a simple prompt
    print("\n=== Running test prompt ===")
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

    print(f"Run result: {result.returncode}")
    print(f"stdout: {result.stdout}")
    if result.stderr:
        print(f"stderr: {result.stderr}")

    # Verify we got a response containing "2"
    assert result.returncode == 0, f"amplifier run failed: {result.stderr}"
    assert "2" in result.stdout, f"Expected '2' in response, got: {result.stdout}"

    print("\n=== Test passed! ===")


if __name__ == "__main__":
    test_provider_integration()
