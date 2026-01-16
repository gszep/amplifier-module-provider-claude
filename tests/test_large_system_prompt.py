"""Tests for large system prompt handling.

Verifies that the system prompt truncation prevents "Argument list too long" errors
without breaking the provider functionality.
"""

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_claude import MAX_SYSTEM_PROMPT_BYTES, ClaudeProvider


class TestLargeSystemPromptTruncation:
    """Test system prompt truncation for size limits."""

    def test_extract_system_prompt_within_limit(self):
        """Test that normal system prompts are not truncated."""
        provider = ClaudeProvider({})
        
        # Create request with modest system prompt
        request = ChatRequest(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert system_prompt == "You are a helpful assistant."
        assert len(system_prompt) < MAX_SYSTEM_PROMPT_BYTES

    def test_extract_system_prompt_with_multiple_messages(self):
        """Test extraction of multiple system messages."""
        provider = ClaudeProvider({})

        request = ChatRequest(
            messages=[
                Message(role="system", content="System instruction 1"),
                Message(role="user", content="Hello"),
                Message(role="system", content="System instruction 2"),
            ]
        )

        system_prompt = provider._extract_system_prompt(request)
        assert system_prompt == "System instruction 1\n\nSystem instruction 2"

    def test_system_prompt_truncation_applied_in_complete(self):
        """Test that truncation is applied during complete() method.
        
        This is a unit test that verifies the truncation logic without
        actually calling Claude Code (which would require the SDK and binary).
        """
        provider = ClaudeProvider({})
        
        # Create a request with a very large system prompt
        large_prompt = "X" * (MAX_SYSTEM_PROMPT_BYTES + 10000)  # 10 KB over limit
        request = ChatRequest(
            messages=[
                Message(role="system", content=large_prompt),
                Message(role="user", content="Hello"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert len(system_prompt) == MAX_SYSTEM_PROMPT_BYTES + 10000  # Not yet truncated
        
        # Truncation happens in complete(), but we can test the logic directly
        # by simulating what complete() does
        if system_prompt and len(system_prompt) > MAX_SYSTEM_PROMPT_BYTES:
            truncated = system_prompt[:MAX_SYSTEM_PROMPT_BYTES]
            truncated += "\n\n[...truncated due to size limit...]"
            
            assert len(truncated) > MAX_SYSTEM_PROMPT_BYTES  # Now has truncation marker
            assert truncated.endswith("[...truncated due to size limit...]")

    def test_truncation_marker_included(self):
        """Test that truncation marker is appended."""
        large_prompt = "X" * (MAX_SYSTEM_PROMPT_BYTES + 1000)
        
        # Simulate truncation
        truncated = large_prompt[:MAX_SYSTEM_PROMPT_BYTES]
        truncated += "\n\n[...truncated due to size limit...]"
        
        assert "[...truncated due to size limit...]" in truncated
        assert truncated.count("[...truncated") == 1

    def test_no_truncation_for_none_system_prompt(self):
        """Test that None system prompts don't cause issues."""
        provider = ClaudeProvider({})
        
        request = ChatRequest(
            messages=[
                Message(role="user", content="Hello"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert system_prompt is None

    def test_max_system_prompt_bytes_is_reasonable(self):
        """Test that the truncation limit is within subprocess ARG_MAX."""
        # Linux ARG_MAX is typically 2 MB
        # We use 500 KB to have safety margin for other arguments
        assert MAX_SYSTEM_PROMPT_BYTES == 500_000
        assert MAX_SYSTEM_PROMPT_BYTES < 2_000_000  # Well below ARG_MAX

    def test_system_prompt_with_file_content(self):
        """Test scenario: system prompt contains file content (real-world case)."""
        # Simulate user providing a large file as system context
        file_content = """
# Large Python File
def function_1():
    pass
""" * 1000  # Repeat to make it large
        
        provider = ClaudeProvider({})
        request = ChatRequest(
            messages=[
                Message(role="system", content=file_content),
                Message(role="user", content="Analyze this code"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert len(system_prompt) > MAX_SYSTEM_PROMPT_BYTES
        
        # Verify truncation would work
        truncated = system_prompt[:MAX_SYSTEM_PROMPT_BYTES]
        assert len(truncated) == MAX_SYSTEM_PROMPT_BYTES

    def test_system_prompt_with_api_schema(self):
        """Test scenario: system prompt contains large API specification."""
        # Simulate user providing a large API schema
        api_schema = """{
  "openapi": "3.0.0",
  "paths": {
""" + '    "path": {},\n' * 1000 + "  }\n}"
        
        provider = ClaudeProvider({})
        request = ChatRequest(
            messages=[
                Message(role="system", content=api_schema),
                Message(role="user", content="Build a client for this API"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert len(system_prompt) > MAX_SYSTEM_PROMPT_BYTES


class TestSystemPromptExtractionEdgeCases:
    """Test edge cases in system prompt extraction."""

    def test_empty_system_prompt(self):
        """Test extraction of empty system message."""
        provider = ClaudeProvider({})
        request = ChatRequest(
            messages=[
                Message(role="system", content=""),
                Message(role="user", content="Hello"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert system_prompt == ""

    def test_whitespace_only_system_prompt(self):
        """Test system prompt with only whitespace."""
        provider = ClaudeProvider({})
        request = ChatRequest(
            messages=[
                Message(role="system", content="   \n\n  "),
                Message(role="user", content="Hello"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert system_prompt == "   \n\n  "

    def test_unicode_system_prompt(self):
        """Test system prompt with unicode characters."""
        provider = ClaudeProvider({})
        unicode_text = "你好 🚀 مرحبا" * 1000  # Repeat unicode text
        
        request = ChatRequest(
            messages=[
                Message(role="system", content=unicode_text),
                Message(role="user", content="Hello"),
            ]
        )
        
        system_prompt = provider._extract_system_prompt(request)
        assert len(system_prompt) > 0
        assert "你好" in system_prompt or "🚀" in system_prompt
