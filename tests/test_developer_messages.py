"""Unit tests for developer message handling.

Developer messages contain context files (@mentions) that should be
included in the conversation, wrapped in <context_file> tags.
"""

import pytest  # type: ignore[import-not-found]

from amplifier_core.message_models import Message  # type: ignore

from amplifier_module_provider_claude import ClaudeProvider


@pytest.fixture
def provider():
    """Create a provider instance."""
    return ClaudeProvider(config={})


class TestDeveloperMessageHandling:
    """Test that developer messages are properly handled."""

    def test_developer_messages_are_included(self, provider):
        """Developer messages should not be silently dropped.

        When a message has role='developer', it contains context files
        that need to be included in the conversation for Claude to use.
        """
        messages = [
            Message(
                role="developer",
                content="# Important Context\nThis is context from @file.md",
            ),
            Message(role="user", content="What does the context say?"),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        # The developer message content should appear somewhere in the output
        # Either in system_prompt or user_prompt
        all_content = system_prompt + user_prompt
        assert "Important Context" in all_content, (
            f"Developer message content not found in output.\n"
            f"System prompt: {system_prompt}\n"
            f"User prompt: {user_prompt}"
        )

    def test_developer_messages_wrapped_in_context_file_tags(self, provider):
        """Developer messages should be wrapped in <context_file> tags.

        Following the Anthropic provider pattern, developer messages
        should be wrapped in XML tags to clearly demarcate context.
        """
        messages = [
            Message(role="developer", content="Content from context file"),
            Message(role="user", content="Use the context"),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        # Check for context_file tags
        all_content = system_prompt + user_prompt
        assert "<context_file>" in all_content, (
            "Developer messages should be wrapped in <context_file> tags"
        )
        assert "</context_file>" in all_content, (
            "Developer messages should be wrapped in </context_file> tags"
        )
        assert "Content from context file" in all_content, (
            "Developer message content should be preserved"
        )

    def test_multiple_developer_messages(self, provider):
        """Multiple developer messages should all be included."""
        messages = [
            Message(role="developer", content="First context file"),
            Message(role="developer", content="Second context file"),
            Message(role="user", content="Use both contexts"),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        all_content = system_prompt + user_prompt
        assert "First context file" in all_content, "First developer message missing"
        assert "Second context file" in all_content, "Second developer message missing"

    def test_developer_messages_appear_before_user_messages(self, provider):
        """Developer messages (context) should appear before user's actual request.

        Following the Anthropic provider pattern, context files are presented
        first, then the user's actual message.
        """
        messages = [
            Message(role="developer", content="Context goes here"),
            Message(role="user", content="User request here"),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        # In user_prompt, context should appear before user request
        if "Context goes here" in user_prompt and "User request here" in user_prompt:
            context_pos = user_prompt.find("Context goes here")
            user_pos = user_prompt.find("User request here")
            assert context_pos < user_pos, (
                f"Context should appear before user request.\n"
                f"Context position: {context_pos}, User position: {user_pos}\n"
                f"User prompt: {user_prompt}"
            )


class TestHookSystemReminderHandling:
    """Test that hook-injected system reminders are formatted correctly.

    Hooks inject <system-reminder> content as user messages, but these
    should appear outside of <user> tags for cleaner prompt structure.
    """

    def test_system_reminder_not_wrapped_in_user_tags(self, provider):
        """System reminder content should not be wrapped in <user> tags.

        When a user message contains only <system-reminder> content (injected by hooks),
        it should be passed through without the <user> wrapper.
        """
        messages = [
            Message(role="user", content="What is 1+1?"),
            Message(
                role="user",
                content='<system-reminder source="hooks-status">Working dir: /home</system-reminder>',
            ),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        # The actual user message should be wrapped
        assert "<user>What is 1+1?</user>" in user_prompt, (
            "Regular user message should be wrapped in <user> tags"
        )

        # The system reminder should NOT be wrapped in <user> tags
        assert "<user><system-reminder" not in user_prompt, (
            "System reminder should NOT be wrapped in <user> tags"
        )
        assert '<system-reminder source="hooks-status">' in user_prompt, (
            "System reminder content should be present"
        )

    def test_system_reminder_appears_after_user_message(self, provider):
        """System reminder should appear after the user's actual message."""
        messages = [
            Message(role="user", content="Hello"),
            Message(
                role="user",
                content="<system-reminder>Hook content</system-reminder>",
            ),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        # Order should be preserved: user message then system reminder
        user_pos = user_prompt.find("<user>Hello</user>")
        reminder_pos = user_prompt.find(
            "<system-reminder>Hook content</system-reminder>"
        )

        assert user_pos < reminder_pos, (
            f"User message should appear before system reminder.\n"
            f"User position: {user_pos}, Reminder position: {reminder_pos}\n"
            f"User prompt: {user_prompt}"
        )

    def test_mixed_user_content_stays_wrapped(self, provider):
        """User content that contains system-reminder but isn't purely a hook should stay wrapped.

        If user types actual content before the system-reminder, keep the <user> wrapper.
        Note: This test requires multiple messages because single-message conversations
        have a special case that returns raw content without wrapping.
        """
        messages = [
            Message(role="system", content="You are helpful"),
            Message(
                role="user",
                content="My question is: <system-reminder>some text</system-reminder>",
            ),
        ]

        system_prompt, user_prompt = provider._convert_messages(messages, tools=None)

        # This should be wrapped because it doesn't START with <system-reminder>
        assert "<user>My question is:" in user_prompt, (
            f"Mixed content should stay wrapped in <user> tags.\n"
            f"Got user_prompt: {user_prompt}"
        )
