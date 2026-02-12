"""Tests for session-block-anchored incremental CLI delivery.

Covers _has_session_block() and _get_recent_messages() which use the
[session]: redacted_thinking block as a structural anchor to determine
which messages have already been delivered to the CLI.
"""

from amplifier_core.message_models import Message, ToolCallBlock, ToolResultBlock

from amplifier_module_provider_claude import ClaudeProvider, Session


def _make_provider() -> ClaudeProvider:
    provider = ClaudeProvider()
    return provider


def _make_session_block(session_id: str = "test-session") -> Session:
    """Create a Session block as the provider would append to responses."""
    session = Session()
    session.id = session_id
    return session


def _assistant_with_session(
    text: str = "", session_id: str = "test-session"
) -> Message:
    """Create an assistant message containing a session block."""
    blocks: list = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.append(_make_session_block(session_id))
    return Message(role="assistant", content=blocks)


# =============================================================================
# _has_session_block
# =============================================================================


def test_has_session_block_detects_session():
    """Assistant message with a session block is detected."""
    msg = _assistant_with_session("Hello")
    assert ClaudeProvider._has_session_block(msg) is True


def test_has_session_block_rejects_plain_assistant():
    """Assistant message without a session block is not detected."""
    msg = Message(role="assistant", content="Hello")
    assert ClaudeProvider._has_session_block(msg) is False


def test_has_session_block_rejects_user_message():
    """User messages are never session block carriers."""
    msg = Message(role="user", content="Hello")
    assert ClaudeProvider._has_session_block(msg) is False


def test_has_session_block_rejects_redacted_thinking_without_tag():
    """Redacted thinking without the [session]: tag is not a session block."""
    session = Session()
    session.data = "some other data"  # no [session]: prefix
    msg = Message(role="assistant", content=[session])
    assert ClaudeProvider._has_session_block(msg) is False


def test_has_session_block_detects_among_other_blocks():
    """Session block is found even when mixed with other content blocks."""
    msg = Message(
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "Let me think..."},
            ToolCallBlock(id="tc1", name="grep", input={"pattern": "test"}),
            _make_session_block(),
        ],
    )
    assert ClaudeProvider._has_session_block(msg) is True


# =============================================================================
# _get_recent_messages: first call (no session)
# =============================================================================


def test_first_call_returns_all_messages():
    """First call (no session) returns all messages unchanged."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
    ]
    result = provider._get_recent_messages(messages)
    assert len(result) == 3
    assert result[0].role == "system"
    assert result[1].role == "user"
    assert result[2].role == "assistant"


def test_no_system_prefix_returns_all():
    """Messages without a system prefix are returned unchanged."""
    provider = _make_provider()
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]
    result = provider._get_recent_messages(messages)
    assert len(result) == 2


# =============================================================================
# _get_recent_messages: resumed session (incremental delivery)
# =============================================================================


def test_resumed_session_sends_only_new_messages():
    """Resumed session sends only messages after the session-block anchor."""
    provider = _make_provider()
    provider._session.id = "test-session"

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        _assistant_with_session("Hi there"),  # anchor
        Message(role="user", content="Do something"),  # NEW
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),  # NEW
    ]
    result = provider._get_recent_messages(messages)

    # Should get: system + anchor assistant + new messages
    assert len(result) == 4  # system + assistant(anchor) + user + reminder
    assert result[0].role == "system"
    assert result[1].role == "assistant"  # anchor included for tool_use ID validation
    assert result[2].content == "Do something"
    assert "<system-reminder" in result[3].content


def test_resumed_session_with_tool_results():
    """Tool results after the session block are delivered."""
    provider = _make_provider()
    provider._session.id = "test-session"

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Fix the bug"),
        _assistant_with_session(),  # anchor - assistant with tool_use + session block
        # Tool results from the assistant's tool calls:
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="tc1", output='{"ok": true}')],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="tc2", output='{"ok": true}')],
        ),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    result = provider._get_recent_messages(messages)

    # system + assistant(anchor) + 2 tool results + reminder
    assert len(result) == 5
    assert result[0].role == "system"
    assert result[1].role == "assistant"  # anchor included for tool_use ID validation
    assert result[2].content[0].type == "tool_result"
    assert result[3].content[0].type == "tool_result"
    assert "<system-reminder" in result[4].content


def test_resumed_session_with_hook_outputs():
    """Hook outputs (like python_check) after session block are delivered.

    This is the scenario that broke the fingerprint approach: python_check
    outputs change between turns, causing fingerprint mismatch and full
    conversation resend. The session-block anchor is immune to this because
    it doesn't depend on message content.
    """
    provider = _make_provider()
    provider._session.id = "test-session"

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Fix the bug"),
        _assistant_with_session(),  # anchor
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="tc1", output='{"ok": true}')],
        ),
        # python_check hook output -- changes between turns
        Message(
            role="user",
            content="Python check found issues in app.py: formatting",
        ),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    result = provider._get_recent_messages(messages)

    # system + assistant(anchor) + tool_result + hook output + reminder
    assert len(result) == 5
    assert result[0].role == "system"
    assert result[1].role == "assistant"  # anchor included
    assert result[3].content == "Python check found issues in app.py: formatting"


def test_multiple_session_blocks_uses_most_recent():
    """When multiple session blocks exist, the most recent one is the anchor."""
    provider = _make_provider()
    provider._session.id = "test-session"

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        _assistant_with_session("First response", session_id="sess-1"),  # older
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="tc1", output="done")],
        ),
        _assistant_with_session("Second response", session_id="sess-2"),  # newer anchor
        Message(role="user", content="Continue"),  # NEW
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),  # NEW
    ]
    result = provider._get_recent_messages(messages)

    # system + anchor assistant + new messages after it
    assert len(result) == 4  # system + assistant(anchor) + user + reminder
    assert result[0].role == "system"
    assert result[1].role == "assistant"  # second session block = anchor
    assert result[2].content == "Continue"


# =============================================================================
# _get_recent_messages: edge cases
# =============================================================================


def test_nothing_new_after_session_block():
    """If the session block is the last message, only system messages returned."""
    provider = _make_provider()
    provider._session.id = "test-session"

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        _assistant_with_session("Done"),
    ]
    result = provider._get_recent_messages(messages)

    assert len(result) == 2  # system + assistant(anchor)
    assert result[0].role == "system"
    assert result[1].role == "assistant"


def test_no_session_block_in_resumed_session_sends_all():
    """If session ID is set but no session block found, send full conversation.

    This handles the edge case where the session was somehow established
    without a session block in the conversation history.
    """
    provider = _make_provider()
    provider._session.id = "test-session"

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),  # no session block
        Message(role="user", content="Continue"),
    ]
    result = provider._get_recent_messages(messages)

    # No anchor found -- send full conversation (first-call behavior)
    assert len(result) == 4


def test_session_block_with_mixed_content_types():
    """Session block detection works with various assistant content types."""
    provider = _make_provider()
    provider._session.id = "test-session"

    # Assistant message with thinking + tool_use + session block
    messages = [
        Message(role="system", content="System prompt"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "Let me analyze..."},
                ToolCallBlock(id="ed01", name="edit_file", input={"path": "a.py"}),
                ToolCallBlock(
                    id="pc01", name="python_check", input={"paths": ["a.py"]}
                ),
                _make_session_block(),
            ],
        ),
        # New messages after the anchor
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_call_id="ed01", output="ok"),
                ToolResultBlock(tool_call_id="pc01", output="ok"),
            ],
        ),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
        Message(
            role="user",
            content="Python check found issues in app.py: line 42",
        ),
    ]
    result = provider._get_recent_messages(messages)

    # system + assistant(anchor) + tool_results + reminder + hook output
    assert len(result) == 5
    assert result[0].role == "system"
    assert result[1].role == "assistant"  # anchor included
    assert result[2].content[0].type == "tool_result"


def test_python_check_change_between_turns_no_resend():
    """Regression test: python_check output changing between turns must NOT
    cause full conversation resend.

    This was the root cause of the 'Prompt is too long' bug. The fingerprint
    approach anchored on python_check output, which changed between turns,
    triggering a full-conversation fallback that exploded CLI context.

    The session-block anchor is immune because it anchors on structure,
    not content.
    """
    provider = _make_provider()
    provider._session.id = "test-session"

    # Turn N: assistant responded, python_check output present
    messages_turn_n = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Fix the bug"),
        _assistant_with_session("I'll fix it"),
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="ef01", output='{"ok": true}')],
        ),
        Message(
            role="user",
            content="Python check found issues: line 10 formatting",
        ),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env v1</system-reminder>',
        ),
    ]
    result_n = provider._get_recent_messages(messages_turn_n)
    # system + assistant(anchor) + 3 new messages
    assert len(result_n) == 5

    # Turn N+1: python_check output CHANGED (code was edited), new messages added
    messages_turn_n1 = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Fix the bug"),
        _assistant_with_session("I'll fix it"),  # old anchor
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="ef01", output='{"ok": true}')],
        ),
        # python_check output CHANGED -- this broke the fingerprint approach
        Message(
            role="user",
            content="Python check found DIFFERENT issues: line 20 types",
        ),
        # New assistant response with session block
        _assistant_with_session(
            "Applied fix", session_id="test-session-2"
        ),  # NEW anchor
        # New tool results
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="ef02", output='{"ok": true}')],
        ),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env v2</system-reminder>',
        ),
    ]
    result_n1 = provider._get_recent_messages(messages_turn_n1)

    # Session-block anchor finds the NEWER "Applied fix" assistant message.
    # Includes anchor + messages after it. NOT the full conversation.
    assert len(result_n1) == 4  # system + assistant(anchor) + tool_result + reminder
    assert result_n1[0].role == "system"
    assert result_n1[1].role == "assistant"  # anchor
    assert result_n1[2].content[0].type == "tool_result"
    assert result_n1[2].content[0].tool_call_id == "ef02"
