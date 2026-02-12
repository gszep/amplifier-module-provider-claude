"""Tests for message identity tracking and incremental CLI delivery.

Covers the fingerprint-based _get_recent_messages() that replaced the
fragile integer-index approach, and the _is_ephemeral_message() heuristic.
"""

from amplifier_core.message_models import Message, ToolCallBlock

from amplifier_module_provider_claude import ClaudeProvider


def _make_provider() -> ClaudeProvider:
    provider = ClaudeProvider()
    return provider


# =============================================================================
# _fingerprint_message
# =============================================================================


def test_fingerprint_stable_for_same_message():
    """Same message produces the same fingerprint across calls."""
    msg = Message(role="assistant", content="Hello world")
    fp1 = ClaudeProvider._fingerprint_message(msg)
    fp2 = ClaudeProvider._fingerprint_message(msg)
    assert fp1 == fp2


def test_fingerprint_differs_for_different_content():
    """Different content produces different fingerprints."""
    msg1 = Message(role="assistant", content="Hello")
    msg2 = Message(role="assistant", content="Goodbye")
    assert ClaudeProvider._fingerprint_message(
        msg1
    ) != ClaudeProvider._fingerprint_message(msg2)


def test_fingerprint_differs_for_different_roles():
    """Same content but different roles produces different fingerprints."""
    msg1 = Message(role="user", content="Hello")
    msg2 = Message(role="assistant", content="Hello")
    assert ClaudeProvider._fingerprint_message(
        msg1
    ) != ClaudeProvider._fingerprint_message(msg2)


def test_fingerprint_handles_list_content():
    """List content (tool call blocks) produces a stable fingerprint."""
    msg = Message(
        role="assistant",
        content=[ToolCallBlock(id="tc1", name="grep", input={"pattern": "test"})],
    )
    fp1 = ClaudeProvider._fingerprint_message(msg)
    fp2 = ClaudeProvider._fingerprint_message(msg)
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 16  # sha256 hex truncated to 16 chars


# =============================================================================
# _is_ephemeral_message
# =============================================================================


def test_ephemeral_detects_string_system_reminder():
    """String content with <system-reminder is detected as ephemeral."""
    msg = Message(
        role="user",
        content='<system-reminder source="hooks-status-context">env info</system-reminder>',
    )
    assert ClaudeProvider._is_ephemeral_message(msg) is True


def test_ephemeral_detects_list_content_system_reminder():
    """List content with <system-reminder in a text block is detected as ephemeral."""
    msg = Message(
        role="user",
        content=[
            {
                "type": "text",
                "text": '<system-reminder source="hooks-todo">todo list</system-reminder>',
            }
        ],
    )
    assert ClaudeProvider._is_ephemeral_message(msg) is True


def test_ephemeral_rejects_regular_user_message():
    """Regular user message is NOT ephemeral."""
    msg = Message(role="user", content="Please fix the bug in auth.py")
    assert ClaudeProvider._is_ephemeral_message(msg) is False


def test_ephemeral_rejects_assistant_message():
    """Assistant messages are never ephemeral."""
    msg = Message(role="assistant", content="I'll fix that for you")
    assert ClaudeProvider._is_ephemeral_message(msg) is False


def test_ephemeral_rejects_tool_message():
    """Tool result messages are never ephemeral."""
    msg = Message(role="tool", content='{"success": true}', tool_call_id="tc1")
    assert ClaudeProvider._is_ephemeral_message(msg) is False


def test_ephemeral_rejects_array_without_reminder():
    """List-content user message WITHOUT system-reminder is not ephemeral.

    This is the python_check hook output case that triggered the original bug.
    The fingerprint approach handles this through the safe fallback.
    """
    msg = Message(
        role="user",
        content=[
            {"type": "text", "text": "Python check found issues in app.py: formatting"},
        ],
    )
    assert ClaudeProvider._is_ephemeral_message(msg) is False


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


def test_first_call_stores_fingerprint():
    """First call sets the fingerprint on the last non-ephemeral message."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    provider._get_recent_messages(messages)

    # Fingerprint should be on the assistant message (last non-ephemeral)
    expected_fp = ClaudeProvider._fingerprint_message(messages[2])
    assert provider._session.last_delivered_fingerprint == expected_fp


# =============================================================================
# _get_recent_messages: resumed session (incremental delivery)
# =============================================================================


def test_resumed_session_sends_only_new_messages():
    """Resumed session sends only messages after the fingerprinted anchor."""
    provider = _make_provider()

    # Simulate first call
    first_messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    provider._get_recent_messages(first_messages)

    # Set session ID (simulating CLI response setting it)
    provider._session.id = "test-session-123"

    # Simulate second call: orchestrator removed old ephemeral, added new messages
    second_messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),  # old (already delivered)
        Message(role="assistant", content="Hi there"),  # old (fingerprint anchor)
        Message(role="tool", content='{"result": "ok"}', tool_call_id="tc1"),  # NEW
        Message(
            role="assistant",
            content=[ToolCallBlock(id="tc2", name="edit_file", input={"path": "a.py"})],
        ),  # NEW
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env v2</system-reminder>',
        ),  # NEW ephemeral
    ]
    result = provider._get_recent_messages(second_messages)

    # Should get: system + only new messages (after the fingerprint anchor)
    assert len(result) == 4  # system + tool + assistant + reminder
    assert result[0].role == "system"
    assert result[1].role == "tool"
    assert result[2].role == "assistant"
    assert result[3].role == "user"


# =============================================================================
# _get_recent_messages: the original bug scenario
# =============================================================================


def test_ephemeral_array_content_does_not_inflate_index():
    """Regression test for the d285ed4d session bug.

    When iteration N has an array-content ephemeral message (like python_check
    output) that the old _count_trailing_reminders() couldn't detect, the
    integer index was inflated, causing iteration N+1 to skip real messages.

    The fingerprint approach handles this: the python_check message is not
    detected as ephemeral, so we fingerprint it. On the next call it's gone
    (ephemeral), so we fall back to sending full conversation (safe).
    """
    provider = _make_provider()

    # Iteration 4: has hook + python_check (array content, no system-reminder)
    iter4_messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Fix the bug"),
        Message(role="assistant", content="I'll fix it"),
        Message(role="tool", content='{"ok": true}', tool_call_id="ef01"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
        # python_check output - array content, NO system-reminder marker
        Message(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": "Python check found issues in app.py: formatting",
                }
            ],
        ),
    ]
    provider._get_recent_messages(iter4_messages)

    # The python_check message is NOT detected as ephemeral, so it becomes the anchor
    python_check_msg = iter4_messages[5]
    expected_fp = ClaudeProvider._fingerprint_message(python_check_msg)
    assert provider._session.last_delivered_fingerprint == expected_fp

    # Set session ID
    provider._session.id = "test-session"

    # Iteration 5: orchestrator removed both ephemeral messages, added new ones
    iter5_messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Fix the bug"),
        Message(role="assistant", content="I'll fix it"),
        Message(role="tool", content='{"ok": true}', tool_call_id="ef01"),
        # NEW: assistant response with tool calls (from iteration 4)
        Message(
            role="assistant",
            content=[
                ToolCallBlock(
                    id="rf04cla", name="read_file", input={"path": "claude.py"}
                ),
            ],
        ),
        # NEW: tool results
        Message(role="tool", content='{"file": "contents"}', tool_call_id="rf04cla"),
        # NEW: ephemeral reminder
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env v2</system-reminder>',
        ),
    ]
    result = provider._get_recent_messages(iter5_messages)

    # The python_check fingerprint is NOT in this list (it was ephemeral and removed).
    # So we get the SAFE FALLBACK: full conversation.
    # This means all non-system messages are included.
    assert len(result) >= 4  # system + at minimum the new messages
    # Critically, the new assistant and tool messages MUST be included
    roles = [m.role for m in result]
    assert "tool" in roles  # tool results are NOT dropped
    assert "assistant" in roles  # assistant response is NOT dropped


def test_normal_cycle_does_not_drop_messages():
    """In a normal cycle (no rogue ephemeral), messages are correctly subset."""
    provider = _make_provider()

    # First call
    messages_1 = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    provider._get_recent_messages(messages_1)
    provider._session.id = "sess-1"

    # Second call - new tool results added after the assistant
    messages_2 = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),  # fingerprint anchor
        Message(role="tool", content='{"data": 1}', tool_call_id="t1"),
        Message(role="tool", content='{"data": 2}', tool_call_id="t2"),
        Message(role="assistant", content="Done"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    result = provider._get_recent_messages(messages_2)

    # Should only include system + new messages after the anchor
    assert result[0].role == "system"
    assert result[0].content == "System prompt"
    # New messages: tool, tool, assistant, reminder
    new_msgs = result[1:]
    assert len(new_msgs) == 4
    assert new_msgs[0].role == "tool"
    assert new_msgs[1].role == "tool"
    assert new_msgs[2].role == "assistant"
    assert new_msgs[3].role == "user"


# =============================================================================
# _get_recent_messages: compaction fallback
# =============================================================================


def test_compaction_triggers_safe_fallback():
    """When the fingerprinted message is removed by compaction, send everything."""
    provider = _make_provider()

    # First call establishes fingerprint on the assistant message
    messages_1 = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="I will do the thing"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    provider._get_recent_messages(messages_1)
    provider._session.id = "sess-1"

    # Second call - compaction removed the original user + assistant
    messages_2 = [
        Message(role="system", content="System prompt"),
        # compaction notice replaces old messages
        Message(role="user", content="[Earlier conversation was compacted]"),
        Message(role="assistant", content="Continuing from compacted context"),
        Message(role="tool", content='{"ok": true}', tool_call_id="t1"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    result = provider._get_recent_messages(messages_2)

    # Fingerprint "I will do the thing" is gone. Fallback: send ALL conversation.
    assert len(result) == 5  # system + all 4 conversation messages
    assert result[0].role == "system"
    # All conversation messages are present (nothing dropped)
    assert result[1].content == "[Earlier conversation was compacted]"
    assert result[3].role == "tool"


# =============================================================================
# _get_recent_messages: no system prefix
# =============================================================================


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
# _get_recent_messages: only ephemeral conversation
# =============================================================================


def test_all_ephemeral_does_not_update_fingerprint():
    """If all conversation messages are ephemeral, fingerprint stays unchanged."""
    provider = _make_provider()
    provider._session.id = "sess-1"
    provider._session.last_delivered_fingerprint = "old_fingerprint"

    messages = [
        Message(role="system", content="System prompt"),
        Message(
            role="user",
            content='<system-reminder source="hooks-status-context">env</system-reminder>',
        ),
    ]
    provider._get_recent_messages(messages)

    # Fingerprint should remain unchanged (no non-ephemeral to anchor on)
    assert provider._session.last_delivered_fingerprint == "old_fingerprint"
