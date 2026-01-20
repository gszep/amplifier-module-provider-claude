"""Claude provider module for Amplifier.

Full Control implementation using Claude Code CLI.
Amplifier's orchestrator handles tool execution - Claude only decides which tools to call.
"""

__all__ = [
    "mount",
    "ClaudeProvider",
    "SessionManager",
    "SessionMetadata",
    "SessionState",
]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import shutil
import time
import uuid
from typing import Any

from amplifier_core import (  # type: ignore
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)
from amplifier_core.events import (  # type: ignore
    CONTENT_BLOCK_DELTA,
    CONTENT_BLOCK_END,
    CONTENT_BLOCK_START,
)
from amplifier_core.message_models import (  # type: ignore
    ChatRequest,
    ChatResponse,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)

from .sessions import SessionManager, SessionMetadata, SessionState

logger = logging.getLogger(__name__)


class ClaudeChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

METADATA_SESSION_ID = "claude:session_id"
METADATA_COST_USD = "claude:cost_usd"
METADATA_DURATION_MS = "claude:duration_ms"

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_TOKENS = 64000
DEFAULT_MAX_THINKING_TOKENS = 10240  # 10K thinking budget - good middle ground
MIN_THINKING_TOKENS = 1024  # API minimum

# Model specifications
MODELS = {
    "sonnet": {
        "id": "sonnet",
        "display_name": "Claude Sonnet",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming", "thinking"],
    },
    "opus": {
        "id": "opus",
        "display_name": "Claude Opus",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": ["tools", "streaming", "thinking"],
    },
    "haiku": {
        "id": "haiku",
        "display_name": "Claude Haiku",
        "context_window": 200000,
        "max_output_tokens": 64000,
        "capabilities": [
            "tools",
            "streaming",
            "thinking",
        ],  # Haiku 4.5 supports thinking
    },
}


# -----------------------------------------------------------------------------
# Mount function
# -----------------------------------------------------------------------------


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
) -> None:
    """Mount the Claude provider using Claude Code CLI.

    Args:
        coordinator: The module coordinator to mount to.
        config: Optional configuration dictionary.

    Returns:
        None (no cleanup needed for CLI-based provider).
    """
    config = config or {}

    # Check if CLI is available
    cli_path = shutil.which("claude")
    if not cli_path:
        logger.warning(
            "Claude Code CLI not found. Install with: "
            "curl -fsSL https://claude.ai/install.sh | bash"
        )
        return None

    provider = ClaudeProvider(config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="claude")
    logger.info("Mounted ClaudeProvider (Claude Code CLI - Full Control mode)")
    return None


# -----------------------------------------------------------------------------
# Provider Implementation
# -----------------------------------------------------------------------------


class ClaudeProvider:
    """Claude Code CLI integration for Amplifier.

    Full Control mode: Amplifier's orchestrator handles all tool execution.
    Claude only decides which tools to call - execution is handled externally.

    This follows the same pattern as the OpenAI provider:
    1. Provider receives request with messages and tools
    2. Provider returns response with tool_calls (if any)
    3. Orchestrator executes tools and adds results to messages
    4. Provider receives next request with tool results
    5. Repeat until no more tool_calls
    """

    name = "claude"
    api_label = "Claude Code"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """Initialize the Claude provider.

        Args:
            config: Provider configuration.
            coordinator: The module coordinator for event emission.
        """
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration
        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.debug = self.config.get("debug", False)
        self.max_thinking_tokens = max(
            MIN_THINKING_TOKENS,
            self.config.get("max_thinking_tokens", DEFAULT_MAX_THINKING_TOKENS),
        )

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations.
        self._repaired_tool_ids: set[str] = set()

        # Session persistence for prompt caching across restarts
        # Uses disk-persisted sessions following amplifier-claude pattern
        self._session_manager = SessionManager(
            session_dir=self.config.get("session_dir"),
        )

        # Get Amplifier session ID from coordinator for session mapping
        amplifier_session_id = self._get_amplifier_session_id()

        # Load existing session or create new one
        # This restores the Claude CLI session ID for cache resumption
        self._session_state = self._session_manager.get_or_create_session(
            session_id=amplifier_session_id,
            name=self.config.get("session_name", "amplifier-claude"),
        )

        # Track valid tool names for the session (updated each turn from request.tools).
        # Used to filter out tool calls that reference non-existent tools.
        self._valid_tool_names: set[str] = set()

        # Track filtered tool calls for the current turn (cleared each turn).
        # These are tool calls that were rejected because the tool name wasn't valid.
        # Fed back to Claude so it knows those tools aren't available.
        self._filtered_tool_calls: list[dict[str, Any]] = []

    def _get_amplifier_session_id(self) -> str | None:
        """Get the Amplifier session ID from the coordinator.

        Returns:
            Session ID string if available, None otherwise.
        """
        if not self.coordinator:
            return None

        # Try to get session ID from coordinator's session attribute
        if hasattr(self.coordinator, "session"):
            session = getattr(self.coordinator, "session", None)
            if session and hasattr(session, "id"):
                return str(session.id)

        # Try to get from coordinator's config
        if hasattr(self.coordinator, "config"):
            config = getattr(self.coordinator, "config", {})
            if isinstance(config, dict) and "session_id" in config:
                return str(config["session_id"])

        return None

    def _get_claude_session_id(self) -> str | None:
        """Get the Claude CLI session ID for resumption.

        Returns:
            Claude session ID if available for cache resumption.
        """
        return self._session_state.metadata.claude_session_id

    def _save_session(self) -> None:
        """Save the current session state to disk."""
        self._session_manager.save_session(self._session_state)
        if self.debug:
            efficiency = self._session_state.get_cache_efficiency()
            logger.debug(
                f"[PROVIDER] Session saved: {self._session_state.metadata.session_id}, "
                f"cache efficiency: {efficiency:.1%}"
            )

    def get_info(self) -> ProviderInfo:
        """Return provider information.

        Returns:
            ProviderInfo with capabilities and configuration fields.
        """
        return ProviderInfo(
            id="claude",
            display_name="Claude Code",
            credential_env_vars=[],  # No API key needed - uses Claude Code auth
            capabilities=["streaming", "tools", "thinking"],
            defaults={
                "model": self.default_model,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "timeout": self.timeout,
            },
            config_fields=[],  # No interactive configuration needed
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available models.

        Returns:
            List of ModelInfo objects for available Claude models.
        """
        return [
            ModelInfo(
                id=spec["id"],
                display_name=spec["display_name"],
                context_window=spec["context_window"],
                max_output_tokens=spec["max_output_tokens"],
                capabilities=spec["capabilities"],
                defaults={"temperature": 1.0}
                if "thinking" in spec["capabilities"]
                else {},
            )
            for spec in MODELS.values()
        ]

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from response.

        Tool calls are extracted in complete() and placed in response.tool_calls.
        Filters out tool calls with empty/missing arguments to handle
        Claude API quirk where empty tool_use blocks are sometimes generated.

        Args:
            response: The ChatResponse to extract tool calls from.

        Returns:
            List of valid tool calls (with non-empty arguments).
        """
        if not response.tool_calls:
            return []

        # Filter out tool calls with empty arguments (Claude API quirk)
        # Claude sometimes generates tool_use blocks with empty input {}
        valid_calls = []
        for tc in response.tool_calls:
            # Skip tool calls with no arguments or empty dict
            if not tc.arguments:
                logger.debug(f"Filtering out tool '{tc.name}' with empty arguments")
                continue
            valid_calls.append(tc)

        if len(valid_calls) < len(response.tool_calls):
            logger.info(
                f"Filtered {len(response.tool_calls) - len(valid_calls)} tool calls with empty arguments"
            )

        return valid_calls

    # -------------------------------------------------------------------------
    # Tool result validation and repair
    # -------------------------------------------------------------------------

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs WITH their
        source message index so they can be inserted in the correct position.

        Excludes tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops.

        Returns:
            List of (msg_index, call_id, tool_name, tool_arguments) tuples for unpaired calls.
            msg_index is the index of the assistant message containing the tool_use block.
        """
        tool_calls = {}  # {call_id: (msg_index, name, args)}
        tool_results = set()  # {call_id}

        for idx, msg in enumerate(messages):
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> Message:
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    # -------------------------------------------------------------------------
    # Main completion method
    # -------------------------------------------------------------------------

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Execute a completion request via Claude Code CLI.

        Full Control mode: Returns tool_calls for Amplifier to execute.

        Args:
            request: The chat request with messages and tools.
            **kwargs: Additional arguments (model override, etc.).

        Returns:
            ChatResponse with content and optional tool_calls.

        Raises:
            RuntimeError: If CLI is not found or fails.
        """
        # Update valid tool names for this session (used for filtering invalid tool calls)
        self._valid_tool_names = set()
        if request.tools:
            for tool in request.tools:
                if hasattr(tool, "name"):
                    self._valid_tool_names.add(tool.name)
                elif isinstance(tool, dict) and "name" in tool:
                    self._valid_tool_names.add(tool["name"])

        # Save filtered tool calls from previous turn (to feed back to Claude)
        # then clear for this turn
        previous_filtered_calls = self._filtered_tool_calls.copy()
        self._filtered_tool_calls = []

        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Claude: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Group missing results by source assistant message index
            # We need to insert synthetic results IMMEDIATELY after each assistant message
            # that contains tool_use blocks (not at the end of the list)
            from collections import defaultdict

            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            # Insert synthetic results in reverse order of message index
            # (so earlier insertions don't shift later indices)
            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    # Track this ID so we don't detect it as missing again in future iterations
                    self._repaired_tool_ids.add(call_id)

                # Insert all synthetic results immediately after the assistant message
                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        # Inject feedback about filtered tool calls from previous turn
        for tool_call in previous_filtered_calls:
            request.messages.append(
                Message(
                    role="tool",
                    content=(
                        f"[SYSTEM NOTICE: Tool call rejected]\n\n"
                        f"Tool: {tool_call['name']}\n"
                        f"is not available in the current context and cannot be called.\n"
                        f"Please acknowledge this error and offer to retry the operation."
                    ),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        if previous_filtered_calls:
            logger.info(
                f"[PROVIDER] Injected feedback about {len(previous_filtered_calls)} "
                f"filtered tool calls from previous turn."
            )

        start_time = time.time()

        # Find CLI
        cli_path = shutil.which("claude")
        if not cli_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: "
                "curl -fsSL https://claude.ai/install.sh | bash"
            )

        # Get model
        model = (
            kwargs.get("model") or getattr(request, "model", None) or self.default_model
        )

        # Check for existing session to resume
        # Priority: 1) Explicit request metadata override, 2) Persisted session state
        request_metadata = getattr(request, "metadata", None) or {}
        existing_session_id = (
            request_metadata.get(METADATA_SESSION_ID) or self._get_claude_session_id()
        )
        resuming = existing_session_id is not None

        # Convert messages to CLI format
        # When resuming, Claude CLI has cached history - only send current turn
        system_prompt, user_prompt = self._convert_messages(
            request.messages, request.tools, resuming=resuming
        )

        # Build command (without system prompt - passed via stdin to avoid ARG_MAX)
        cmd = self._build_command(
            cli_path=cli_path,
            model=model,
            session_id=existing_session_id,
        )

        # Combine system prompt and user prompt for stdin
        # System instructions go first, then the user's actual request
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        if self.debug:
            logger.debug(f"[PROVIDER] Command: {' '.join(cmd[:10])}...")
            logger.debug(f"[PROVIDER] System prompt length: {len(system_prompt)}")
            logger.debug(f"[PROVIDER] User prompt: {user_prompt[:200]}...")

        # Emit request event
        await self._emit_event(
            "llm:request",
            {
                "provider": self.name,
                "model": model,
                "messages_count": len(request.messages),
                "tools_count": len(request.tools) if request.tools else 0,
                "resume_session": existing_session_id is not None,
            },
        )

        # Execute CLI and parse response (prompt passed via stdin)
        response_data = await self._execute_cli(cmd, full_prompt)

        # Store session ID in persistent session state for cache resumption
        response_session_id = response_data.get("metadata", {}).get(METADATA_SESSION_ID)
        if response_session_id:
            self._session_state.set_claude_session_id(response_session_id)
            logger.debug(
                f"[PROVIDER] Stored session ID for resumption: {response_session_id}"
            )

        duration = time.time() - start_time

        # Build ChatResponse
        chat_response = self._build_response(response_data, duration)

        # Update session usage statistics and save to disk
        usage_data = response_data.get("usage", {})
        self._session_state.update_usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read=usage_data.get("cache_read_input_tokens", 0),
            cache_creation=usage_data.get("cache_creation_input_tokens", 0),
            cost_usd=response_data.get("metadata", {}).get(METADATA_COST_USD, 0.0)
            or 0.0,
            duration_ms=int(duration * 1000),
        )
        self._save_session()

        # Emit response event
        await self._emit_event(
            "llm:response",
            {
                "provider": self.name,
                "model": model,
                "status": "ok",
                "duration_ms": int(duration * 1000),
                "usage": {
                    "input": chat_response.usage.input_tokens
                    if chat_response.usage
                    else 0,
                    "output": chat_response.usage.output_tokens
                    if chat_response.usage
                    else 0,
                },
                "has_tool_calls": bool(chat_response.tool_calls),
                "tool_calls_count": len(chat_response.tool_calls)
                if chat_response.tool_calls
                else 0,
            },
        )

        return chat_response

    # -------------------------------------------------------------------------
    # Message conversion
    # -------------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[Message],
        tools: list[Any] | None,
        resuming: bool = False,
    ) -> tuple[str, str]:
        """Convert Amplifier messages to Claude CLI format.

        Args:
            messages: List of Amplifier Message objects.
            tools: List of tool specifications.
            resuming: If True, only include current turn (Claude CLI has cached history).

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_parts = []
        conversation_parts = []
        tool_schema = ""

        # When resuming, only process current turn messages
        # Claude CLI caches conversation history, so we only need new content
        if resuming:
            messages = self._get_current_turn_messages(messages)
            # Skip system prompt and tool definitions - already cached by Claude CLI
        else:
            # Build tool definitions for system prompt (first call only)
            # Stored separately to append AFTER bundle system messages
            if tools:
                tool_definitions = self._convert_tools(tools)
                tool_schema = self._build_tool_schema(tool_definitions)

        # Process messages
        for msg in messages:
            role = msg.role
            content = self._extract_content(msg)

            if role == "system":
                # Skip system messages when resuming - already cached by Claude CLI
                if not resuming:
                    system_parts.append(f"<system-reminder>{content}</system-reminder>")

            elif role == "user":
                # Check if this is a hook-injected system reminder (not actual user content)
                # Hook content should appear outside <user> tags for cleaner prompt structure
                if content.strip().startswith("<system-reminder"):
                    conversation_parts.append(content)
                else:
                    conversation_parts.append(f"<user>{content}</user>")

            elif role == "assistant":
                # Check for tool calls in assistant message
                assistant_content = self._format_assistant_message(msg)
                conversation_parts.append(f"<assistant>{assistant_content}</assistant>")

            elif role == "tool":
                # Tool result - format for Claude
                tool_result = self._format_tool_result(msg)
                conversation_parts.append(f"{tool_result}")

            elif role == "developer":
                # Developer messages contain context files (like @mentions)
                # Wrap in <context_file> tags following Anthropic provider pattern
                wrapped = f"<context_file>\n{content}\n</context_file>"
                conversation_parts.append(f"{wrapped}")

        # Build final prompts
        # Order: bundle system messages first (persona/behavior), then tool schema (transport)
        if tool_schema:
            system_parts.append(tool_schema)
        system_prompt = "\n\n".join(system_parts) if system_parts else ""

        # The user prompt is the conversation history
        # For multi-turn, we include the full conversation
        user_prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

        # If there's only one user message and no conversation history, simplify
        if len(messages) == 1 and messages[0].role == "user":
            user_prompt = self._extract_content(messages[0])

        return system_prompt, user_prompt

    def _get_current_turn_messages(self, messages: list[Message]) -> list[Message]:
        """Get only messages from the current turn (after last assistant response).

        When resuming a Claude CLI session, the CLI has the conversation history
        cached. We only need to send:
        - Tool results from the current turn (after the last assistant message)
        - Any new user message

        Args:
            messages: Full list of conversation messages.

        Returns:
            Messages from the current turn only.
        """
        # Find the last assistant message index
        last_assistant_idx = -1
        for i, msg in enumerate(messages):
            if msg.role == "assistant":
                last_assistant_idx = i

        if last_assistant_idx == -1:
            # No assistant message yet - this is first turn, return all
            return messages

        # Return everything after the last assistant message
        # This includes tool results and any new user/developer messages
        current_turn = messages[last_assistant_idx + 1 :]

        return current_turn

    def _extract_content(self, msg: Message) -> str:
        """Extract text content from a message.

        Args:
            msg: The message to extract content from.

        Returns:
            String content of the message.
        """
        content = msg.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
            return "\n".join(text_parts)

        return str(content) if content else ""

    def _format_assistant_message(self, msg: Message) -> str:
        """Format an assistant message, including any tool calls.

        Args:
            msg: The assistant message.

        Returns:
            Formatted string representation.
        """
        parts = []

        content = msg.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        parts.append(block.text)
                    elif block.type in ("tool_use", "tool_call"):
                        # Format tool call for conversation history
                        tool_call_str = json.dumps(
                            {
                                "tool": block.name,
                                "id": block.id,
                                "input": getattr(block, "input", {}),
                            }
                        )
                        parts.append(f"<tool_use>{tool_call_str}</tool_use>")
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") in ("tool_use", "tool_call"):
                        tool_call_str = json.dumps(
                            {
                                "tool": block.get("name"),
                                "id": block.get("id"),
                                "input": block.get("input", {}),
                            }
                        )
                        parts.append(f"<tool_use>{tool_call_str}</tool_use>")

        return "\n".join(parts)

    def _format_tool_result(self, msg: Message) -> str:
        """Format a tool result message.

        Args:
            msg: The tool result message.

        Returns:
            Formatted tool result string.
        """
        tool_call_id = getattr(msg, "tool_call_id", None)
        tool_name = getattr(msg, "name", "unknown")
        content = self._extract_content(msg)
        is_error = getattr(msg, "is_error", False)

        result = {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "result": content,
            "is_error": is_error,
        }

        return f"<tool_result>{json.dumps(result)}</tool_result>"

    # -------------------------------------------------------------------------
    # Tool conversion
    # -------------------------------------------------------------------------

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Amplifier tool specs to Claude format.

        Args:
            tools: List of ToolSpec objects.

        Returns:
            List of tool definitions in Claude format.
        """
        tool_definitions = []

        for tool in tools:
            if hasattr(tool, "name"):
                # ToolSpec object - use "parameters" (ToolSpec field name)
                tool_def = {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "input_schema": getattr(tool, "parameters", {}),
                }
            elif isinstance(tool, dict):
                # Dictionary format - prefer "parameters" (ToolSpec convention)
                tool_def = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get(
                        "parameters", tool.get("input_schema", {})
                    ),
                }
            else:
                continue

            tool_definitions.append(tool_def)

        return tool_definitions

    def _build_tool_schema(self, tools: list[dict[str, Any]]) -> str:
        """Build pure tool schema for system prompt.

        This provides only the structural information needed for tool calls.
        Behavioral instructions should come from Amplifier's bundle system prompts.

        Args:
            tools: List of tool definitions.

        Returns:
            Tool schema string with transport preamble.
        """
        if not tools:
            return ""

        tools_json = json.dumps(tools, indent=2)
        tool_use_example = json.dumps(
            {
                "tool": "tool_name",
                "id": "unique_id",
                "input": {"param1": "value1"},
            }
        )

        return f"""<system-reminder source="tools-context">
Available tools:
<tools>
{tools_json}
</tools>

To call a tool, use this format:
<tool_use>
{tool_use_example}
</tool_use>

Generate a unique ID for each call (e.g., "call_1", "call_2").
Tool results will be provided in <tool_result> blocks.
</system-reminder>"""

    # -------------------------------------------------------------------------
    # CLI execution
    # -------------------------------------------------------------------------

    def _build_command(
        self,
        cli_path: str,
        model: str,
        session_id: str | None,
    ) -> list[str]:
        """Build the CLI command.

        Note: System prompt is passed via stdin to avoid ARG_MAX limits.
        The --system-prompt arg overrides Claude's default persona with
        an Amplifier-specific role description.

        Args:
            cli_path: Path to the claude CLI.
            model: Model name.
            session_id: Optional session ID for resumption.

        Returns:
            List of command arguments.
        """
        cmd = [
            cli_path,
            "-p",  # Print mode (non-interactive)
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--tools",
            "",  # Disable ALL built-in tools - we provide our own
        ]

        # Add session resumption OR empty system prompt (mutually exclusive)
        # When resuming, Claude CLI has cached context including original system prompt
        if session_id:
            cmd.extend(["--resume", session_id])
            logger.info(f"[PROVIDER] Resuming Claude session: {session_id}")
        else:
            # First call - override Claude's default persona with empty prompt
            # Amplifier's bundle system messages define the persona, not the provider
            cmd.extend(["--system-prompt", ""])

        # Enable extended thinking for models that support it
        if self.max_thinking_tokens > 0:
            cmd.extend(["--max-thinking-tokens", str(self.max_thinking_tokens)])

        return cmd

    async def _execute_cli(self, cmd: list[str], prompt: str) -> dict[str, Any]:
        """Execute the CLI command and parse output.

        Args:
            cmd: The command to execute.
            prompt: The prompt to send via stdin (avoids ARG_MAX limits).

        Returns:
            Dictionary with parsed response data.
        """
        logger.debug("[PROVIDER] Executing Claude CLI")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Write prompt to stdin and close it
        assert proc.stdin is not None
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()

        response_text = ""
        usage_data: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        thinking_text = ""
        thinking_signature = ""
        block_index = 0

        assert proc.stdout is not None

        # Read in chunks to handle large JSON lines (thinking blocks can exceed 64KB)
        # The default asyncio StreamReader limit is 64KB which is too small
        buffer = b""
        while True:
            chunk = await proc.stdout.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            buffer += chunk

            # Process complete lines from buffer
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    event_data = json.loads(line_str)
                except json.JSONDecodeError:
                    if self.debug:
                        logger.warning(f"[PROVIDER] Failed to parse: {line_str[:100]}")
                    continue

                event_type = event_data.get("type")

                # Handle assistant messages - extract content blocks directly
                if event_type == "assistant":
                    message = event_data.get("message", {})
                    content_blocks = message.get("content", [])

                    for block in content_blocks:
                        block_type = block.get("type")

                        if block_type == "thinking":
                            # Extract thinking content and signature
                            thinking_text = block.get("thinking", "")
                            thinking_signature = block.get("signature", "")
                            logger.debug(
                                f"[PROVIDER] Thinking block: {len(thinking_text)} chars"
                            )

                            # Emit streaming events for thinking block
                            await self._emit_event(
                                CONTENT_BLOCK_START,
                                {
                                    "index": block_index,
                                    "content_block": ThinkingBlock(
                                        thinking="",
                                        signature=thinking_signature,
                                    ),
                                },
                            )
                            await self._emit_event(
                                CONTENT_BLOCK_DELTA,
                                {
                                    "index": block_index,
                                    "delta": ThinkingBlock(
                                        thinking=thinking_text,
                                        signature=thinking_signature,
                                    ),
                                },
                            )
                            await self._emit_event(
                                CONTENT_BLOCK_END,
                                {
                                    "index": block_index,
                                    "content_block": ThinkingBlock(
                                        thinking=thinking_text,
                                        signature=thinking_signature,
                                    ),
                                },
                            )
                            block_index += 1

                        elif block_type == "text":
                            # Accumulate text content (multiple text blocks can appear)
                            text_content = block.get("text", "")
                            if response_text and text_content:
                                response_text += "\n" + text_content
                            else:
                                response_text += text_content

                            # Emit streaming events for text block
                            await self._emit_event(
                                CONTENT_BLOCK_START,
                                {
                                    "index": block_index,
                                    "content_block": {"type": "text"},
                                },
                            )
                            await self._emit_event(
                                CONTENT_BLOCK_DELTA,
                                {
                                    "index": block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": text_content,
                                    },
                                },
                            )
                            await self._emit_event(
                                CONTENT_BLOCK_END,
                                {"index": block_index},
                            )
                            block_index += 1

                # Handle final result
                elif event_type == "result":
                    # Use result text as fallback
                    if not response_text:
                        response_text = event_data.get("result", "")

                    usage_data = event_data.get("usage", {})
                    session_id = event_data.get("session_id")
                    metadata = {
                        METADATA_SESSION_ID: session_id,
                        METADATA_DURATION_MS: event_data.get("duration_ms"),
                        METADATA_COST_USD: event_data.get("total_cost_usd"),
                        "num_turns": event_data.get("num_turns"),
                    }

        # Wait for process to complete
        await proc.wait()

        if proc.returncode != 0:
            stderr_data = await proc.stderr.read() if proc.stderr else b""
            error_msg = stderr_data.decode("utf-8").strip()

            if error_msg != "":
                logger.error(f"[PROVIDER] CLI failed: {error_msg}")
                raise RuntimeError(
                    f"Claude Code CLI failed (exit {proc.returncode}): {error_msg}"
                )
            else:
                raise RuntimeError(
                    f"Claude Code CLI failed (exit {proc.returncode}): Subscription limits may have been exceeded. Visit https://claude.ai/settings/usage."
                )

        # Parse tool calls from response text
        tool_calls = self._extract_tool_calls(response_text)

        return {
            "text": response_text,
            "tool_calls": tool_calls,
            "usage": usage_data,
            "metadata": metadata,
            "thinking": thinking_text,
            "thinking_signature": thinking_signature,
        }

    # -------------------------------------------------------------------------
    # Response building
    # -------------------------------------------------------------------------

    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from response text.

        Looks for <tool_use>...</tool_use> blocks in the response, filtering out:
        - Tool calls inside markdown code blocks (documentation examples)
        - Tool calls with names not in self._valid_tool_names

        Filtered tool calls (invalid names) are stored in self._filtered_tool_calls
        so they can be fed back to Claude in the next turn.

        Args:
            text: The response text to parse.

        Returns:
            List of valid tool call dictionaries.
        """
        tool_calls = []

        # Find all tool_use blocks
        import re

        pattern = r"<tool_use>\s*(.*?)\s*</tool_use>"
        matches = re.findall(
            pattern,
            re.sub(
                r"```[\s\S]*?```",  # remove all markdown code blocks to avoid parsing documentation examples
                "",
                text,
                flags=re.DOTALL,
            ),
            re.DOTALL,
        )

        for match in matches:
            # Skip if content doesn't look like JSON (e.g., documentation text about tool_use)
            stripped = match.strip()
            if not stripped.startswith("{"):
                logger.debug(
                    f"[PROVIDER] Skipping non-JSON content in tool_use block: "
                    f"{stripped[:50]}..."
                )
                continue

            try:
                tool_data = json.loads(match)
                tool_call = {
                    "id": tool_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": tool_data.get("tool", tool_data.get("name", "")),
                    "arguments": tool_data.get("input", tool_data.get("arguments", {})),
                }

                # Validate tool name against allowed list
                if (
                    self._valid_tool_names
                    and tool_call["name"] not in self._valid_tool_names
                ):
                    logger.debug(
                        f"[PROVIDER] Filtering tool call with invalid name: {tool_call['name']!r}. "
                        f"Valid tools: {sorted(self._valid_tool_names)[:10]}..."
                    )
                    # Store filtered call for feedback to Claude
                    self._filtered_tool_calls.append(tool_call)
                    continue

                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[PROVIDER] Failed to parse tool call JSON: {e}. "
                    f"Content: {match[:200]}"
                )
                continue

        return tool_calls

    def _clean_response_text(self, text: str) -> str:
        """Remove tool_use blocks from response text.

        Args:
            text: The response text.

        Returns:
            Cleaned text without tool_use blocks.
        """
        import re

        # Remove tool_use blocks
        cleaned = re.sub(r"<tool_use>.*?</tool_use>", "", text, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _build_response(
        self, response_data: dict[str, Any], duration: float
    ) -> ClaudeChatResponse:
        """Build a ClaudeChatResponse from parsed response data.

        Args:
            response_data: Parsed response from CLI.
            duration: Request duration in seconds.

        Returns:
            ClaudeChatResponse object with content_blocks for UI compatibility.
        """
        raw_text = response_data.get("text", "")
        tool_call_dicts = response_data.get("tool_calls", [])
        usage_data = response_data.get("usage", {})
        metadata = response_data.get("metadata", {})

        # Clean response text (remove tool_use blocks)
        clean_text = self._clean_response_text(raw_text)

        # Build content blocks - ORDER: thinking -> text -> tool_use
        content_blocks: list[Any] = []
        # Build event_blocks for streaming UI compatibility (like Anthropic provider)
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []

        # Add thinking block first if present (with signature preservation)
        thinking_content = response_data.get("thinking", "")
        thinking_sig = response_data.get("thinking_signature", "")
        if thinking_content:
            thinking_block = ThinkingBlock(
                thinking=thinking_content,
                signature=thinking_sig,
                visibility="internal",
            )
            logger.debug(f"[PROVIDER] ThinkingBlock: {len(thinking_content)} chars")
            content_blocks.append(thinking_block)
            # Add ThinkingContent for UI streaming compatibility
            event_blocks.append(ThinkingContent(text=thinking_content))

        # Then text block
        if clean_text:
            content_blocks.append(TextBlock(text=clean_text))
            event_blocks.append(TextContent(text=clean_text))

        # Add tool call blocks to content
        for tc in tool_call_dicts:
            content_blocks.append(
                ToolCallBlock(
                    id=tc["id"],
                    name=tc["name"],
                    input=tc["arguments"],
                )
            )
            event_blocks.append(
                ToolCallContent(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
            )

        # Build tool_calls list for orchestrator
        tool_calls: list[ToolCall] | None = None
        if tool_call_dicts:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                )
                for tc in tool_call_dicts
            ]

        # Build usage
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cache_read = usage_data.get("cache_read_input_tokens", 0)
        cache_creation = usage_data.get("cache_creation_input_tokens", 0)
        total_input = input_tokens + cache_read + cache_creation

        # Build usage dict with cache metrics if available
        # (matches Anthropic provider pattern - only include when non-zero)
        usage_kwargs: dict[str, Any] = {
            "input_tokens": total_input,
            "output_tokens": output_tokens,
            "total_tokens": total_input + output_tokens,
        }
        if cache_creation:
            usage_kwargs["cache_creation_input_tokens"] = cache_creation
        if cache_read:
            usage_kwargs["cache_read_input_tokens"] = cache_read

        usage = Usage(**usage_kwargs)

        # Determine finish reason
        finish_reason = "tool_use" if tool_calls else "end_turn"

        logger.info(
            f"[PROVIDER] Response: {len(clean_text)} chars, "
            f"{len(tool_call_dicts)} tool calls, {duration:.2f}s "
            f"(tokens: {total_input} in, {output_tokens} out)"
        )

        return ClaudeChatResponse(
            content=content_blocks,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata,
            content_blocks=event_blocks if event_blocks else None,
            text=clean_text or None,
        )

    # -------------------------------------------------------------------------
    # Event emission
    # -------------------------------------------------------------------------

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event through the coordinator's hooks if available.

        Args:
            event: Event name.
            data: Event data.
        """
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(event, data)
