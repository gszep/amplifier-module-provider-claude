"""Claude provider module for Amplifier.

Integrates with Claude Code CLI via claude-agent-sdk for Claude Max subscription usage.
Enables using Amplifier with a Claude Max subscription instead of Anthropic API billing.
"""

__all__ = ["mount", "ClaudeProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import logging
import time
from typing import Any

from amplifier_core import (
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)

logger = logging.getLogger(__name__)

# Maximum size for system prompt to prevent "Argument list too long" errors
# The kernel's ARG_MAX limit is ~2 MB, safe limit for system_prompt is ~500 KB
MAX_SYSTEM_PROMPT_BYTES = 500_000


class ClaudeChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


# Claude Code built-in tools that can be allowed/disallowed
CLAUDE_CODE_BUILTIN_TOOLS = frozenset(
    {
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "Bash",
        "Glob",
        "Grep",
        "LS",
        "TodoRead",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "NotebookRead",
        "NotebookEdit",
        "Task",
    }
)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Claude provider using Claude Code (Claude Max subscription).

    This fork uses Claude Code CLI for all requests, enabling use of Claude Max
    subscription instead of API billing.

    Args:
        coordinator: Module coordinator
        config: Provider configuration

    Returns:
        Optional cleanup function (None - SDK manages its own resources)
    """
    config = config or {}

    provider = ClaudeProvider(config, coordinator)
    await coordinator.mount("providers", provider, name="claude")
    logger.info("Mounted ClaudeProvider (via Claude Code subscription)")

    # No cleanup needed - claude-agent-sdk manages subprocess lifecycle
    return None


class ClaudeProvider:
    """Claude Code integration via claude-agent-sdk.

    Provides Claude models through Claude Code CLI, using a Claude Max
    subscription instead of direct API billing.

    Features:
    - Uses Claude Max subscription (no API key required)
    - Supports sonnet, opus, and haiku models
    - Streaming responses via async iterator
    - Tool calling support with automatic mapping
    - Session continuity (continue/resume)
    """

    name = "claude"
    api_label = "Claude (Claude Code)"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """
        Initialize Claude Code provider.

        Args:
            config: Provider configuration
            coordinator: Module coordinator for event emission
        """
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "sonnet")
        self.max_turns = self.config.get("max_turns", 1)
        self.debug = self.config.get("debug", False)

        # Session management
        self._last_session_id: str | None = None
        self._session_cwd: str | None = self.config.get("cwd")

        # Tool configuration
        self._allowed_tools: list[str] | None = self.config.get("allowed_tools")
        self._disallowed_tools: list[str] | None = self.config.get("disallowed_tools")

        # Permission mode: 'default', 'plan', 'acceptEdits', 'bypassPermissions'
        self._permission_mode: str | None = self.config.get("permission_mode")

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="claude",
            display_name="Claude",
            credential_env_vars=[],  # Uses Claude Code's own authentication
            capabilities=["streaming", "tools"],
            defaults={
                "model": "sonnet",
                "max_turns": 1,
            },
            config_fields=[],  # No configuration needed - uses Claude Code CLI
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available Claude Code models.

        Returns:
            List of ModelInfo for available models (sonnet, opus, haiku).
        """
        return [
            ModelInfo(
                id="sonnet",
                display_name="Claude Sonnet (via Claude Code)",
                context_window=200000,
                max_output_tokens=64000,
                capabilities=["tools", "streaming"],
                defaults={"max_tokens": 64000},
            ),
            ModelInfo(
                id="opus",
                display_name="Claude Opus (via Claude Code)",
                context_window=200000,
                max_output_tokens=64000,
                capabilities=["tools", "thinking", "streaming"],
                defaults={"max_tokens": 64000},
            ),
            ModelInfo(
                id="haiku",
                display_name="Claude Haiku (via Claude Code)",
                context_window=200000,
                max_output_tokens=64000,
                capabilities=["tools", "streaming", "fast"],
                defaults={"max_tokens": 64000},
            ),
        ]

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion via Claude Code.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options:
                - model: Model alias (sonnet, opus, haiku)
                - max_turns: Maximum agentic turns
                - continue_session: Continue last session in cwd
                - session_id: Resume specific session by ID
                - allowed_tools: List of tools to allow
                - disallowed_tools: List of tools to disallow
                - permission_mode: Permission mode override

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        # Import here to allow module to load even if claude-agent-sdk not installed
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKError,
            CLIConnectionError,
            CLIJSONDecodeError,
            CLINotFoundError,
            ProcessError,
            ToolUseBlock,
            query,
        )
        from claude_agent_sdk import TextBlock as SDKTextBlock
        from claude_agent_sdk import ThinkingBlock as SDKThinkingBlock

        start_time = time.time()

        # Extract prompt and system message from request
        prompt = self._extract_prompt(request)
        system_prompt = self._extract_system_prompt(request)

        # Truncate system prompt if it exceeds the safe size limit
        # This prevents "Argument list too long" errors from subprocess invocation
        if system_prompt and len(system_prompt) > MAX_SYSTEM_PROMPT_BYTES:
            original_size = len(system_prompt)
            logger.warning(
                f"System prompt truncated from {original_size:,} to {MAX_SYSTEM_PROMPT_BYTES:,} bytes "
                f"to prevent 'Argument list too long' error when invoking Claude Code"
            )
            system_prompt = system_prompt[:MAX_SYSTEM_PROMPT_BYTES]
            system_prompt += "\n\n[...truncated due to size limit...]"

        if not prompt:
            logger.warning("[PROVIDER] Claude Code: No user prompt found in request")
            return ChatResponse(
                content=[TextBlock(text="No prompt provided")],
                tool_calls=None,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                finish_reason="error",
            )

        # Build options for Claude Agent SDK
        model = kwargs.get("model", self.default_model)
        max_turns = kwargs.get("max_turns", self.max_turns)

        # Build allowed/disallowed tools list
        allowed_tools = self._build_allowed_tools(request, kwargs)
        disallowed_tools = kwargs.get("disallowed_tools", self._disallowed_tools)

        # Permission mode
        permission_mode = kwargs.get("permission_mode", self._permission_mode)

        # Session continuity
        continue_session = kwargs.get("continue_session", False)
        resume_session_id = kwargs.get("session_id")

        # Build ClaudeAgentOptions
        options_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "max_turns": max_turns,
        }

        if self._session_cwd:
            options_kwargs["cwd"] = self._session_cwd

        if allowed_tools:
            options_kwargs["allowed_tools"] = allowed_tools

        if disallowed_tools:
            options_kwargs["disallowed_tools"] = disallowed_tools

        if permission_mode:
            options_kwargs["permission_mode"] = permission_mode

        # Session continuity options
        if continue_session:
            options_kwargs["continue_session"] = True
        elif resume_session_id:
            options_kwargs["session_id"] = resume_session_id
        elif self._last_session_id and kwargs.get("auto_continue", False):
            options_kwargs["session_id"] = self._last_session_id

        options = ClaudeAgentOptions(**options_kwargs)

        logger.info(
            f"[PROVIDER] Claude Code API call - model: {model}, max_turns: {max_turns}, "
            f"tools: {len(allowed_tools) if allowed_tools else 'default'}"
        )

        # Emit request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "claude-code",
                    "model": model,
                    "message_count": len(request.messages),
                    "has_system": system_prompt is not None,
                    "has_tools": bool(allowed_tools),
                    "continue_session": continue_session,
                    "resume_session": resume_session_id is not None,
                },
            )

        # Collect response from async iterator
        content_blocks: list[TextBlock | ThinkingBlock | ToolCallBlock] = []
        tool_calls: list[ToolCall] = []
        session_id: str | None = None

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, SDKTextBlock):
                            content_blocks.append(TextBlock(text=block.text))
                        elif isinstance(block, SDKThinkingBlock):
                            content_blocks.append(
                                ThinkingBlock(
                                    thinking=block.thinking,
                                    signature=getattr(block, "signature", None),
                                )
                            )
                        elif isinstance(block, ToolUseBlock):
                            content_blocks.append(
                                ToolCallBlock(
                                    id=block.id, name=block.name, input=block.input
                                )
                            )
                            tool_calls.append(
                                ToolCall(
                                    id=block.id, name=block.name, arguments=block.input
                                )
                            )

                # Capture session ID for continuity
                if hasattr(message, "session_id") and message.session_id:
                    session_id = message.session_id

            # Store session ID for potential continuation
            if session_id:
                self._last_session_id = session_id

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Emit success response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "claude-code",
                        "model": model,
                        "status": "ok",
                        "content_blocks": len(content_blocks),
                        "tool_calls": len(tool_calls),
                        "duration_ms": elapsed_ms,
                        "session_id": session_id,
                    },
                )

        except CLINotFoundError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = (
                "Claude Code CLI not found. Please install it: "
                "curl -fsSL https://claude.ai/install.sh | bash"
            )
            logger.error(f"[PROVIDER] {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise RuntimeError(error_msg) from e

        except CLIConnectionError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Claude Code connection error: {e}"
            logger.error(f"[PROVIDER] {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise RuntimeError(error_msg) from e

        except ProcessError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            exit_code = getattr(e, "exit_code", "unknown")
            error_msg = f"Claude Code process failed (exit code {exit_code}): {e}"
            logger.error(f"[PROVIDER] {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise RuntimeError(error_msg) from e

        except CLIJSONDecodeError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Claude Code returned invalid JSON: {e}"
            logger.error(f"[PROVIDER] {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise RuntimeError(error_msg) from e

        except ClaudeSDKError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Claude Code SDK error: {e}"
            logger.error(f"[PROVIDER] {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error(f"[PROVIDER] Claude Code error: {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise

        # Build response
        # Note: Claude Code doesn't expose token usage metrics
        return ChatResponse(
            content=content_blocks if content_blocks else [TextBlock(text="")],
            tool_calls=tool_calls if tool_calls else None,
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            finish_reason="end_turn",
        )

    async def _emit_error_event(
        self, model: str, error_msg: str, elapsed_ms: int
    ) -> None:
        """Emit error event to hooks."""
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(
                "llm:response",
                {
                    "provider": "claude-code",
                    "model": model,
                    "status": "error",
                    "error": error_msg,
                    "duration_ms": elapsed_ms,
                },
            )

    def _build_allowed_tools(
        self, request: ChatRequest, kwargs: dict[str, Any]
    ) -> list[str] | None:
        """Build the list of allowed tools for Claude Code.

        Maps Amplifier tool names to Claude Code built-in tools where possible,
        and includes any explicitly configured allowed tools.

        Args:
            request: Chat request with optional tools
            kwargs: Additional options that may override allowed_tools

        Returns:
            List of allowed tool names, or None for default behavior
        """
        # Start with explicitly configured allowed tools
        allowed = kwargs.get("allowed_tools", self._allowed_tools)
        if allowed is not None:
            return list(allowed)

        # If request has tools, try to map them to Claude Code built-ins
        if request.tools:
            mapped_tools: list[str] = []
            for tool in request.tools:
                # Direct match with Claude Code built-in
                if tool.name in CLAUDE_CODE_BUILTIN_TOOLS:
                    mapped_tools.append(tool.name)
                # Map common Amplifier tool names to Claude Code equivalents
                elif tool.name in _AMPLIFIER_TO_CLAUDE_CODE_TOOL_MAP:
                    mapped_tools.append(_AMPLIFIER_TO_CLAUDE_CODE_TOOL_MAP[tool.name])

            if mapped_tools:
                return mapped_tools

        return None

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Args:
            response: Typed chat response

        Returns:
            List of tool calls from the response
        """
        return response.tool_calls or []

    def _extract_prompt(self, request: ChatRequest) -> str:
        """Extract user prompt from ChatRequest.

        Finds the most recent user message content.

        Args:
            request: Chat request with messages

        Returns:
            User prompt string, or empty string if not found
        """
        for msg in reversed(request.messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                    if text_parts:
                        return "\n".join(text_parts)
        return ""

    def _extract_system_prompt(self, request: ChatRequest) -> str | None:
        """Extract system prompt from ChatRequest.

        Combines all system messages into a single prompt.

        Args:
            request: Chat request with messages

        Returns:
            Combined system prompt, or None if no system messages
        """
        system_parts = []
        for msg in request.messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_parts.append(msg.content)
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            system_parts.append(block.text)
                        elif isinstance(block, dict) and "text" in block:
                            system_parts.append(block["text"])

        return "\n\n".join(system_parts) if system_parts else None

    @property
    def last_session_id(self) -> str | None:
        """Get the last session ID for continuation."""
        return self._last_session_id


# Mapping from common Amplifier tool names to Claude Code built-in tools
_AMPLIFIER_TO_CLAUDE_CODE_TOOL_MAP: dict[str, str] = {
    # File operations
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "multi_edit": "MultiEdit",
    # Shell
    "bash": "Bash",
    "shell": "Bash",
    "execute": "Bash",
    # Search
    "glob": "Glob",
    "grep": "Grep",
    "search": "Grep",
    "find_files": "Glob",
    # Directory
    "list_directory": "LS",
    "ls": "LS",
    # Web
    "web_fetch": "WebFetch",
    "fetch_url": "WebFetch",
    "web_search": "WebSearch",
    # Task delegation
    "task": "Task",
    "spawn_agent": "Task",
    # Notebook
    "notebook_read": "NotebookRead",
    "notebook_edit": "NotebookEdit",
}
