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

__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import re
import shutil
import time
import uuid
from collections import defaultdict
from typing import Any

from amplifier_core import (  # type: ignore
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
    TextContent,
    ThinkingContent,
    ToolCallContent,
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
from amplifier_core.utils import truncate_values
from claude_agent_sdk import ClaudeSDKClient  # type: ignore[import-untyped]
from claude_agent_sdk._errors import (  # type: ignore[import-untyped]
    ClaudeSDKError,
)
from claude_agent_sdk._errors import (
    CLINotFoundError as SDKCLINotFoundError,
)
from claude_agent_sdk._errors import (
    ProcessError as SDKProcessError,
)
from claude_agent_sdk.types import (  # type: ignore[import-untyped]
    AssistantMessage as SDKAssistantMessage,
)
from claude_agent_sdk.types import (
    ClaudeAgentOptions,
)
from claude_agent_sdk.types import (
    ResultMessage as SDKResultMessage,
)
from claude_agent_sdk.types import (
    TextBlock as SDKTextBlock,
)
from claude_agent_sdk.types import (
    ThinkingBlock as SDKThinkingBlock,
)

from .sessions import SessionManager, SessionMetadata, SessionState

logger = logging.getLogger(__name__)


class ClaudeChatResponse(ChatResponse):
    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


class ClaudeProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str = "unknown",
        is_recoverable: bool = False,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.is_recoverable = is_recoverable
        self.details = details or {}

    def __str__(self) -> str:
        return f"{self.error_type}: {super().__str__()}"


class ContextLimitExceededError(ClaudeProviderError):
    def __init__(
        self,
        message: str = "Prompt is too long",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message,
            error_type="context_limit_exceeded",
            is_recoverable=False,
            details=details,
        )


class RateLimitError(ClaudeProviderError):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message,
            error_type="rate_limit",
            is_recoverable=True,
            details=details,
        )


class InvalidRequestError(ClaudeProviderError):
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message,
            error_type="invalid_request",
            is_recoverable=False,
            details=details,
        )


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


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Claude provider using Claude Code CLI.

    Args:
        coordinator: The module coordinator to mount to.
        config: Optional configuration dictionary.

    Returns:
        Optional cleanup function.
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

    async def cleanup():
        try:
            await asyncio.shield(asyncio.to_thread(provider._save_session))
        except (asyncio.CancelledError, Exception):
            pass

    return cleanup


class ClaudeProvider:
    """Claude Code CLI integration for Amplifier.

    Full Control mode: Amplifier's orchestrator handles all tool execution.
    Claude only decides which tools to call.
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

        self.default_model = self.config.get("default_model", DEFAULT_MODEL)
        self.timeout = self.config.get("timeout", DEFAULT_TIMEOUT)
        self.debug = self.config.get("debug", False)
        self.raw_debug = self.config.get("raw_debug", False)
        self.debug_truncate_length = self.config.get("debug_truncate_length", 180)
        self.max_thinking_tokens = max(
            MIN_THINKING_TOKENS,
            self.config.get("max_thinking_tokens", DEFAULT_MAX_THINKING_TOKENS),
        )

        self._repaired_tool_ids: set[str] = set()
        self._session_manager = SessionManager(
            session_dir=self.config.get("session_dir"),
        )

        amplifier_session_id = self._get_amplifier_session_id()
        self._session_state = self._session_manager.get_or_create_session(
            session_id=amplifier_session_id,
            name=self.config.get("session_name", "amplifier-claude"),
        )

        self._valid_tool_names: set[str] = set()
        self._filtered_tool_calls: list[dict[str, Any]] = []

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Delegates to shared utility from amplifier_core.utils.
        """
        if max_length is None:
            max_length = self.debug_truncate_length
        return truncate_values(obj, max_length)

    def _get_amplifier_session_id(self) -> str | None:
        """Get the Amplifier session ID from the coordinator.

        Returns:
            Session ID string if available, None otherwise.
        """
        if not self.coordinator:
            return None

        if hasattr(self.coordinator, "session"):
            session = getattr(self.coordinator, "session", None)
            if session and hasattr(session, "id"):
                return str(session.id)

        if hasattr(self.coordinator, "config"):
            config = getattr(self.coordinator, "config", {})
            if isinstance(config, dict) and "session_id" in config:
                return str(config["session_id"])

        return None

    def _get_claude_session_id(self) -> str | None:
        return self._session_state.metadata.claude_session_id

    def _save_session(self) -> None:
        self._session_manager.save_session(self._session_state)
        if self.debug:
            efficiency = self._session_state.get_cache_efficiency()
            logger.debug(
                f"[PROVIDER] Session saved: {self._session_state.metadata.session_id}, "
                f"cache efficiency: {efficiency:.1%}"
            )

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="claude",
            display_name="Claude Code",
            credential_env_vars=[],
            capabilities=["streaming", "tools", "thinking"],
            defaults={
                "model": self.default_model,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "timeout": self.timeout,
            },
            config_fields=[],
        )

    async def list_models(self) -> list[ModelInfo]:
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

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        tool_calls = {}  # {call_id: (msg_index, name, args)}
        tool_results = set()  # {call_id}

        for idx, msg in enumerate(messages):
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

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

        self._valid_tool_names = set()
        if request.tools:
            for tool in request.tools:
                if hasattr(tool, "name"):
                    self._valid_tool_names.add(tool.name)
                elif isinstance(tool, dict) and "name" in tool:
                    self._valid_tool_names.add(tool["name"])

        previous_filtered_calls = self._filtered_tool_calls.copy()
        self._filtered_tool_calls = []

        missing = self._find_missing_tool_results(request.messages)
        if missing:
            logger.warning(
                f"[PROVIDER] Claude: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    self._repaired_tool_ids.add(call_id)

                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

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
        model = (
            kwargs.get("model") or getattr(request, "model", None) or self.default_model
        )

        request_metadata = getattr(request, "metadata", None) or {}
        existing_session_id = (
            request_metadata.get(METADATA_SESSION_ID) or self._get_claude_session_id()
        )

        resuming = existing_session_id is not None
        system_prompt, user_prompt = self._convert_messages(
            request.messages, request.tools, resuming=resuming
        )

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        if self.debug:
            logger.debug(f"[PROVIDER] System prompt length: {len(system_prompt)}")
            logger.debug(f"[PROVIDER] User prompt: {user_prompt[:200]}...")

        # Emit request events
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

        # DEBUG level: Request details with truncated values
        if self.debug:
            await self._emit_event(
                "llm:request:debug",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "request": self._truncate_values(
                        {
                            "model": model,
                            "prompt_length": len(full_prompt),
                            "resume_session": existing_session_id,
                            "tools_count": len(request.tools) if request.tools else 0,
                            "max_thinking_tokens": self.max_thinking_tokens,
                        }
                    ),
                },
            )

        # RAW level: Complete prompt as sent to CLI
        if self.debug and self.raw_debug:
            await self._emit_event(
                "llm:request:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "prompt": full_prompt,
                    "model": model,
                    "session_id": existing_session_id,
                },
            )

        try:
            async with asyncio.timeout(self.timeout):
                response_data = await self._execute_sdk_query(
                    full_prompt, model, existing_session_id
                )
        except TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Request timed out after {self.timeout}s"
            logger.error(f"[PROVIDER] {error_msg}")
            await self._emit_event(
                "llm:response",
                {
                    "provider": self.name,
                    "model": model,
                    "status": "error",
                    "duration_ms": elapsed_ms,
                    "error": error_msg,
                },
            )
            raise TimeoutError(error_msg) from None
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error(f"[PROVIDER] Claude error: {error_msg}")
            await self._emit_event(
                "llm:response",
                {
                    "provider": self.name,
                    "model": model,
                    "status": "error",
                    "duration_ms": elapsed_ms,
                    "error": error_msg,
                },
            )
            if not str(e):
                raise type(e)(error_msg) from e
            raise

        response_session_id = response_data.get("metadata", {}).get(METADATA_SESSION_ID)
        if response_session_id:
            self._session_state.set_claude_session_id(response_session_id)
            logger.debug(
                f"[PROVIDER] Stored session ID for resumption: {response_session_id}"
            )

        duration = time.time() - start_time
        chat_response = self._build_response(response_data, duration)

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

        # Emit response events
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

        # DEBUG level: Full response with truncated values
        if self.debug:
            await self._emit_event(
                "llm:response:debug",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "response": self._truncate_values(response_data),
                    "status": "ok",
                    "duration_ms": int(duration * 1000),
                },
            )

        # RAW level: Complete response data
        if self.debug and self.raw_debug:
            await self._emit_event(
                "llm:response:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "response": response_data,
                },
            )

        return chat_response

    def _convert_messages(
        self,
        messages: list[Message],
        tools: list[Any] | None,
        resuming: bool = False,
    ) -> tuple[str, str]:
        system_parts = []
        conversation_parts = []
        tool_schema = ""

        if resuming:
            messages = self._get_current_turn_messages(messages)
        else:
            if tools:
                tool_definitions = self._convert_tools(tools)
                tool_schema = self._build_tool_schema(tool_definitions)

        for msg in messages:
            role = msg.role
            content = self._extract_content(msg)

            if role == "system":
                if not resuming:
                    system_parts.append(f"<system-reminder>{content}</system-reminder>")

            elif role == "user":
                if content.strip().startswith("<system-reminder"):
                    conversation_parts.append(content)
                else:
                    conversation_parts.append(f"<user>{content}</user>")

            elif role == "assistant":
                assistant_content = self._format_assistant_message(msg)
                conversation_parts.append(f"<assistant>{assistant_content}</assistant>")

            elif role == "tool":
                tool_result = self._format_tool_result(msg)
                conversation_parts.append(f"{tool_result}")

            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                conversation_parts.append(f"{wrapped}")

        if tool_schema:
            system_parts.append(tool_schema)
        system_prompt = "\n\n".join(system_parts) if system_parts else ""
        user_prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

        if len(messages) == 1 and messages[0].role == "user":
            user_prompt = self._extract_content(messages[0])

        return system_prompt, user_prompt

    def _get_current_turn_messages(self, messages: list[Message]) -> list[Message]:
        last_assistant_idx = -1

        for i, msg in enumerate(messages):
            if msg.role == "assistant":
                last_assistant_idx = i

        if last_assistant_idx == -1:
            return messages

        current_turn = messages[last_assistant_idx + 1 :]
        return current_turn

    def _extract_content(self, msg: Message) -> str:
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
                            },
                            default=str,
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
                            },
                            default=str,
                        )
                        parts.append(f"<tool_use>{tool_call_str}</tool_use>")

        return "\n".join(parts)

    def _format_tool_result(self, msg: Message) -> str:
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

        return f"<tool_result>{json.dumps(result, default=str)}</tool_result>"

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        tool_definitions = []

        for tool in tools:
            if hasattr(tool, "name"):
                tool_def = {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "input_schema": getattr(tool, "parameters", {}),
                }
            elif isinstance(tool, dict):
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
        if not tools:
            return ""

        tools_json = json.dumps(tools, default=str)
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

To call a tool, generate a valid JSON with "tool", "id", and "input" fields, wrapped in XML tags in this format:
<tool_use>
{tool_use_example}
</tool_use>

CRITICAL RULES:
1. Place ALL <tool_use> blocks at the END of your response
2. After generating <tool_use> blocks, STOP IMMEDIATELY - do not generate any content after them
3. Do NOT generate hypothetical tool results or continue with analysis before receiving actual results
4. Do NOT generate JSON arrays with "result_type": "tool_result" - wait for actual <tool_result> blocks
5. Multiple tool calls are allowed - generate separate <tool_use> blocks for each

Generate a unique ID for each call (e.g., "call_1", "call_2").
Tool results will be provided in <tool_result> blocks in the next message. Wait for them.
</system-reminder>"""

    async def _execute_sdk_query(
        self,
        prompt: str,
        model: str,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Execute a query via the Claude Agent SDK using ClaudeSDKClient.

        Uses ClaudeSDKClient for better session management while keeping
        the XML-based tool protocol for Amplifier orchestrator compatibility.
        """
        if session_id:
            logger.info(f"[PROVIDER] Resuming Claude session: {session_id}")

        sdk_options = ClaudeAgentOptions(
            tools=[],  # Disable built-in tools (Full Control mode)
            model=model,
            system_prompt="",  # System prompt is included in user message
            resume=session_id,
            max_thinking_tokens=(
                self.max_thinking_tokens if self.max_thinking_tokens > 0 else None
            ),
            include_partial_messages=True,
        )

        logger.debug("[PROVIDER] Executing Claude CLI via ClaudeSDKClient")

        response_text = ""
        thinking_text = ""
        thinking_signature = ""
        usage_data: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        _text_block_index = 0
        _thinking_block_index = 0

        try:
            async with ClaudeSDKClient(options=sdk_options) as client:
                await client.query(prompt)

                async for msg in client.receive_response():
                    if isinstance(msg, SDKAssistantMessage):
                        if msg.error:
                            error_labels = {
                                "rate_limit": "Rate limit exceeded",
                                "authentication_failed": "Authentication failed",
                                "billing_error": "Billing error",
                                "invalid_request": "Invalid request",
                                "server_error": "Server error",
                            }
                            raise RuntimeError(
                                f"Claude Code error: "
                                f"{error_labels.get(msg.error, msg.error)}"
                            )

                        for block in msg.content:
                            if isinstance(block, SDKTextBlock):
                                if response_text and block.text:
                                    response_text += "\n" + block.text
                                else:
                                    response_text += block.text

                                if block.text:
                                    if _text_block_index == 0:
                                        await self._emit_event(
                                            "content_block:start",
                                            {
                                                "provider": self.name,
                                                "model": model,
                                                "type": "text",
                                                "index": _text_block_index,
                                            },
                                        )
                                    await self._emit_event(
                                        "content_block:delta",
                                        {
                                            "provider": self.name,
                                            "model": model,
                                            "type": "text",
                                            "index": _text_block_index,
                                            "text": block.text,
                                        },
                                    )
                                    _text_block_index += 1

                            elif isinstance(block, SDKThinkingBlock):
                                thinking_text = block.thinking
                                thinking_signature = block.signature
                                logger.debug(
                                    f"[PROVIDER] Thinking block: {len(thinking_text)} chars"
                                )

                                if block.thinking:
                                    if _thinking_block_index == 0:
                                        await self._emit_event(
                                            "content_block:start",
                                            {
                                                "provider": self.name,
                                                "model": model,
                                                "type": "thinking",
                                                "index": _thinking_block_index,
                                            },
                                        )
                                    await self._emit_event(
                                        "thinking:delta",
                                        {
                                            "provider": self.name,
                                            "model": model,
                                            "index": _thinking_block_index,
                                            "text": block.thinking,
                                        },
                                    )
                                    _thinking_block_index += 1

                    elif isinstance(msg, SDKResultMessage):
                        if not response_text and msg.result:
                            response_text = msg.result

                        metadata = {
                            METADATA_SESSION_ID: msg.session_id,
                            METADATA_DURATION_MS: msg.duration_ms,
                            METADATA_COST_USD: msg.total_cost_usd,
                            "num_turns": msg.num_turns,
                        }
                        usage_data = msg.usage or {}

        except SDKCLINotFoundError as e:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: "
                "curl -fsSL https://claude.ai/install.sh | bash"
            ) from e
        except SDKProcessError as e:
            error_msg = e.stderr or ""
            if error_msg:
                logger.error(f"[PROVIDER] CLI failed: {error_msg}")
                raise RuntimeError(
                    f"Claude Code CLI failed (exit {e.exit_code}): {error_msg}"
                ) from e
            else:
                raise RuntimeError(
                    f"Claude Code CLI failed (exit {e.exit_code}): "
                    "Limits may have been exceeded.\n"
                    "Check usage https://claude.ai/settings/usage "
                    "and https://platform.claude.com/settings/billing.\n"
                    "If API billing is being used, this means Amplifier "
                    "has access to ANTHROPIC_API_KEY."
                ) from e
        except ClaudeSDKError as e:
            raise RuntimeError(f"Claude Code SDK error: {e}") from e

        # Close any open content block streams
        if _thinking_block_index > 0:
            await self._emit_event(
                "thinking:final",
                {
                    "provider": self.name,
                    "model": model,
                    "text": thinking_text,
                },
            )
            await self._emit_event(
                "content_block:end",
                {
                    "provider": self.name,
                    "model": model,
                    "type": "thinking",
                    "index": _thinking_block_index - 1,
                },
            )
        if _text_block_index > 0:
            await self._emit_event(
                "content_block:end",
                {
                    "provider": self.name,
                    "model": model,
                    "type": "text",
                    "index": _text_block_index - 1,
                },
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

    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from response text."""
        tool_calls = []
        import re

        code_block_ranges = [
            (m.start(), m.end()) for m in re.finditer(r"```[\s\S]*?```", text)
        ]

        def is_inside_code_block(pos: int) -> bool:
            return any(start <= pos < end for start, end in code_block_ranges)

        pattern = r"<tool_use>\s*(.*?)\s*</tool_use>"
        for match in re.finditer(pattern, text, re.DOTALL):
            if is_inside_code_block(match.start()):
                logger.debug("[PROVIDER] Skipping tool_use inside code block")
                continue

            content = match.group(1)
            stripped = content.strip()
            if not stripped.startswith("{"):
                logger.debug(f"[PROVIDER] Skipping non-JSON: {stripped[:50]}...")
                continue

            try:
                tool_data = json.loads(content)
                tool_call = {
                    "id": tool_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": tool_data.get("tool", tool_data.get("name", "")),
                    "arguments": tool_data.get("input", tool_data.get("arguments", {})),
                }

                if (
                    self._valid_tool_names
                    and tool_call["name"] not in self._valid_tool_names
                ):
                    logger.debug(
                        f"[PROVIDER] Filtering invalid tool: {tool_call['name']!r}"
                    )
                    self._filtered_tool_calls.append(tool_call)
                    continue

                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                error_detail = self._format_json_parse_error(content, e)
                logger.warning(f"[PROVIDER] Failed to parse tool JSON:\n{error_detail}")
                continue

        return tool_calls

    def _format_json_parse_error(
        self, content: str, error: json.JSONDecodeError
    ) -> str:
        pos = error.pos
        lineno = error.lineno
        colno = error.colno

        context_before = 40
        context_after = 40

        start = max(0, pos - context_before)
        end = min(len(content), pos + context_after)

        excerpt = content[start:end]
        pointer_pos = pos - start

        pointer_line = " " * pointer_pos + "^"
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""

        lines = [
            f"JSON error at line {lineno}, column {colno}: {error.msg}",
            f"Content ({len(content)} chars total):",
            f"  {prefix}{excerpt}{suffix}",
            f"  {' ' * len(prefix)}{pointer_line}",
        ]

        if len(content) <= 200:
            lines.append(f"Full content: {content!r}")

        return "\n".join(lines)

    def _clean_response_text(self, text: str) -> str:
        cleaned = re.sub(r"<tool_use>.*?</tool_use>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def _build_response(
        self, response_data: dict[str, Any], duration: float
    ) -> ClaudeChatResponse:
        """Build a ClaudeChatResponse from parsed response data."""

        raw_text = response_data.get("text", "")
        tool_call_dicts = response_data.get("tool_calls", [])
        usage_data = response_data.get("usage", {})
        metadata = response_data.get("metadata", {})

        clean_text = self._clean_response_text(raw_text)

        # Build content blocks - ORDER: thinking -> text -> tool_use
        content_blocks: list[Any] = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []

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
            event_blocks.append(ThinkingContent(text=thinking_content))

        if clean_text:
            content_blocks.append(TextBlock(text=clean_text))
            event_blocks.append(TextContent(text=clean_text))

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

        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        cache_read = usage_data.get("cache_read_input_tokens", 0)
        cache_creation = usage_data.get("cache_creation_input_tokens", 0)
        total_input = input_tokens + cache_read + cache_creation

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

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(event, data)
