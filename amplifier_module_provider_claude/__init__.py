"""Claude provider module for Amplifier.

Full Control implementation using Claude Code CLI.
Amplifier's orchestrator handles tool execution - Claude only decides which tools to call.
"""

__all__ = ["mount", "ClaudeProvider"]
__amplifier_module_type__ = "provider"
import asyncio
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import claude_agent_sdk  # type: ignore
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
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)
from amplifier_core.utils import truncate_values
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock  # type: ignore
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock  # type: ignore
from anthropic.types.parsed_message import (  # type: ignore
    ParsedContentBlock,
    ParsedMessage,
    ParsedTextBlock,
)
from anthropic.types.usage import Usage as AnthropicUsage  # type: ignore
from claude_agent_sdk import ClaudeSDKClient  # type: ignore
from claude_agent_sdk.types import ClaudeAgentOptions  # type: ignore
from pydantic import ValidationError

logger = logging.getLogger(__name__)
SESSION_TAG = "[session]:"
SESSION = SESSION_TAG + """{"id": null, "last_message_idx": 0}"""


@dataclass
class WebSearchContent:
    """Content block for web search results from native Anthropic web search."""

    type: str = "web_search"
    query: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)


class Session(RedactedThinkingBlock):
    """Content block for storing the session ID."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    visibility: Literal["internal"] = "internal"
    data: str = SESSION

    @property
    def json_string(self) -> str:
        return self.data.replace(SESSION_TAG, "")

    @json_string.setter
    def json_string(self, value: str):
        self.data = f"{SESSION_TAG}{value}"

    @property
    def json(self) -> dict[str, int | str]:
        return json.loads(self.json_string)

    @json.setter
    def json(self, value: dict[str, int | str]):
        self.json_string = json.dumps(value)

    @property
    def id(self) -> str | None:
        return self.json.get("id", None)

    @id.setter
    def id(self, value: str | None):
        self.json |= {"id": value}

    @property
    def last_message_idx(self) -> int:
        return self.json.get("last_message_idx", 0)

    @last_message_idx.setter
    def last_message_idx(self, value: int):
        self.json |= {"last_message_idx": value}


class ClaudeChatResponse(ChatResponse):
    content_blocks: (
        list[
            TextContent | ThinkingContent | ToolCallContent | WebSearchContent | Session
        ]
        | None
    ) = None
    text: str | None = None
    web_search_results: list[dict[str, Any]] | None = None


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
    logger.info("Mounted ClaudeProvider")

    async def cleanup():
        try:
            await asyncio.shield(asyncio.sleep(0))  # placeholder for any async cleanup
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

        self.default_model: str = self.config.get("default_model", "sonnet")
        self.temperature: float = self.config.get("temperature", 1.0)  # thinking
        self.enable_prompt_caching = True  # always enable prompt caching

        self._beta_headers: list[str] = []
        self.context_window = 200000  # beta-headers: 1m context not available

        self.max_tokens: int = self.config.get("max_tokens", 64000)
        self.max_output_tokens: int = 64000  # claude models support 64K output
        self.max_thinking_tokens: int = self.config.get("max_thinking_tokens", 32000)

        self.priority: int = self.config.get("priority", 100)
        self.debug: bool = self.config.get("debug", False)
        self.raw_debug: bool = self.config.get("raw_debug", False)
        self.debug_truncate_length: int = self.config.get("debug_truncate_length", 180)

        self.timeout: float = self.config.get("timeout", 300.0)
        self.use_streaming = True  # sdk only supports streaming
        self.enable_web_search: bool = self.config.get("enable_web_search", False)

        self._repaired_tool_ids: set[str] = set()
        self._session: Session = Session()

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="claude",
            display_name="Claude Code",
            credential_env_vars=[],
            capabilities=["streaming", "tools", "thinking"],
            defaults={
                "model": self.default_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout": self.timeout,
                "context_window": self.context_window,
                "max_output_tokens": self.max_output_tokens,
            },
            config_fields=[],
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="sonnet",
                display_name="Claude Sonnet",
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
                capabilities=["tools", "thinking", "streaming", "json_mode"],
                defaults={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            ),
            ModelInfo(
                id="opus",
                display_name="Claude Opus",
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
                capabilities=["tools", "thinking", "streaming", "json_mode"],
                defaults={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            ),
            ModelInfo(
                id="haiku",
                display_name="Claude Haiku",
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
                capabilities=["tools", "streaming", "json_mode", "fast"],
                defaults={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            ),
        ]

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Delegates to shared utility from amplifier_core.utils.
        """
        if max_length is None:
            max_length = self.debug_truncate_length
        return truncate_values(obj, max_length)

    def _count_trailing_reminders(self, messages: list[Message]) -> int:
        """Count trailing system reminder messages from end of list."""
        count = 0
        for m in reversed(messages):
            content = m.content if isinstance(m.content, str) else ""
            if m.role == "user" and "<system-reminder" in content:
                count += 1
            else:
                break
        return count

    def _get_recent_messages(self, messages: list[Message]) -> list[Message]:
        has_system_prefix = bool(messages) and messages[0].role == "system"
        next_message_idx: int | None = None

        if has_system_prefix:
            trailing_reminders = self._count_trailing_reminders(messages)
            next_message_idx = len(messages) - trailing_reminders

            if self._session.id and self._session.last_message_idx > 0:
                idx = self._session.last_message_idx

                # Find end of system messages at the start
                system_end = 0
                for i, m in enumerate(messages):
                    if m.role == "system":
                        system_end = i + 1
                    else:
                        break

                # Only subset if there are old messages to skip
                if idx > system_end:
                    messages = list(messages[:system_end]) + list(messages[idx:])

        # Update session index for next turn's message subsetting
        if next_message_idx is not None:
            self._session.last_message_idx = next_message_idx

        return messages

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
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Anthropic: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            from collections import defaultdict

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

        return await self._complete_chat_request(request, **kwargs)

    def _format_system_with_cache(
        self, system_msgs: list[Message]
    ) -> list[dict[str, Any]] | None:
        """Format system messages as content block array with cache_control.

        Anthropic requires system as array of content blocks for caching.
        Cache breakpoint goes on the LAST block.

        Returns:
            List of content blocks, or None if no system messages
        """
        if not system_msgs:
            return None

        # Combine into single text (preserves current behavior)
        combined = "\n\n".join(
            m.content if isinstance(m.content, str) else "" for m in system_msgs
        )

        if not combined:
            return None

        block: dict[str, Any] = {"type": "text", "text": combined}

        # Add cache_control if enabled
        if self.enable_prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}

        return [block]

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """

        self._set_session_from_request(request)
        request.messages = self._get_recent_messages(request.messages)

        logger.debug(
            f"Received ChatRequest with {len(request.messages)} messages (debug={self.debug})"
        )

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        logger.debug(
            f"Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Format system messages as content block array (required for caching)
        system_blocks = self._format_system_with_cache(system_msgs)

        if system_blocks:
            logger.info(
                f"[PROVIDER] System message length: {len(system_blocks[0]['text'])} chars (caching={'cache_control' in system_blocks[0]})"
            )
        else:
            logger.info("[PROVIDER] No system messages")

        # Convert developer messages to XML-wrapped user messages (at top)
        context_user_msgs = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(
                f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}"
            )
            logger.debug(f"[PROVIDER] Developer message preview: {content_preview}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "content": wrapped})

        logger.info(
            f"[PROVIDER] Created {len(context_user_msgs)} XML-wrapped context messages"
        )

        # Convert conversation messages
        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        logger.info(
            f"[PROVIDER] Converted {len(conversation_msgs)} conversation messages"
        )

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs
        # Apply cache control to last message for incremental context caching
        all_messages = self._apply_message_cache_control(all_messages)
        logger.info(f"[PROVIDER] Final message count for API: {len(all_messages)}")

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": all_messages,
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature
            or kwargs.get("temperature", self.temperature),
        }

        if system_blocks:
            params["system"] = system_blocks

        # Add tools if provided
        if request.tools:
            tools = self._convert_tools_from_request(request.tools)
            params["tools"] = self._apply_tool_cache_control(tools)
            # Add tool_choice if specified
            if tool_choice := kwargs.get("tool_choice"):
                params["tool_choice"] = tool_choice

        # Add native web search tool if enabled (via config or kwargs)
        # This is a model-native tool that doesn't need function conversion
        web_search_enabled = kwargs.get("enable_web_search", self.enable_web_search)
        if web_search_enabled:
            web_search_tool = self._build_web_search_tool(kwargs)
            if "tools" not in params:
                params["tools"] = []
            # Add web search tool at the beginning (native tools typically come first)
            params["tools"].insert(0, web_search_tool)
            logger.info("[PROVIDER] Native web search tool enabled")

        # Enable extended thinking if requested (equivalent to OpenAI's reasoning)
        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        interleaved_thinking_enabled = False
        if thinking_enabled:
            budget_tokens = (
                kwargs.get("thinking_budget_tokens")
                or self.config.get("thinking_budget_tokens")
                or 32000
            )
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 4096
            )

            thinking_budget = budget_tokens
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

            # CRITICAL: Anthropic requires temperature=1.0 when thinking is enabled
            params["temperature"] = 1.0

            # Ensure max_tokens accommodates thinking budget + response
            target_tokens = budget_tokens + buffer_tokens
            if params.get("max_tokens"):
                params["max_tokens"] = max(params["max_tokens"], target_tokens)
            else:
                params["max_tokens"] = target_tokens

            # iinterleaved thinking not available via cli
            interleaved_thinking_enabled = False

            logger.info(
                "[PROVIDER] Extended thinking enabled (budget=%s, buffer=%s, temperature=1.0, max_tokens=%s, interleaved=%s)",
                thinking_budget,
                buffer_tokens,
                params["max_tokens"],
                interleaved_thinking_enabled,
            )

        # Add stop_sequences if specified
        if stop_sequences := kwargs.get("stop_sequences"):
            params["stop_sequences"] = stop_sequences

        logger.info(
            f"[PROVIDER] Anthropic API call - model: {params['model']}, messages: {len(params['messages'])}, system: {bool(system_blocks)}, tools: {len(params.get('tools', []))}, thinking: {thinking_enabled}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "anthropic",
                    "model": params["model"],
                    "message_count": len(params["messages"]),
                    "has_system": bool(system_blocks),
                    "thinking_enabled": thinking_enabled,
                    "thinking_budget": thinking_budget,
                    "interleaved_thinking": interleaved_thinking_enabled,
                },
            )

            # DEBUG level: Full request payload with truncated values (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "anthropic",
                        "request": self._truncate_values(params),
                    },
                )

            # RAW level: Complete params dict as sent to Anthropic API (if debug AND raw_debug enabled)
            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "anthropic",
                        "params": params,  # Complete untruncated params
                    },
                )

        start_time = time.time()

        async with asyncio.timeout(self.timeout):
            async with ClaudeSDKClient(
                options=ClaudeAgentOptions(
                    tools=[],  # disable built-in tools
                    model=self.default_model,
                    resume=self._session.id,
                    system_prompt="---",  # system prompt is passed in messages
                    max_thinking_tokens=self.max_thinking_tokens,
                    betas=self._beta_headers,
                )
            ) as client:
                prompt = self._convert_prompt_from_request_params(params)
                await client.query(prompt, session_id=self._session.id)
                response = await self._parse_response(client)

        # If we get here, request succeeded - continue with response handling
        try:
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Anthropic API")
            logger.debug(f"[PROVIDER] Response type: {response.model}")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary with rate limit info
                response_event: dict[str, Any] = {
                    "provider": "anthropic",
                    "model": params["model"],
                    "usage": {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                        **(
                            {"cache_read": response.usage.cache_read_input_tokens}
                            if hasattr(response.usage, "cache_read_input_tokens")
                            and response.usage.cache_read_input_tokens
                            else {}
                        ),
                        **(
                            {"cache_write": response.usage.cache_creation_input_tokens}
                            if hasattr(response.usage, "cache_creation_input_tokens")
                            and response.usage.cache_creation_input_tokens
                            else {}
                        ),
                    },
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }

                await self.coordinator.hooks.emit("llm:response", response_event)

                # DEBUG level: Full response with truncated values (if debug enabled)
                if self.debug:
                    response_dict = response.model_dump()  # Pydantic model → dict
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "anthropic",
                            "response": self._truncate_values(response_dict),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Complete response object from Anthropic API (if debug AND raw_debug enabled)
                if self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "anthropic",
                            "response": response.model_dump(),  # Complete untruncated response
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except TimeoutError:
            # Handle timeout specifically - TimeoutError has empty str() representation
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Request timed out after {self.timeout}s"
            logger.error(f"[PROVIDER] Anthropic API error: {error_msg}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise TimeoutError(error_msg) from None

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Ensure error message is never empty
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error(f"[PROVIDER] Anthropic API error: {error_msg}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            # Re-raise with meaningful message if original was empty
            if not str(e):
                raise type(e)(error_msg) from e
            raise

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

    def _clean_content_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Clean a content block for API by removing fields not accepted by Anthropic API.

        Anthropic API may include extra fields (like 'visibility') in responses,
        but does NOT accept these fields when blocks are sent as input in messages.

        Args:
            block: Raw content block dict (may include visibility, etc.)

        Returns:
            Cleaned content block dict with only API-accepted fields
        """
        block_type = block.get("type")

        if block_type == "text":
            return {"type": "text", "text": block.get("text", "")}
        if block_type == "thinking":
            cleaned = {"type": "thinking", "thinking": block.get("thinking", "")}
            if "signature" in block:
                cleaned["signature"] = block["signature"]
            return cleaned
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
        if block_type == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id", ""),
                "content": block.get("content", ""),
            }
        if block_type == "web_search_tool_result":
            # Web search results are model-native and should be passed through
            # with minimal cleaning (just remove internal fields)
            cleaned: dict[str, Any] = {
                "type": "web_search_tool_result",
            }
            if "tool_use_id" in block:
                cleaned["tool_use_id"] = block["tool_use_id"]
            if "content" in block:
                cleaned["content"] = block["content"]
            return cleaned
        # Unknown block type - return as-is but remove visibility
        cleaned = dict(block)
        cleaned.pop("visibility", None)
        return cleaned

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format.

        CRITICAL: Anthropic requires ALL tool_result blocks from one assistant's tool_use
        to be batched into a SINGLE user message with multiple tool_result blocks in the
        content array. We cannot send separate user messages for each tool result.

        This method batches consecutive tool messages into one user message.

        DEFENSIVE: Also validates that each tool_result has a corresponding tool_use
        in a preceding assistant message. Orphaned tool_results (from context compaction)
        are skipped to avoid API errors.
        """
        # First pass: collect all valid tool_use_ids from assistant messages
        valid_tool_use_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id") or tc.get("tool_call_id")
                    if tc_id:
                        valid_tool_use_ids.add(tc_id)

        anthropic_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                i += 1
                continue

            # Batch consecutive tool messages into ONE user message
            if role == "tool":
                # Collect all consecutive tool results, but only valid ones
                tool_results = []
                skipped_count = 0
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_use_id = tool_msg.get("tool_call_id")

                    # DEFENSIVE: Skip tool_results without valid tool_use_id
                    # This prevents API errors from orphaned tool_results after compaction
                    if not tool_use_id or tool_use_id not in valid_tool_use_ids:
                        logger.warning(
                            f"Skipping orphaned tool_result (no matching tool_use): "
                            f"tool_call_id={tool_use_id}, content_preview={str(tool_msg.get('content', ''))[:100]}"
                        )
                        skipped_count += 1
                        i += 1
                        continue

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_msg.get("content", ""),
                        }
                    )
                    i += 1

                # Only add user message if we have valid tool_results
                if tool_results:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": tool_results,  # Array of tool_result blocks
                        }
                    )
                elif skipped_count > 0:
                    logger.warning(
                        f"All {skipped_count} consecutive tool_results were orphaned and skipped"
                    )
                continue  # i already advanced in while loop
            if role == "assistant":
                # Assistant messages - check for tool calls or thinking blocks
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content_blocks = []

                    # CRITICAL: Check for thinking block and add it FIRST
                    has_thinking = "thinking_block" in msg and msg["thinking_block"]
                    if has_thinking:
                        # Clean thinking block (remove visibility field not accepted by API)
                        cleaned_thinking = self._clean_content_block(
                            msg["thinking_block"]
                        )
                        content_blocks.append(cleaned_thinking)

                    # Add text content if present, BUT skip when we have thinking + tool_calls
                    # When all three are present (thinking + text + tool_use), the text was generated
                    # but not shown to user yet (tool calls execute first). Including it in history
                    # misleads the model into thinking it already communicated that info.
                    if content and not has_thinking:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    content_blocks.append(
                                        {"type": "text", "text": block.get("text", "")}
                                    )
                                elif (
                                    not isinstance(block, dict)
                                    and hasattr(block, "type")
                                    and block.type == "text"
                                ):
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": getattr(block, "text", ""),
                                        }
                                    )
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": content})

                    # Add tool_use blocks
                    for tc in msg["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("tool", ""),
                                "input": tc.get("arguments", {}),
                            }
                        )

                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                elif "thinking_block" in msg and msg["thinking_block"]:
                    # Assistant message with thinking block
                    # Clean thinking block (remove visibility field not accepted by API)
                    cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                    content_blocks = [cleaned_thinking]
                    if content:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    content_blocks.append(
                                        {"type": "text", "text": block.get("text", "")}
                                    )
                                elif (
                                    not isinstance(block, dict)
                                    and hasattr(block, "type")
                                    and block.type == "text"
                                ):
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": getattr(block, "text", ""),
                                        }
                                    )
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": content})
                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    # Regular assistant message - may have structured content blocks
                    if isinstance(content, list):
                        # Content is a list of blocks - clean each block
                        cleaned_blocks = [
                            self._clean_content_block(block) for block in content
                        ]
                        anthropic_messages.append(
                            {"role": "assistant", "content": cleaned_blocks}
                        )
                    else:
                        # Content is a simple string
                        anthropic_messages.append(
                            {"role": "assistant", "content": content}
                        )
                i += 1
            elif role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                wrapped = f"<context_file>\n{content}\n</context_file>"
                anthropic_messages.append({"role": "user", "content": wrapped})
                i += 1
            else:
                # User messages - handle structured content (text + images)
                if isinstance(content, list):
                    content_blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                content_blocks.append(
                                    {"type": "text", "text": block.get("text", "")}
                                )
                            elif block_type == "image":
                                # Convert ImageBlock to Anthropic image format
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    content_blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": source.get(
                                                    "media_type", "image/jpeg"
                                                ),
                                                "data": source.get("data"),
                                            },
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Unsupported image source type: {source.get('type')}"
                                    )

                    if content_blocks:
                        anthropic_messages.append(
                            {"role": "user", "content": content_blocks}
                        )
                else:
                    # Simple string content
                    anthropic_messages.append({"role": "user", "content": content})
                i += 1

        return anthropic_messages

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to Anthropic format.

        Handles both standard function tools (converted to Anthropic format) and
        model-native tools like web_search_20250305 (passed through unchanged).

        Model-native tools are identified by having a 'type' attribute that is NOT
        'function'. These tools use Anthropic's built-in capabilities and should
        NOT be converted to the standard function tool format.

        Args:
            tools: List of ToolSpec objects or native tool definitions

        Returns:
            List of Anthropic-formatted tool definitions
        """
        anthropic_tools = []
        for tool in tools:
            # Check if this is a model-native tool (has 'type' that's not 'function')
            # Native tools like web_search_20250305 are passed through unchanged
            tool_type = getattr(tool, "type", None)
            if tool_type and tool_type != "function":
                # Model-native tool - pass through as-is (converted to dict if needed)
                if hasattr(tool, "model_dump"):
                    anthropic_tools.append(tool.model_dump(exclude_none=True))
                elif isinstance(tool, dict):
                    anthropic_tools.append(tool)
                else:
                    # Fallback: build dict from known attributes
                    native_tool: dict[str, Any] = {"type": tool_type}
                    if hasattr(tool, "name") and tool.name:
                        native_tool["name"] = tool.name
                    # Add any additional config (e.g., max_uses for web search)
                    if hasattr(tool, "max_uses") and tool.max_uses is not None:
                        native_tool["max_uses"] = tool.max_uses
                    if (
                        hasattr(tool, "user_location")
                        and tool.user_location is not None
                    ):
                        native_tool["user_location"] = tool.user_location
                    anthropic_tools.append(native_tool)
                logger.debug(f"[PROVIDER] Added native tool: {tool_type}")
            else:
                # Standard function tool - convert to Anthropic format
                anthropic_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.parameters,
                    }
                )
        return anthropic_tools

    def _extract_web_search_citations(self, block: Any) -> list[dict[str, Any]]:
        """Extract citation information from a web search result block.

        Web search results contain citations with source information that can be
        displayed to users for transparency and attribution.

        Args:
            block: Web search tool result block from Anthropic response

        Returns:
            List of citation dicts with title, url, and optional snippet
        """
        citations = []

        # Web search results have a 'content' field with search results
        content = getattr(block, "content", None)
        if not content:
            return citations

        # Content may be a list of result items or a single object
        results = content if isinstance(content, list) else [content]

        for result in results:
            # Each result may have source information
            if hasattr(result, "type") and result.type == "web_search_result":
                citation: dict[str, Any] = {}

                # Extract URL (required)
                if hasattr(result, "url") and result.url:
                    citation["url"] = result.url
                elif hasattr(result, "source_url") and result.source_url:
                    citation["url"] = result.source_url

                # Extract title
                if hasattr(result, "title") and result.title:
                    citation["title"] = result.title

                # Extract snippet/description
                if hasattr(result, "snippet") and result.snippet:
                    citation["snippet"] = result.snippet
                elif hasattr(result, "description") and result.description:
                    citation["snippet"] = result.description
                elif hasattr(result, "encrypted_content") and result.encrypted_content:
                    # Some results use encrypted_content - just note it exists
                    citation["has_content"] = True

                # Only add if we have at least a URL
                if citation.get("url"):
                    citations.append(citation)

        return citations

    def _build_web_search_tool(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the native web search tool definition.

        The web_search_20250305 tool is a model-native tool that enables Claude
        to search the web for current information. Unlike function tools, it uses
        Anthropic's built-in web search capability.

        Tool definition format:
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,  # optional, limits searches per request
                "user_location": {...}  # optional, for location-aware results
            }

        Args:
            kwargs: Request kwargs that may contain web search configuration

        Returns:
            Web search tool definition dict
        """
        tool: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",  # Anthropic requires this exact name
        }

        # Optional: max_uses limits number of searches per request
        max_uses = kwargs.get("web_search_max_uses") or self.config.get(
            "web_search_max_uses"
        )
        if max_uses is not None:
            tool["max_uses"] = max_uses

        # Optional: user_location for location-aware search results
        user_location = kwargs.get("web_search_user_location") or self.config.get(
            "web_search_user_location"
        )
        if user_location is not None:
            tool["user_location"] = user_location

        return tool

    def _apply_tool_cache_control(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add cache_control to the last tool definition.

        Per Anthropic spec: cache breakpoint on last tool creates
        checkpoint for entire tool list.

        Args:
            tools: List of Anthropic-formatted tool definitions

        Returns:
            Same list with cache_control on last tool (if caching enabled)
        """
        if not tools or not self.enable_prompt_caching:
            return tools

        # Add cache_control to last tool
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def _apply_message_cache_control(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add cache_control to last content block of last message.

        Per Anthropic spec: this creates a checkpoint at the end of
        conversation history, caching the full context.

        Args:
            messages: Anthropic-formatted message list

        Returns:
            Same list with cache_control on last message's last block
        """
        if not messages or not self.enable_prompt_caching:
            return messages

        last_msg = messages[-1]
        content = last_msg.get("content")

        # Handle different content formats
        if isinstance(content, list) and content:
            # Array of content blocks - mark last block
            content[-1]["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str):
            # String content - convert to block array with cache marker
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return messages

    def _convert_to_chat_response(self, response: ParsedMessage) -> ChatResponse:
        """Convert Anthropic response to ChatResponse format.

        Args:
            response: Anthropic API response

        Returns:
            AnthropicChatResponse with content blocks and streaming-compatible fields
        """

        content_blocks = []
        tool_calls = []
        web_search_results: list[dict[str, Any]] = []
        event_blocks: list[
            TextContent | ThinkingContent | ToolCallContent | WebSearchContent | Session
        ] = []
        text_accumulator: list[str] = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
                text_accumulator.append(block.text)
                event_blocks.append(TextContent(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(
                    ThinkingBlock(
                        thinking=block.thinking,
                        signature=getattr(block, "signature", None),
                        visibility="internal",
                    )
                )
                event_blocks.append(ThinkingContent(text=block.thinking))
                # NOTE: Do NOT add thinking to text_accumulator - it's internal process, not response content
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolCallBlock(id=block.id, name=block.name, input=block.input)
                )
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
                event_blocks.append(
                    ToolCallContent(id=block.id, name=block.name, arguments=block.input)
                )
            elif block.type == "web_search_tool_result":
                # Handle native web search results from Anthropic
                # Extract citations from search results for observability
                citations = self._extract_web_search_citations(block)
                web_search_results.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", None),
                        "citations": citations,
                    }
                )
                # Add to event blocks for UI display
                event_blocks.append(
                    WebSearchContent(
                        query=getattr(block, "query", ""),
                        citations=citations,
                    )
                )
                logger.debug(
                    f"[PROVIDER] Web search returned {len(citations)} citations"
                )

        # Build usage dict with cache metrics if available
        usage_kwargs: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Add cache metrics if available (Anthropic includes these when caching is active)
        if (
            hasattr(response.usage, "cache_creation_input_tokens")
            and response.usage.cache_creation_input_tokens
        ):
            usage_kwargs["cache_creation_input_tokens"] = (
                response.usage.cache_creation_input_tokens
            )
        if (
            hasattr(response.usage, "cache_read_input_tokens")
            and response.usage.cache_read_input_tokens
        ):
            usage_kwargs["cache_read_input_tokens"] = (
                response.usage.cache_read_input_tokens
            )

        usage = Usage(**usage_kwargs)
        combined_text = "\n\n".join(text_accumulator).strip()
        content_blocks.append(self._session)

        return ClaudeChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
            web_search_results=web_search_results if web_search_results else None,
        )

    def _convert_prompt_from_request_params(
        self, params: dict[str, list[dict[str, str | list[dict[str, Any]]]]]
    ) -> str:
        # only on first prompt
        system: list[str] = []
        tools: list[str] = []

        # at each turn
        system_reminders: list[str] = []
        user_messages: list[str] = []
        tool_results: list[str] = []

        for message in params.get("system", []):
            system.append(message["text"])

        for message in params.get("tools", []):
            tools.append(
                f"""{{"name": "{message["name"]}", "input_schema": {json.dumps(message["input_schema"])}}}\n<instructions>\n{message["description"]}\n</instructions>"""
            )

        for message in params.get("messages", []):
            match message["role"]:
                case "user":
                    match message["content"]:
                        case str():
                            user_messages.append(f"[user]: {message['content']}")

                        case list():
                            for block in message["content"]:
                                match block["type"]:
                                    case "tool_result":
                                        try:
                                            output = json.loads(block["content"])
                                        except json.JSONDecodeError:
                                            output = block["content"]

                                        tool_json = json.dumps(
                                            {
                                                "id": block["tool_use_id"],
                                                "output": output,
                                            }
                                        )
                                        tool_results.append(f"[tool]: {tool_json}")
                                    case "text":
                                        system_reminders.append(f"""{block["text"]}""")

                                    case _:
                                        raise NotImplementedError(
                                            f"[PROVIDER] Unknown block type for user message: {block['type']}"
                                        )
                        case _:
                            raise NotImplementedError(
                                f"[PROVIDER] Unknown content type for user message: {type(message['content'])}"
                            )

                case "assistant":  # assistant blocks not included in prompt
                    match message["content"]:
                        case list():
                            for block in message["content"]:
                                match block["type"]:
                                    case (
                                        "thinking"
                                        | "tool_use"
                                        | "text"
                                        | "redacted_thinking"
                                    ):
                                        pass

                                    case _:
                                        raise NotImplementedError(
                                            f"[PROVIDER] Unknown block type for assistant message: {block['type']}"
                                        )
                        case _:
                            raise NotImplementedError(
                                f"[PROVIDER] Unknown content type for assistant message: {type(message['content'])}"
                            )
                case _:
                    raise NotImplementedError(
                        f"[PROVIDER] Unknown message role: {message['role']}"
                    )

        prompt = ""

        if not self._session.id:  # only include system/tools on first prompt
            prompt += "<system>\n" + "\n\n".join(system) + "\n</system>\n\n"
            prompt += "<tools>\n" + "\n\n".join(tools) + "\n</tools>\n\n"

        if tool_results:
            prompt += "\n".join(tool_results) + "\n\n"
            prompt += f"""<system-reminder source="hooks-interleaved-thinking">\n{INTERLEAVED_THINKING_REMINDER}\n</system-reminder>\n\n"""

        if user_messages:
            prompt += "\n".join(user_messages) + "\n\n"

        if system_reminders:
            prompt += "\n".join(system_reminders) + "\n\n"
            prompt += f"""<system-reminder source="hooks-tools-reminder">\n{TOOL_USE_REMINDER}</system-reminder>"""

        return prompt

    def _parse_tool_blocks_from_text(self, text: str) -> list[AnthropicToolUseBlock]:
        """Parse [tool]: {...} blocks from response text."""

        tool_blocks: list[AnthropicToolUseBlock] = []

        # 1. Match Fenced Code (``` ... ```)
        # 2. Match Inline Code (` ... `) - assumes no newlines inside inline code
        # 3. Match Tool Start ([tool]:)
        tokenizer = re.compile(r"(```(?:.|\n)*?```)|(`[^`\n]*`)|(\[tool\]:)", re.DOTALL)

        pos = 0
        while pos < len(text):
            match = tokenizer.search(text, pos)
            if not match:
                break

            start, end = match.span()
            if match.group(1):
                # Case 1: Found a ``` fenced block.
                # IGNORE IT. Move past it.
                pos = end

            elif match.group(2):
                # Case 2: Found an inline ` code block.
                # IGNORE IT. Move past it.
                pos = end

            elif match.group(3):
                # Case 3: Found "[tool]:". This is in "safe" text.
                json_start = end

                while json_start < len(text) and text[json_start].isspace():
                    json_start += 1

                try:
                    decoder = json.JSONDecoder()
                    obj, end_offset = decoder.raw_decode(text, idx=json_start)

                    tool_blocks.append(AnthropicToolUseBlock.model_validate(obj))
                    pos = end_offset

                except json.JSONDecodeError | ValidationError:
                    pos = end

        return tool_blocks

    async def _parse_response(self, client: ClaudeSDKClient) -> ParsedMessage:
        """Parse messages from Claude Agent SDK into an Anthropic ParsedMessage."""

        model: str = ""
        content: list[ParsedContentBlock] = []
        usage = AnthropicUsage(input_tokens=0, output_tokens=0)

        async for message in client.receive_response():
            match message:
                case claude_agent_sdk.types.AssistantMessage():
                    model = message.model
                    for block in message.content:
                        match block:
                            case claude_agent_sdk.types.TextBlock():
                                tool_blocks = self._parse_tool_blocks_from_text(
                                    block.text
                                )
                                if tool_blocks:
                                    content.extend(tool_blocks)
                                if block.text != "(no content)":
                                    content.append(
                                        ParsedTextBlock(
                                            type="text",
                                            text=block.text,
                                        )
                                    )

                            case claude_agent_sdk.types.ThinkingBlock():
                                content.append(
                                    AnthropicThinkingBlock(
                                        type="thinking",
                                        thinking=block.thinking,
                                        signature=block.signature,
                                    )
                                )
                            case _:
                                raise NotImplementedError(
                                    f"[PROVIDER] AssistantMessage content block type from SDK: {type(block)}"
                                )

                case claude_agent_sdk.types.ResultMessage():
                    self._session.id = (
                        message.session_id
                    )  # update session - enables resume
                    if message.usage:
                        usage.input_tokens = message.usage.get(
                            "input_tokens",
                            0,
                        )
                        usage.output_tokens = message.usage.get(
                            "output_tokens",
                            0,
                        )
                        usage.cache_read_input_tokens = message.usage.get(
                            "cache_read_input_tokens",
                            None,
                        )
                        usage.cache_creation_input_tokens = message.usage.get(
                            "cache_creation_input_tokens",
                            None,
                        )

                    if message.is_error:
                        logger.warning(
                            f"[PROVIDER] SDK response indicates error: {message.result}"
                        )

                case claude_agent_sdk.types.SystemMessage():
                    logger.debug(
                        f"[PROVIDER] SDK {message.subtype} system message:{message.data}"
                    )

                case _:
                    raise NotImplementedError(
                        f"[PROVIDER] Unknown message type from SDK: {type(message)}"
                    )

        stop_reason = (
            "tool_use"
            if any(isinstance(block, AnthropicToolUseBlock) for block in content)
            else "end_turn"
        )

        return ParsedMessage(
            id=self._session.id,
            content=content,
            model=model,
            role="assistant",
            type="message",
            usage=usage,
            stop_reason=stop_reason,
        )

    def _set_session_from_request(self, request: ChatRequest):
        if not self._session.id:
            for message in request.messages:
                if message.role == "assistant":
                    for block in message.content:
                        if block.type == "redacted_thinking" and block.data.startswith(
                            SESSION_TAG
                        ):
                            self._session.data = block.data


TOOL_USE_EXAMPLE1 = json.dumps(
    {
        "type": "tool_use",
        "name": "first_tool",
        "id": "tl4xcu5",
        "input": {"param1": "value1", "param2": 42},
    }
)

TOOL_USE_EXAMPLE2 = json.dumps(
    {
        "type": "tool_use",
        "name": "second_tool",
        "id": "tl4t214",
        "input": {"param1": "valueX"},
    }
)

TOOL_USE_REMINDER = f"""You have access to the all the tools defined with the <tools> XML block.
To call tools respond with tool blocks with a valid JSON with "name", "id", and "input" fields.
In the example below two tools are being called in parallel.
<example>
[tool]: {TOOL_USE_EXAMPLE1}
[tool]: {TOOL_USE_EXAMPLE2}
</example>
<instructions>
Usage:
- The response must ONLY contain tool blocks. No additional text.
- Generate a 7 character high-entropy id for each tool block
- The "input" field must respect the "input_schema" in the tool definitions
- Wait for the next turn for the tool results
</instructions>
"""

INTERLEAVED_THINKING_REMINDER = """Before proceeding, briefly reflect on what you just learned from the tool result
- What does this tell you about the problem?
- Does it change your approach or next steps?
- Are there any unexpected findings to consider?

Think through this internally, then continue with your work."""
