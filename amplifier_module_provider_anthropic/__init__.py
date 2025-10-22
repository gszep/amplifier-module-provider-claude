"""
Anthropic provider module for Amplifier.
Integrates with Anthropic's Claude API.
"""

import asyncio
import logging
import os
import time
from typing import Any
from typing import Optional

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Anthropic provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including API key

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.warning("No API key found for Anthropic provider")
        return None

    provider = AnthropicProvider(api_key, config, coordinator)
    await coordinator.mount("providers", provider, name="anthropic")
    logger.info("Mounted AnthropicProvider")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class AnthropicProvider:
    """Anthropic API integration."""

    name = "anthropic"

    def __init__(
        self, api_key: str, config: dict[str, Any] | None = None, coordinator: ModuleCoordinator | None = None
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "claude-sonnet-4-5")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.priority = self.config.get("priority", 100)  # Store priority for selection
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.timeout = self.config.get("timeout", 300.0)  # API timeout in seconds (default 5 minutes)

    async def complete(self, messages: list[dict[str, Any]] | ChatRequest, **kwargs) -> ProviderResponse | ChatResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation history (list of dicts or ChatRequest)
            **kwargs: Additional parameters

        Returns:
            Provider response or ChatResponse
        """
        # Handle ChatRequest format
        if isinstance(messages, ChatRequest):
            return await self._complete_chat_request(messages, **kwargs)

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages(messages)

        # Extract system message if present
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system:
            params["system"] = system

        # Support extended thinking (Anthropic API format)
        if kwargs.get("extended_thinking"):
            # Anthropic expects thinking={type: "enabled", budget_tokens: N}
            budget_tokens = kwargs.get("thinking_budget_tokens", 10000)
            params["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            # When thinking is enabled, temperature must be 1
            params["temperature"] = 1.0
            # max_tokens must be greater than budget_tokens (add buffer for actual response)
            if params["max_tokens"] <= budget_tokens:
                params["max_tokens"] = budget_tokens + 4096  # Budget + response tokens
            logger.info(
                f"Extended thinking enabled with budget: {budget_tokens} tokens (temperature=1.0, max_tokens={params['max_tokens']})"
            )

        # Add tools if provided
        if "tools" in kwargs:
            params["tools"] = self._convert_tools(kwargs["tools"])

        logger.info(f"Anthropic API call - model: {params['model']}, thinking: {params.get('thinking')}")

        # Emit llm:request event if coordinator is available
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "anthropic",
                        "model": params["model"],
                        "message_count": len(anthropic_messages),
                        "thinking_enabled": params.get("thinking") is not None,
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "anthropic",
                            "request": {
                                "model": params["model"],
                                "messages": anthropic_messages,
                                "system": system,
                                "max_tokens": params["max_tokens"],
                                "temperature": params["temperature"],
                                "thinking": params.get("thinking"),
                            },
                        },
                    },
                )

        start_time = time.time()
        try:
            response = await asyncio.wait_for(self.client.messages.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Anthropic API response received - content blocks: {len(response.content)}")

            # Convert response to standard format
            content = ""
            tool_calls = []
            content_blocks = []
            thinking_text = ""

            for block in response.content:
                logger.info(f"Processing block type: {block.type}")
                if block.type == "text":
                    content = block.text
                    content_blocks.append(TextContent(text=block.text, raw=block))
                elif block.type == "thinking":
                    logger.info(f"Found thinking block with {len(block.thinking)} chars")
                    thinking_text = block.thinking
                    content_blocks.append(ThinkingContent(text=block.thinking, raw=block))

                    # Emit thinking:final event for the complete thinking block
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit("thinking:final", {"text": block.thinking})
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(tool=block.name, arguments=block.input, id=block.id))
                    content_blocks.append(
                        ToolCallContent(id=block.id, name=block.name, arguments=block.input, raw=block)
                    )

            # Emit llm:response event with success
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "anthropic",
                            "model": params["model"],
                            "usage": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                            "has_thinking": bool(thinking_text),
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "anthropic",
                                "response": {
                                    "content": content,
                                    "thinking": thinking_text[:500] + "..."
                                    if len(thinking_text) > 500
                                    else thinking_text,
                                    "tool_calls": [{"tool": tc.tool, "id": tc.id} for tc in tool_calls]
                                    if tool_calls
                                    else [],
                                    "stop_reason": response.stop_reason,
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            return ProviderResponse(
                content=content,
                raw=response,
                usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                tool_calls=tool_calls if tool_calls else None,
                content_blocks=content_blocks if content_blocks else None,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")

            # Emit llm:response event with error
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params.get("model", self.default_model),
                        "status": "error",
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "error": str(e),
                    },
                )

            raise

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        print(f"\n{'=' * 80}\n_complete_chat_request() CALLED | debug={self.debug}\n{'=' * 80}")
        logger.info(f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages")
        logger.info(f"[PROVIDER] Message roles: {[m.role for m in request.messages]}")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant")]

        logger.info(
            f"[PROVIDER] Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages
        system = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
        )

        if system:
            logger.info(f"[PROVIDER] Combined system message length: {len(system)}")
        else:
            logger.info("[PROVIDER] No system messages")

        # Convert developer messages to XML-wrapped user messages (at top)
        context_user_msgs = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}")
            logger.debug(f"[PROVIDER] Developer message preview: {content_preview}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "content": wrapped})

        logger.info(f"[PROVIDER] Created {len(context_user_msgs)} XML-wrapped context messages")

        # Convert conversation messages
        conversation_msgs = self._convert_messages([m.model_dump() for m in conversation])
        logger.info(f"[PROVIDER] Converted {len(conversation_msgs)} conversation messages")

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs
        logger.info(f"[PROVIDER] Final message count for API: {len(all_messages)}")

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": all_messages,
            "max_tokens": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature or kwargs.get("temperature", self.temperature),
        }

        if system:
            params["system"] = system

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)

        logger.info(
            f"[PROVIDER] Anthropic API call - model: {params['model']}, messages: {len(params['messages'])}, system: {bool(system)}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "anthropic",
                        "model": params["model"],
                        "message_count": len(params["messages"]),
                        "has_system": bool(system),
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "anthropic",
                            "request": {
                                "model": params["model"],
                                "messages": params["messages"],
                                "system": system,
                                "max_tokens": params["max_tokens"],
                                "temperature": params["temperature"],
                            },
                        },
                    },
                )

        start_time = time.time()

        # Call Anthropic API
        try:
            response = await asyncio.wait_for(self.client.messages.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Anthropic API")
            logger.debug(f"[PROVIDER] Response type: {response.model}")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "anthropic",
                            "model": params["model"],
                            "usage": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    content_preview = str(response.content)[:500] if response.content else ""
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "anthropic",
                                "response": {
                                    "content_preview": content_preview,
                                    "stop_reason": response.stop_reason,
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Anthropic API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Filters out tool calls with empty/missing arguments to handle
        Anthropic API quirk where empty tool_use blocks are sometimes generated.

        Args:
            response: Provider response

        Returns:
            List of valid tool calls (with non-empty arguments)
        """
        if not response.tool_calls:
            return []

        # Filter out tool calls with empty arguments (Anthropic API quirk)
        # Claude sometimes generates tool_use blocks with empty input {}
        valid_calls = []
        for tc in response.tool_calls:
            # Skip tool calls with no arguments or empty dict
            if not tc.arguments:
                logger.debug(f"Filtering out tool '{tc.tool}' with empty arguments")
                continue
            valid_calls.append(tc)

        if len(valid_calls) < len(response.tool_calls):
            logger.info(f"Filtered {len(response.tool_calls) - len(valid_calls)} tool calls with empty arguments")

        return valid_calls

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format."""
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Convert role names
            if role == "tool":
                # Tool results in Anthropic format
                tool_use_id = msg.get("tool_call_id")
                if not tool_use_id:
                    logger.warning(f"Tool result missing tool_call_id: {msg}")
                    tool_use_id = "unknown"  # Fallback, but will likely fail

                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                            }
                        ],
                    }
                )
            elif role == "assistant":
                # Assistant messages - check for tool calls or thinking blocks
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content_blocks = []

                    # CRITICAL: Check for thinking block and add it FIRST
                    if "thinking_block" in msg and msg["thinking_block"]:
                        # Use the raw thinking block which includes signature
                        content_blocks.append(msg["thinking_block"])

                    # Add text content if present
                    if content:
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

                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                elif "thinking_block" in msg and msg["thinking_block"]:
                    # Assistant message with thinking block
                    content_blocks = [msg["thinking_block"]]
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    # Regular assistant message
                    anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                wrapped = f"<context_file>\n{content}\n</context_file>"
                anthropic_messages.append({"role": "user", "content": wrapped})
            else:
                # User messages
                anthropic_messages.append({"role": "user", "content": content})

        return anthropic_messages

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            # Get schema from tool if available, otherwise use empty schema
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": input_schema,
                }
            )

        return anthropic_tools

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to Anthropic format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Anthropic-formatted tool definitions
        """
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.parameters,
                }
            )
        return anthropic_tools

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert Anthropic response to ChatResponse format.

        Args:
            response: Anthropic API response

        Returns:
            ChatResponse with content blocks
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(
                    ThinkingBlock(thinking=block.thinking, signature=getattr(block, "signature", None))
                )
            elif block.type == "tool_use":
                content_blocks.append(ToolCallBlock(id=block.id, name=block.name, input=block.input))
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
        )
