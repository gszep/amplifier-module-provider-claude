"""
Anthropic provider module for Amplifier.
Integrates with Anthropic's Claude API.
"""

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

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> ProviderResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation history
            **kwargs: Additional parameters

        Returns:
            Provider response
        """
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
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "anthropic",
                    "model": params["model"],
                    "messages": len(anthropic_messages),  # Count only, not content (privacy)
                    "thinking_enabled": params.get("thinking") is not None,
                },
            )

        start_time = time.time()
        try:
            response = await self.client.messages.create(**params)
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
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                        "usage": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                        "has_thinking": bool(thinking_text),
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
