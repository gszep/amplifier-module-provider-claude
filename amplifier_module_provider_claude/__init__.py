"""Claude provider module for Amplifier.

Full Control implementation using Claude Code CLI.
Amplifier's orchestrator handles tool execution - Claude only decides which tools to call.
"""

__all__ = ["mount", "ClaudeProvider"]

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
)
from amplifier_core.content_models import TextContent  # type: ignore
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
    ToolCall,
    ToolCallBlock,
    Usage,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

METADATA_SESSION_ID = "claude:session_id"
METADATA_COST_USD = "claude:cost_usd"
METADATA_DURATION_MS = "claude:duration_ms"

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_TOKENS = 64000

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
        "capabilities": ["tools", "streaming"],
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
    api_label = "Claude (Claude Code)"

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

    def get_info(self) -> ProviderInfo:
        """Return provider information.

        Returns:
            ProviderInfo with capabilities and configuration fields.
        """
        return ProviderInfo(
            id="claude",
            display_name="Claude (Claude Code)",
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

        Args:
            response: The ChatResponse to extract tool calls from.

        Returns:
            List of ToolCall objects.
        """
        return response.tool_calls or []

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
        request_metadata = getattr(request, "metadata", None) or {}
        existing_session_id = request_metadata.get(METADATA_SESSION_ID)

        # Convert messages to CLI format
        system_prompt, user_prompt = self._convert_messages(
            request.messages, request.tools
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
            full_prompt = f"{system_prompt}\n\n---\n\nUser request:\n{user_prompt}"
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

        duration = time.time() - start_time

        # Build ChatResponse
        chat_response = self._build_response(response_data, duration)

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
        self, messages: list[Message], tools: list[Any] | None
    ) -> tuple[str, str]:
        """Convert Amplifier messages to Claude CLI format.

        Args:
            messages: List of Amplifier Message objects.
            tools: List of tool specifications.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_parts = []
        conversation_parts = []

        # Build tool definitions for system prompt
        if tools:
            tool_definitions = self._convert_tools(tools)
            system_parts.append(self._build_tool_instructions(tool_definitions))

        # Process messages
        for msg in messages:
            role = msg.role
            content = self._extract_content(msg)

            if role == "system":
                system_parts.append(content)

            elif role == "user":
                conversation_parts.append(f"Human: {content}")

            elif role == "assistant":
                # Check for tool calls in assistant message
                assistant_content = self._format_assistant_message(msg)
                conversation_parts.append(f"Assistant: {assistant_content}")

            elif role == "tool":
                # Tool result - format for Claude
                tool_result = self._format_tool_result(msg)
                conversation_parts.append(f"Human: {tool_result}")

        # Build final prompts
        system_prompt = "\n\n".join(system_parts) if system_parts else ""

        # The user prompt is the conversation history
        # For multi-turn, we include the full conversation
        user_prompt = "\n\n".join(conversation_parts) if conversation_parts else ""

        # If there's only one user message and no conversation history, simplify
        if len(messages) == 1 and messages[0].role == "user":
            user_prompt = self._extract_content(messages[0])

        return system_prompt, user_prompt

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
                    elif block.type == "tool_use":
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
                    elif block.get("type") == "tool_use":
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

    def _build_tool_instructions(self, tools: list[dict[str, Any]]) -> str:
        """Build tool usage instructions for system prompt.

        Args:
            tools: List of tool definitions.

        Returns:
            Instruction string for system prompt.
        """
        if not tools:
            return ""

        tools_json = json.dumps(tools, indent=2)

        return f"""You have access to the following tools:

{tools_json}

CRITICAL - Follow the input_schema exactly:
- For "enum" fields, ONLY use values listed in the enum array
- For "required" fields, you MUST provide a value
- Do NOT invent parameter names or values not defined in the schema

To use a tool, output a tool_use block in this exact format:
<tool_use>
{{"tool": "tool_name", "id": "unique_id", "input": {{"param1": "value1"}}}}
</tool_use>

Important:
- Generate a unique ID for each tool call (e.g., "call_1", "call_2", etc.)
- Wait for the tool result before continuing
- You can use multiple tools in sequence
- Tool results will be provided in <tool_result> blocks
- After receiving a tool result, continue your response or use another tool"""

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

        # Add session resumption if we have an existing session
        if session_id:
            cmd.extend(["--resume", session_id])
            logger.info(f"[PROVIDER] Resuming Claude session: {session_id}")

        return cmd

    async def _execute_cli(self, cmd: list[str], prompt: str) -> dict[str, Any]:
        """Execute the CLI command and parse streaming output.

        Args:
            cmd: The command to execute.
            prompt: The prompt to send via stdin (avoids ARG_MAX limits).

        Returns:
            Dictionary with parsed response data.
        """
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
        tool_calls: list[dict[str, Any]] = []
        block_index = 0
        block_started = False

        assert proc.stdout is not None

        async for line in proc.stdout:
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

            # Handle stream events (content deltas)
            if event_type == "stream_event":
                inner_event = event_data.get("event", {})
                inner_type = inner_event.get("type")

                if inner_type == "content_block_start":
                    block_started = True
                    await self._emit_event(
                        CONTENT_BLOCK_START,
                        {
                            "index": block_index,
                            "content_block": inner_event.get("content_block", {}),
                        },
                    )

                elif inner_type == "content_block_delta":
                    delta = inner_event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_chunk = delta.get("text", "")
                        response_text += text_chunk

                        await self._emit_event(
                            CONTENT_BLOCK_DELTA,
                            {
                                "index": block_index,
                                "delta": TextContent(text=text_chunk),
                            },
                        )

                elif inner_type == "content_block_stop":
                    if block_started:
                        await self._emit_event(
                            CONTENT_BLOCK_END,
                            {"index": block_index},
                        )
                        block_index += 1
                        block_started = False

            # Handle assistant messages
            elif event_type == "assistant":
                message = event_data.get("message", {})
                content_blocks = message.get("content", [])

                for block in content_blocks:
                    if block.get("type") == "text":
                        # Accumulate text if we missed it from streaming
                        text = block.get("text", "")
                        if text and text not in response_text:
                            response_text += text

            # Handle final result
            elif event_type == "result":
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
            logger.error(f"[PROVIDER] CLI failed: {error_msg}")
            raise RuntimeError(
                f"Claude Code CLI failed (exit {proc.returncode}): {error_msg}"
            )

        # Parse tool calls from response text
        tool_calls = self._extract_tool_calls(response_text)

        return {
            "text": response_text,
            "tool_calls": tool_calls,
            "usage": usage_data,
            "metadata": metadata,
        }

    # -------------------------------------------------------------------------
    # Response building
    # -------------------------------------------------------------------------

    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from response text.

        Looks for <tool_use>...</tool_use> blocks in the response.

        Args:
            text: The response text to parse.

        Returns:
            List of tool call dictionaries.
        """
        tool_calls = []

        # Find all tool_use blocks
        import re

        pattern = r"<tool_use>\s*(.*?)\s*</tool_use>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_call = {
                    "id": tool_data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": tool_data.get("tool", tool_data.get("name", "")),
                    "arguments": tool_data.get("input", tool_data.get("arguments", {})),
                }
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.warning(f"[PROVIDER] Failed to parse tool call: {match[:100]}")
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
    ) -> ChatResponse:
        """Build a ChatResponse from parsed response data.

        Args:
            response_data: Parsed response from CLI.
            duration: Request duration in seconds.

        Returns:
            ChatResponse object.
        """
        raw_text = response_data.get("text", "")
        tool_call_dicts = response_data.get("tool_calls", [])
        usage_data = response_data.get("usage", {})
        metadata = response_data.get("metadata", {})

        # Clean response text (remove tool_use blocks)
        clean_text = self._clean_response_text(raw_text)

        # Build content blocks
        content_blocks: list[Any] = []

        if clean_text:
            content_blocks.append(TextBlock(type="text", text=clean_text))

        # Add tool call blocks to content
        for tc in tool_call_dicts:
            content_blocks.append(
                ToolCallBlock(
                    id=tc["id"],
                    name=tc["name"],
                    input=tc["arguments"],
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

        usage = Usage(
            input_tokens=total_input,
            output_tokens=output_tokens,
            total_tokens=total_input + output_tokens,
        )

        # Determine finish reason
        finish_reason = "tool_use" if tool_calls else "end_turn"

        logger.info(
            f"[PROVIDER] Response: {len(clean_text)} chars, "
            f"{len(tool_call_dicts)} tool calls, {duration:.2f}s "
            f"(tokens: {total_input} in, {output_tokens} out)"
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata,
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
