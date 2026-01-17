"""Claude provider module for Amplifier.

Direct CLI integration with Claude Code for Claude Max subscription usage.
Claude Code handles tool execution internally - Amplifier uses this as a pure LLM provider.
"""

__all__ = ["mount", "ClaudeProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import re
import shutil
import time
from typing import Any

from amplifier_core import (  # type: ignore
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
)
from amplifier_core.content_models import TextContent, ToolCallContent  # type: ignore
from amplifier_core.events import (  # type: ignore
    CONTENT_BLOCK_DELTA,
    CONTENT_BLOCK_END,
    CONTENT_BLOCK_START,
    TOOL_POST,
    TOOL_PRE,
)
from amplifier_core.message_models import (  # type: ignore
    ChatRequest,
    ChatResponse,
    TextBlock,
    ToolCall,
    Usage,
)

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Claude provider using Claude Code CLI."""
    config = config or {}

    # Check if CLI is available
    cli_path = shutil.which("claude")
    if not cli_path:
        logger.warning(
            "Claude Code CLI not found. Install with: curl -fsSL https://claude.ai/install.sh | bash"
        )
        return None

    provider = ClaudeProvider(config, coordinator)
    await coordinator.mount("providers", provider, name="claude")
    logger.info("Mounted ClaudeProvider (Claude Code CLI)")
    return None


class ClaudeProvider:
    """Claude Code CLI integration for Amplifier.

    Uses Claude Code CLI as a pure LLM provider. All tool calling and session
    management is handled by Amplifier's orchestrator.
    """

    name = "claude"

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

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "sonnet")

    def get_info(self) -> ProviderInfo:
        """Return provider information."""
        return ProviderInfo(
            id="claude",
            display_name="Claude (Claude Code)",
            credential_env_vars=[],  # No API key needed - uses Claude Code auth
            capabilities=["streaming", "tools", "thinking"],
            defaults={"model": self.default_model},
            config_fields=[],  # No config needed - CLI handles auth
        )

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
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
            for spec in self.MODELS.values()
        ]

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Parse tool calls from response.

        Tool calls are already extracted in complete() and placed in response.tool_calls.
        """
        return response.tool_calls or []

    def _extract_prompt(self, request: ChatRequest) -> str:
        """Extract the actual user prompt from messages.

        Amplifier adds system reminders as separate user messages, so we need to find
        the actual user input (not just system reminders).
        """
        for msg in reversed(request.messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    text_parts = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                    content = "\n".join(text_parts)
                else:
                    continue

                # Skip messages that are ONLY system reminders (no actual user content)
                stripped = re.sub(
                    r"<system-reminder[^>]*>.*?</system-reminder>\s*",
                    "",
                    content,
                    flags=re.DOTALL,
                ).strip()

                if stripped:
                    return stripped

        raise RuntimeError("No user message found in request")

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event through the coordinator's hooks if available."""
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(event, data)

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Execute a completion request via Claude Code CLI with real-time streaming.

        Uses stream-json output format to emit content_block events as text arrives,
        enabling real-time UI updates while still returning a complete ChatResponse.

        Session Continuity:
        - If request.metadata contains 'claude_session_id', resumes that session
        - Response metadata always includes 'claude_session_id' for future continuity
        - Use --resume flag to continue multi-turn conversations
        """
        start_time = time.time()

        # Find CLI
        cli_path = shutil.which("claude")
        if not cli_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: curl -fsSL https://claude.ai/install.sh | bash"
            )

        # Get model and extract prompt
        model = getattr(request, "model", None) or self.default_model
        prompt = self._extract_prompt(request)

        # Check for existing session to resume
        request_metadata = getattr(request, "metadata", None) or {}
        existing_session_id = request_metadata.get("claude_session_id")

        # Build command with streaming JSON output for real-time events
        # --verbose is required for stream-json format
        # --include-partial-messages gives us content_block_delta events
        cmd = [
            cli_path,
            "-p",  # Print mode (non-interactive)
            "--model",
            model,
            "--output-format",
            "stream-json",  # Real-time streaming
            "--verbose",  # Required for stream-json
            "--include-partial-messages",  # Get content deltas
        ]

        # Add session resumption if we have an existing session
        if existing_session_id:
            cmd.extend(["--resume", existing_session_id])
            logger.info(f"[PROVIDER] Resuming Claude session: {existing_session_id}")
        else:
            # Only set system prompt for new sessions
            cmd.extend(
                [
                    "--system-prompt",
                    "You are a helpful AI assistant. Answer the user's request directly.",
                ]
            )

        # Add the prompt
        cmd.append(prompt)

        logger.info(
            f"[PROVIDER] Claude CLI streaming: model={model}, prompt_len={len(prompt)}, "
            f"resume={'yes' if existing_session_id else 'no'}"
        )

        # Start the process
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Process streaming output line by line
        response_text = ""
        usage_data: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        tool_calls: list[ToolCall] = []
        tool_results: dict[str, Any] = {}  # Map tool_use_id to result
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
                logger.warning(
                    f"[PROVIDER] Failed to parse streaming line: {line_str[:100]}"
                )
                continue

            event_type = event_data.get("type")

            # Handle stream events (content deltas)
            if event_type == "stream_event":
                inner_event = event_data.get("event", {})
                inner_type = inner_event.get("type")

                if inner_type == "content_block_start":
                    # Emit content_block:start event
                    block_started = True
                    await self._emit_event(
                        CONTENT_BLOCK_START,
                        {
                            "index": block_index,
                            "content_block": inner_event.get("content_block", {}),
                        },
                    )

                elif inner_type == "content_block_delta":
                    # Extract text delta and emit event
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
                    # Emit content_block:end event
                    if block_started:
                        await self._emit_event(
                            CONTENT_BLOCK_END,
                            {"index": block_index},
                        )
                        block_index += 1
                        block_started = False

            # Handle assistant messages (may contain tool_use blocks)
            elif event_type == "assistant":
                message = event_data.get("message", {})
                content_blocks = message.get("content", [])

                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        # Claude Code is requesting a tool call
                        tool_id = block.get("id", "")
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})

                        # Create ToolCall for Amplifier (uses 'arguments' not 'input')
                        tool_call = ToolCall(
                            id=tool_id,
                            name=tool_name,
                            arguments=tool_input,
                        )
                        tool_calls.append(tool_call)

                        # Emit tool:pre event
                        await self._emit_event(
                            TOOL_PRE,
                            {
                                "tool_call": ToolCallContent(
                                    id=tool_id,
                                    name=tool_name,
                                    arguments=tool_input,
                                ),
                                "provider": "claude",
                                "note": "Tool executed by Claude Code internally",
                            },
                        )

                        logger.info(
                            f"[PROVIDER] Tool call: {tool_name}({json.dumps(tool_input)[:100]}...)"
                        )

            # Handle user messages (contain tool results from Claude Code)
            elif event_type == "user":
                message = event_data.get("message", {})
                content_blocks = message.get("content", [])

                for block in content_blocks:
                    if block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        is_error = block.get("is_error", False)

                        tool_results[tool_use_id] = {
                            "content": result_content,
                            "is_error": is_error,
                        }

                        # Emit tool:post event
                        await self._emit_event(
                            TOOL_POST,
                            {
                                "tool_call_id": tool_use_id,
                                "result": result_content[:500]
                                if isinstance(result_content, str)
                                else str(result_content)[:500],
                                "is_error": is_error,
                                "provider": "claude",
                            },
                        )

            # Handle final result (contains usage stats)
            elif event_type == "result":
                # Use result text if we didn't accumulate from deltas
                if not response_text:
                    response_text = event_data.get("result", "")

                usage_data = event_data.get("usage", {})
                # Store session_id with consistent key for resumption
                session_id = event_data.get("session_id")
                metadata = {
                    "claude_session_id": session_id,  # Key for session continuity
                    "session_id": session_id,  # Also keep original key
                    "duration_ms": event_data.get("duration_ms"),
                    "duration_api_ms": event_data.get("duration_api_ms"),
                    "cost_usd": event_data.get("total_cost_usd"),
                    "num_turns": event_data.get("num_turns"),
                    "tool_calls_count": len(tool_calls),
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

        duration = time.time() - start_time

        # Build usage information
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

        logger.info(
            f"[PROVIDER] Response received in {duration:.2f}s "
            f"(tokens: {total_input} in, {output_tokens} out)"
        )

        # Build content blocks including any tool calls
        content_blocks: list[Any] = [TextBlock(type="text", text=response_text)]

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason="end_turn" if not tool_calls else "tool_use",
            metadata=metadata,
        )
