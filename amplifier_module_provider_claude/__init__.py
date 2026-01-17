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
import shutil
import time
from typing import Any

from amplifier_core import (  # type: ignore
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
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

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Execute a completion request via Claude Code CLI.

        Handles:
        - System prompts (passed via --system-prompt flag)
        - User prompts (passed as CLI argument)
        - JSON output parsing for structured responses
        - Token usage tracking
        """
        start_time = time.time()

        # Find CLI
        cli_path = shutil.which("claude")
        if not cli_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: curl -fsSL https://claude.ai/install.sh | bash"
            )

        # Get model
        model = getattr(request, "model", None) or self.default_model

        # Extract system prompt from messages
        system_parts: list[str] = []
        for msg in request.messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_parts.append(msg.content)
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            system_parts.append(block.text)

        # Extract developer messages (context files) - wrap in XML
        for msg in request.messages:
            if msg.role == "developer":
                if isinstance(msg.content, str):
                    system_parts.append(
                        f"<context_file>\n{msg.content}\n</context_file>"
                    )
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            system_parts.append(
                                f"<context_file>\n{block.text}\n</context_file>"
                            )

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        # Extract the user prompt (last user message)
        prompt = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    prompt = msg.content
                elif isinstance(msg.content, list):
                    # Combine all text blocks
                    text_parts = []
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                    prompt = "\n".join(text_parts)
                break

        if not prompt:
            raise RuntimeError("No user message found in request")

        # Build command with JSON output for structured parsing
        # NOTE: System prompts from Amplifier are intentionally NOT passed to Claude CLI
        # because they can exceed ARG_MAX limits. Claude Code operates as a pure LLM
        # provider - Amplifier handles all context management separately.
        cmd = [
            cli_path,
            "-p",  # Print mode (non-interactive)
            "--model",
            model,
            "--output-format",
            "json",  # Get structured response
            prompt,  # User prompt as argument
        ]

        logger.info(
            f"[PROVIDER] Claude CLI: model={model}, prompt_len={len(prompt)}, "
            f"system_len={len(system_prompt) if system_prompt else 0} (not passed to CLI)"
        )

        # Execute
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        raw_output = stdout.decode("utf-8").strip()

        # Debug: log the raw CLI output
        logger.info(f"[PROVIDER] Raw CLI output: {raw_output[:500]}...")

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8").strip() or raw_output
            logger.error(f"[PROVIDER] CLI failed: {error_msg}")
            raise RuntimeError(
                f"Claude Code CLI failed (exit {proc.returncode}): {error_msg}"
            )

        duration = time.time() - start_time

        # Parse JSON response
        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.error(f"[PROVIDER] Failed to parse JSON response: {e}")
            # Fall back to raw text if JSON parsing fails
            return ChatResponse(
                content=[TextBlock(type="text", text=raw_output)],
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                finish_reason="end_turn",
            )

        # Extract response text
        response_text = result.get("result", "")

        # Extract usage information
        usage_data = result.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)
        # Include cache tokens in input count for accurate tracking
        cache_read = usage_data.get("cache_read_input_tokens", 0)
        cache_creation = usage_data.get("cache_creation_input_tokens", 0)
        total_input = input_tokens + cache_read + cache_creation

        usage = Usage(
            input_tokens=total_input,
            output_tokens=output_tokens,
            total_tokens=total_input + output_tokens,
        )

        # Build metadata with session info for potential future use
        metadata = {
            "session_id": result.get("session_id"),
            "duration_ms": result.get("duration_ms"),
            "duration_api_ms": result.get("duration_api_ms"),
            "cost_usd": result.get("total_cost_usd"),
        }

        logger.info(
            f"[PROVIDER] Response received in {duration:.2f}s "
            f"(tokens: {total_input} in, {output_tokens} out)"
        )

        return ChatResponse(
            content=[TextBlock(type="text", text=response_text)],
            usage=usage,
            finish_reason="end_turn",
            metadata=metadata,
        )
