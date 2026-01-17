"""Claude provider module for Amplifier.

Simple CLI integration with Claude Code for Claude Max subscription usage.
"""

__all__ = ["mount", "ClaudeProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
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
    Usage,
)

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Claude provider using Claude Code CLI."""
    config = config or {}
    provider = ClaudeProvider(config, coordinator)
    await coordinator.mount("providers", provider, name="claude")
    logger.info("Mounted ClaudeProvider (Claude Code CLI)")
    return None


class ClaudeProvider:
    """Simple Claude Code CLI integration for Amplifier."""

    name = "claude"
    api_label = "Claude (Claude Code)"

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
        cli_path = shutil.which("claude")
        return ProviderInfo(
            name=self.name,
            display_name="Claude (Claude Code)",
            api_label=self.api_label,
            is_configured=cli_path is not None,
            supports_streaming=False,
            models=[
                ModelInfo(name="sonnet", display_name="Claude Sonnet", is_default=True),
                ModelInfo(name="opus", display_name="Claude Opus", is_default=False),
                ModelInfo(name="haiku", display_name="Claude Haiku", is_default=False),
            ],
        )

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        """Execute a completion request via Claude Code CLI."""
        start_time = time.time()

        # Find CLI
        cli_path = shutil.which("claude")
        if not cli_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        # Get model
        model = request.model or self.default_model

        # Extract the user prompt from messages
        prompt = ""
        for msg in request.messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    prompt = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            prompt = block.text
                            break

        if not prompt:
            raise RuntimeError("No user message found in request")

        # Build simple command
        cmd = [cli_path, "-p", "--model", model, prompt]

        logger.info(f"[PROVIDER] Claude CLI: model={model}, prompt_len={len(prompt)}")

        # Execute
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        response_text = stdout.decode("utf-8").strip()

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8").strip() or response_text
            logger.error(f"[PROVIDER] CLI failed: {error_msg}")
            raise RuntimeError(
                f"Claude Code CLI failed (exit {proc.returncode}): {error_msg}"
            )

        duration = time.time() - start_time
        logger.info(f"[PROVIDER] Response received in {duration:.2f}s")

        return ChatResponse(
            content=[TextBlock(type="text", text=response_text)],
            model=model,
            usage=Usage(input_tokens=0, output_tokens=0),
            stop_reason="end_turn",
        )
