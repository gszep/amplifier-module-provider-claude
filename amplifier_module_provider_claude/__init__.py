"""Claude provider module for Amplifier.

Direct CLI integration with Claude Code for Claude Max subscription usage.
Bypasses claude-agent-sdk limitations (ARG_MAX) by using file-based system prompts
and stdin streaming for unlimited context sizes.
"""

__all__ = ["mount", "ClaudeProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from amplifier_core import (
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
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


class ClaudeCodeCLIError(Exception):
    """Base error for Claude Code CLI issues."""

    pass


class CLINotFoundError(ClaudeCodeCLIError):
    """Claude Code CLI not found."""

    pass


class CLIProcessError(ClaudeCodeCLIError):
    """Claude Code CLI process failed."""

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Claude provider using direct Claude Code CLI integration.

    This implementation bypasses claude-agent-sdk limitations by:
    - Using --system-prompt-file for unlimited system prompt sizes
    - Using --input-format stream-json for stdin-based prompts
    - Streaming responses via stdout JSON parsing

    Args:
        coordinator: Module coordinator
        config: Provider configuration

    Returns:
        Optional cleanup function
    """
    config = config or {}

    provider = ClaudeProvider(config, coordinator)
    await coordinator.mount("providers", provider, name="claude")
    logger.info("Mounted ClaudeProvider (direct CLI integration)")

    return None


class ClaudeProvider:
    """Direct Claude Code CLI integration for Amplifier.

    Provides Claude models through direct CLI invocation, using a Claude Max
    subscription instead of API billing. Bypasses SDK limitations for
    unlimited context sizes.

    Features:
    - Uses Claude Max subscription (no API key required)
    - Unlimited system prompt size via --system-prompt-file
    - Unlimited user prompt size via stdin streaming
    - Supports sonnet, opus, and haiku models
    - Session continuity (continue/resume)
    - No subprocess argument length limits
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

        # Verify CLI is available
        self._cli_path: str | None = None

    def _find_cli(self) -> str:
        """Find the Claude Code CLI executable.

        Returns:
            Path to the CLI executable

        Raises:
            CLINotFoundError: If CLI is not found
        """
        if self._cli_path:
            return self._cli_path

        # Check common locations
        cli_path = shutil.which("claude")
        if cli_path:
            self._cli_path = cli_path
            return cli_path

        # Check npm global install locations
        npm_paths = [
            Path.home() / ".npm-global" / "bin" / "claude",
            Path("/usr/local/bin/claude"),
            Path.home() / ".local" / "bin" / "claude",
        ]
        for path in npm_paths:
            if path.exists():
                self._cli_path = str(path)
                return self._cli_path

        raise CLINotFoundError(
            "Claude Code CLI not found. Please install it: "
            "curl -fsSL https://claude.ai/install.sh | bash"
        )

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
        Generate completion via direct Claude Code CLI invocation.

        Uses file-based system prompt and stdin streaming to bypass
        ARG_MAX limitations, enabling unlimited context sizes.

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
        start_time = time.time()

        # Find CLI
        try:
            cli_path = self._find_cli()
        except CLINotFoundError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            await self._emit_error_event(
                kwargs.get("model", self.default_model), str(e), elapsed_ms
            )
            raise RuntimeError(str(e)) from e

        # Extract prompt and system message from request
        prompt = self._extract_prompt(request)
        system_prompt = self._extract_system_prompt(request)

        if not prompt:
            logger.warning("[PROVIDER] Claude Code: No user prompt found in request")
            return ChatResponse(
                content=[TextBlock(text="No prompt provided")],
                tool_calls=None,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                finish_reason="error",
            )

        # Build options
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

        logger.info(
            f"[PROVIDER] Claude Code CLI call - model: {model}, max_turns: {max_turns}, "
            f"tools: {len(allowed_tools) if allowed_tools else 'default'}, "
            f"system_prompt_size: {len(system_prompt) if system_prompt else 0} bytes"
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
                    "system_prompt_bytes": len(system_prompt) if system_prompt else 0,
                },
            )

        # Create temp file for system prompt (bypasses ARG_MAX)
        system_prompt_file: str | None = None
        try:
            if system_prompt:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    delete=False,
                    encoding="utf-8",
                ) as f:
                    f.write(system_prompt)
                    system_prompt_file = f.name
                logger.debug(
                    f"[PROVIDER] Wrote system prompt to temp file: {system_prompt_file} "
                    f"({len(system_prompt):,} bytes)"
                )

            # Build CLI command
            cmd = self._build_command(
                cli_path=cli_path,
                model=model,
                max_turns=max_turns,
                system_prompt_file=system_prompt_file,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                permission_mode=permission_mode,
                continue_session=continue_session,
                resume_session_id=resume_session_id,
            )

            # Execute CLI with stdin streaming
            response = await self._execute_cli(cmd, prompt, model, start_time)
            return response

        finally:
            # Clean up temp file
            if system_prompt_file:
                try:
                    Path(system_prompt_file).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def _build_command(
        self,
        cli_path: str,
        model: str,
        max_turns: int,
        system_prompt_file: str | None,
        allowed_tools: list[str] | None,
        disallowed_tools: list[str] | None,
        permission_mode: str | None,
        continue_session: bool,
        resume_session_id: str | None,
    ) -> list[str]:
        """Build the CLI command with all options.

        Args:
            cli_path: Path to claude CLI
            model: Model name (sonnet, opus, haiku)
            max_turns: Maximum agentic turns
            system_prompt_file: Path to system prompt file (bypasses ARG_MAX)
            allowed_tools: List of allowed tools
            disallowed_tools: List of disallowed tools
            permission_mode: Permission mode
            continue_session: Whether to continue last session
            resume_session_id: Session ID to resume

        Returns:
            Command list for subprocess
        """
        cmd = [
            cli_path,
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",  # Stdin streaming for unlimited prompt size
            "--model",
            model,
            "--max-turns",
            str(max_turns),
        ]

        # System prompt via file (bypasses ARG_MAX entirely)
        if system_prompt_file:
            cmd.extend(["--system-prompt-file", system_prompt_file])

        # Tool configuration
        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        if disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(disallowed_tools)])

        # Permission mode
        if permission_mode:
            cmd.extend(["--permission-mode", permission_mode])

        # Session continuity
        if continue_session:
            cmd.append("--continue")
        elif resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        elif self._last_session_id and self.config.get("auto_continue", False):
            cmd.extend(["--resume", self._last_session_id])

        # Working directory
        if self._session_cwd:
            cmd.extend(["--add-dir", self._session_cwd])

        # Debug mode
        if self.debug:
            cmd.append("--verbose")

        return cmd

    async def _execute_cli(
        self,
        cmd: list[str],
        prompt: str,
        model: str,
        start_time: float,
    ) -> ChatResponse:
        """Execute the CLI command and parse response.

        Uses stdin streaming for the prompt, bypassing ARG_MAX for user input.

        Args:
            cmd: CLI command list
            prompt: User prompt to send via stdin
            model: Model name for event emission
            start_time: Request start time for duration tracking

        Returns:
            ChatResponse with parsed content

        Raises:
            RuntimeError: On CLI execution failure
        """
        # Set up environment
        env = os.environ.copy()
        env["CLAUDE_CODE_ENTRYPOINT"] = "amplifier"

        logger.debug(f"[PROVIDER] Executing: {' '.join(cmd[:5])}...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._session_cwd,
            )

            # Verify streams are available
            if proc.stdin is None or proc.stdout is None or proc.stderr is None:
                raise CLIProcessError("Failed to open subprocess streams", None)

            # Send prompt via stdin (NDJSON format)
            # Format: {"type": "user", "message": {"role": "user", "content": "..."}}
            input_msg = {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "session_id": "default",
            }
            stdin_data = json.dumps(input_msg) + "\n"

            # Write to stdin and close to signal end of input
            proc.stdin.write(stdin_data.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()

            # Read and parse streaming output
            content_blocks: list[TextBlock | ThinkingBlock | ToolCallBlock] = []
            tool_calls: list[ToolCall] = []
            session_id: str | None = None
            result_text: str | None = None

            # Read stdout line by line (NDJSON)
            async for line in proc.stdout:
                line = line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                    parsed = self._parse_message(msg)

                    if parsed:
                        if "content_blocks" in parsed:
                            content_blocks.extend(parsed["content_blocks"])
                        if "tool_calls" in parsed:
                            tool_calls.extend(parsed["tool_calls"])
                        if "session_id" in parsed:
                            session_id = parsed["session_id"]
                        if "result" in parsed:
                            result_text = parsed["result"]

                except json.JSONDecodeError as e:
                    logger.warning(f"[PROVIDER] Failed to parse JSON line: {e}")
                    continue

            # Wait for process to complete
            await proc.wait()

            # Check for errors
            if proc.returncode != 0:
                stderr_data = await proc.stderr.read()
                stderr_text = stderr_data.decode("utf-8") if stderr_data else ""
                elapsed_ms = int((time.time() - start_time) * 1000)
                error_msg = f"Claude Code CLI failed (exit {proc.returncode}): {stderr_text[:500]}"
                logger.error(f"[PROVIDER] {error_msg}")
                await self._emit_error_event(model, error_msg, elapsed_ms)
                raise CLIProcessError(error_msg, proc.returncode)

            # Store session ID for continuation
            if session_id:
                self._last_session_id = session_id

            elapsed_ms = int((time.time() - start_time) * 1000)

            # If we got a result but no content blocks, use result as text
            if result_text and not content_blocks:
                content_blocks.append(TextBlock(text=result_text))

            # Emit success event
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

            return ChatResponse(
                content=content_blocks if content_blocks else [TextBlock(text="")],
                tool_calls=tool_calls if tool_calls else None,
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
                finish_reason="end_turn",
            )

        except asyncio.CancelledError:
            logger.warning("[PROVIDER] CLI execution cancelled")
            raise
        except CLIProcessError:
            raise
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error(f"[PROVIDER] Claude Code CLI error: {error_msg}")
            await self._emit_error_event(model, error_msg, elapsed_ms)
            raise RuntimeError(f"Claude Code CLI error: {error_msg}") from e

    def _parse_message(self, msg: dict) -> dict | None:
        """Parse a streaming JSON message from Claude Code CLI.

        Message types:
        - assistant: Contains content blocks (text, thinking, tool_use)
        - result: Final result with session info
        - system: System messages (init, etc.)

        Args:
            msg: Parsed JSON message

        Returns:
            Dict with extracted data, or None if not relevant
        """
        msg_type = msg.get("type")

        if msg_type == "assistant":
            # Assistant message with content blocks
            message = msg.get("message", {})
            content = message.get("content", [])

            content_blocks = []
            tool_calls = []

            for block in content:
                block_type = block.get("type")

                if block_type == "text":
                    content_blocks.append(TextBlock(text=block.get("text", "")))

                elif block_type == "thinking":
                    content_blocks.append(
                        ThinkingBlock(
                            thinking=block.get("thinking", ""),
                            signature=block.get("signature"),
                        )
                    )

                elif block_type == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    content_blocks.append(
                        ToolCallBlock(id=tool_id, name=tool_name, input=tool_input)
                    )
                    tool_calls.append(
                        ToolCall(id=tool_id, name=tool_name, arguments=tool_input)
                    )

            result = {}
            if content_blocks:
                result["content_blocks"] = content_blocks
            if tool_calls:
                result["tool_calls"] = tool_calls

            # Extract session ID if present
            if "session_id" in msg:
                result["session_id"] = msg["session_id"]

            return result if result else None

        elif msg_type == "result":
            # Final result message
            result = {
                "session_id": msg.get("session_id"),
                "result": msg.get("result"),
            }
            return result

        elif msg_type == "system":
            # System messages (init, ready, etc.) - extract session ID
            if "session_id" in msg:
                return {"session_id": msg["session_id"]}

        return None

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
