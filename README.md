# Amplifier Claude Provider Module

Claude model integration for Amplifier via direct Claude Code CLI integration.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager
- **Claude Code CLI** - Required for Claude Max subscription access

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installing Claude Code CLI

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

## Purpose

Provides access to Anthropic's Claude models (Sonnet, Opus, Haiku) via Claude Code CLI, enabling use of a **Claude Max subscription** instead of API billing.

### Key Features

- **No API key required** - Uses Claude Code's authentication (Claude Max subscription)
- **Unlimited context size** - Bypasses ARG_MAX limits via file-based system prompts
- **High throughput** - Stdin streaming for prompts, stdout streaming for responses
- **Session continuity** - Continue/resume conversation sessions
- **Zero SDK dependencies** - Direct CLI integration via subprocess

## Architecture

This provider bypasses the `claude-agent-sdk` limitations by using direct CLI invocation:

```
┌─────────────────────────────────────────────────────────────────┐
│  Amplifier Provider                                              │
│                                                                  │
│  1. System prompt → temp file (bypasses ARG_MAX)                │
│  2. CLI: claude --system-prompt-file /tmp/xxx.txt               │
│          --input-format stream-json                             │
│  3. User prompt → stdin (NDJSON, unlimited size)                │
│  4. Response ← stdout (stream-json parsing)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Why Direct CLI?

The `claude-agent-sdk` passes system prompts as CLI arguments, hitting OS ARG_MAX limits (~500KB). For agentic tools with large context windows, this is a blocker.

| Approach | System Prompt Limit | User Prompt Limit |
|----------|--------------------|--------------------|
| `claude-agent-sdk` | ~500 KB (ARG_MAX) | ~500 KB (ARG_MAX) |
| **Direct CLI** | **Unlimited** (file) | **Unlimited** (stdin) |

## Contract

**Module Type:** Provider  
**Mount Point:** `providers`  
**Entry Point:** `amplifier_module_provider_claude:mount`

## Supported Models

- `sonnet` - Claude Sonnet (recommended, default)
- `opus` - Claude Opus (most capable, extended thinking)
- `haiku` - Claude Haiku (fastest)

## Configuration

```yaml
providers:
  - module: provider-claude
    name: claude
    config:
      default_model: sonnet
      max_turns: 1
      debug: false
      # Optional: working directory for sessions
      cwd: /path/to/project
      # Optional: permission mode (default, plan, acceptEdits, bypassPermissions)
      permission_mode: default
      # Optional: auto-continue previous session
      auto_continue: false
```

### Tool Configuration

Control which Claude Code built-in tools are available:

```yaml
providers:
  - module: provider-claude
    config:
      allowed_tools:
        - Read
        - Write
        - Edit
        - Bash
        - Grep
        - Glob
      disallowed_tools:
        - WebSearch
        - WebFetch
```

### Available Built-in Tools

| Tool | Description |
|------|-------------|
| `Read` | Read file contents |
| `Write` | Write file contents |
| `Edit` | Edit file contents |
| `MultiEdit` | Edit multiple files |
| `Bash` | Execute shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `LS` | List directory contents |
| `WebFetch` | Fetch web content |
| `WebSearch` | Search the web |
| `Task` | Delegate to sub-agents |
| `TodoRead` | Read todo list |
| `TodoWrite` | Write todo list |
| `NotebookRead` | Read Jupyter notebooks |
| `NotebookEdit` | Edit Jupyter notebooks |

## Usage

```python
# In amplifier configuration
providers:
  - module: provider-claude
    name: claude
    config:
      default_model: sonnet
```

### Session Continuity

Continue a previous session:

```python
response = await provider.complete(
    request,
    continue_session=True,  # Continue last session in cwd
)

# Or resume a specific session:
response = await provider.complete(
    request,
    session_id="session-uuid-here",
)
```

## Features

- **Streaming support** - Real-time response streaming
- **Tool use** - Full tool calling support with automatic mapping
- **Extended thinking** - Supported on Opus model
- **Session management** - Continue/resume conversations
- **Unlimited context** - No ARG_MAX limitations

## Dependencies

- `amplifier-core>=1.0.0`
- Claude Code CLI (installed separately)

**No Python SDK dependencies** - This provider uses direct subprocess communication.

## Events

The provider emits standard Amplifier events:

- `llm:request` - Before CLI invocation (includes system_prompt_bytes for monitoring)
- `llm:response` - After successful response (includes session_id for continuity)

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `CLINotFoundError` | Claude Code CLI not installed | Run install script |
| `CLIProcessError` | CLI returned non-zero exit | Check stderr output |

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
