# Amplifier Claude Code Provider Module

**Use your Claude Max/Pro subscription with Amplifier** - no API keys or per-token billing required.

This provider integrates Claude models via the Claude Code CLI, enabling Amplifier users to leverage their existing Claude subscription instead of paying for API access.

## Architecture

This provider follows the same pattern as the official OpenAI provider:

```
┌─────────────────────────────────────────────────────────────┐
│  Amplifier Orchestrator                                     │
│      │                                                      │
│      ▼                                                      │
│  1. Send request with messages + tools                      │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ClaudeProvider.complete()                          │   │
│  │  - Converts messages to Claude format               │   │
│  │  - Injects tool definitions via system prompt       │   │
│  │  - Invokes Claude Code CLI (--tools "" disabled)    │   │
│  │  - Parses response for tool_use blocks              │   │
│  │  - Returns ChatResponse with tool_calls             │   │
│  └─────────────────────────────────────────────────────┘   │
│      │                                                      │
│      ▼                                                      │
│  2. Orchestrator executes tools                             │
│      │                                                      │
│      ▼                                                      │
│  3. Send request with tool results added to messages        │
│      │                                                      │
│      ▼                                                      │
│  4. Repeat until no more tool_calls                         │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Claude Code's built-in tools are disabled (`--tools ""`). Amplifier's orchestrator has full control over tool execution, enabling:
- Use of Amplifier's tool ecosystem
- Tool execution policies and approvals
- Consistent behavior across providers

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
- **Full Control mode** - Amplifier orchestrator handles all tool execution
- **Streaming support** - Real-time content streaming via stream-json
- **Session continuity** - Continue/resume conversation sessions
- **Zero SDK dependencies** - Direct CLI integration via subprocess

## Contract

**Module Type:** Provider  
**Mount Point:** `providers`  
**Entry Point:** `amplifier_module_provider_claude:mount`

## Supported Models

| Model | ID | Description |
|-------|------|-------------|
| Claude Sonnet | `sonnet` | Recommended default, good balance |
| Claude Opus | `opus` | Most capable, extended thinking |
| Claude Haiku | `haiku` | Fastest responses |

## Installation

```bash
amplifier module add provider-claude --source git+https://github.com/gszep/amplifier-module-provider-claude@main
```

## Configuration

```yaml
providers:
  - module: provider-claude
    name: claude
    config:
      default_model: sonnet
      timeout: 300.0
      debug: false
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_model` | string | `sonnet` | Default model (sonnet, opus, haiku) |
| `timeout` | number | `300.0` | Request timeout in seconds |
| `debug` | boolean | `false` | Enable debug logging |

## Usage

### Basic Usage

```bash
amplifier run --provider claude "What is the capital of France?"
```

### With Tools

Tools are automatically available through Amplifier's orchestrator:

```bash
amplifier run --provider claude "Search the web for recent news about AI"
```

The orchestrator will:
1. Receive tool_calls from the provider
2. Execute tools using mounted tool modules
3. Send results back to the provider
4. Continue until the response is complete

### Session Continuity

Sessions are automatically managed via the `claude:session_id` metadata:

```python
# First request - new session
response1 = await provider.complete(request1)
session_id = response1.metadata.get("claude:session_id")

# Subsequent request - continues session
request2.metadata = {"claude:session_id": session_id}
response2 = await provider.complete(request2)
```

## How It Works

### Tool Calling Flow

1. **Tool Definition Injection**: Tool specs from `request.tools` are formatted as JSON and injected into the system prompt with instructions for Claude to output `<tool_use>` blocks.

2. **Built-in Tools Disabled**: The CLI is invoked with `--tools ""` which disables all of Claude Code's built-in tools (Read, Write, Bash, etc.).

3. **Tool Call Extraction**: The provider parses the response for `<tool_use>` blocks and converts them to `ToolCall` objects.

4. **Orchestrator Execution**: Amplifier's orchestrator receives the `tool_calls` and executes them using mounted tool modules.

5. **Tool Results**: On the next `complete()` call, tool results are formatted as `<tool_result>` blocks in the conversation.

### Message Format

The provider converts Amplifier's message format to Claude CLI format:

| Amplifier Role | Claude Format |
|----------------|---------------|
| `system` | System prompt |
| `user` | `Human: {content}` |
| `assistant` | `Assistant: {content}` |
| `tool` | `Human: <tool_result>...</tool_result>` |

### Events Emitted

| Event | When |
|-------|------|
| `llm:request` | Before CLI invocation |
| `llm:response` | After successful response |
| `content_block:start` | When streaming block starts |
| `content_block:delta` | For each text chunk |
| `content_block:end` | When streaming block ends |

## Comparison with OpenAI Provider

| Aspect | Claude Provider | OpenAI Provider |
|--------|-----------------|-----------------|
| Integration | CLI subprocess | Python SDK |
| Authentication | Claude Max subscription | API key |
| Tool Control | Full Control (orchestrator) | Full Control (orchestrator) |
| Streaming | Native stream-json | Blocking API |
| Dependencies | Zero (CLI only) | openai SDK |

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `RuntimeError: CLI not found` | Claude Code CLI not installed | Run install script |
| `RuntimeError: CLI failed` | CLI returned non-zero exit | Check stderr output |

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Quality

```bash
uv run ruff check .
uv run ruff format .
uv run pyright
```

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
