# Amplifier Claude Code Provider

**Use your Claude Max/Pro subscription with Amplifier** — no API keys or per-token billing required.

## Quick Start

### 1. Install Prerequisites

```bash
# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash

# Install Amplifier
uv tool install git+https://github.com/microsoft/amplifier
```

### 2. Add the Provider

```bash
amplifier module add provider-claude --source git+https://github.com/gszep/amplifier-module-provider-claude@main
```

### 3. Configuration

```bash
amplifier init
```

> **Note**: If `ANTHROPIC_API_KEY` is set, Amplifier prefers direct API access. Remove it from `~/.amplifier/keys.env` to use your subscription.

This provider parses tool calls from model outputs and therefore the `tools-reminder` hook is needed to minimize hallucinations `~/.amplifier/settings.yaml`:

```yaml
behaviors:
  - source: git+https://github.com/gszep/amplifier-module-provider-claude@main
    path: behaviors/claude-with-tools-reminder.yaml
```

## Models

| Model | ID | Best For |
|-------|------|----------|
| Sonnet | `sonnet` | Default — balanced speed and capability |
| Opus | `opus` | Complex reasoning, extended thinking |
| Haiku | `haiku` | Fast responses |

## How It Works

This provider wraps the Claude Code CLI in "Full Control" mode:
- Tool definitions are injected via system prompt
- Claude's built-in tools are disabled (`--tools ""`)
- Amplifier's orchestrator handles all tool execution
- Responses are parsed for `<tool_use>` blocks

This gives Amplifier full control over the tool ecosystem while using your Claude subscription.

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — Prompt structure, text-based tool calling, session caching
- **[Feature Coverage](docs/FEATURE_COVERAGE.md)** — Comparison with the Anthropic API provider, known limitations

## Development

```bash
uv run pytest tests/ -v      # Run tests
uv run ruff check .          # Lint
uv run pyright               # Type check
```

## Contributing

This project is not currently accepting external contributions. Feel free to fork and experiment.

Most contributions require a [Contributor License Agreement](https://cla.opensource.microsoft.com).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
