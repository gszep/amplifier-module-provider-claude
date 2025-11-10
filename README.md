# Amplifier Anthropic Provider Module

Claude model integration for Amplifier via Anthropic API.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides access to Anthropic's Claude models (Claude 4 series: Sonnet, Opus, Haiku) as an LLM provider for Amplifier.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_anthropic:mount`

## Supported Models

- `claude-sonnet-4-5` - Claude Sonnet 4.5 (recommended, default)
- `claude-opus-4-1` - Claude Opus 4.1 (most capable)
- `claude-haiku-4-5` - Claude Haiku 4.5 (fastest, cheapest)

## Configuration

```toml
[[providers]]
module = "provider-anthropic"
name = "anthropic"
config = {
    default_model = "claude-sonnet-4-5",
    max_tokens = 8192,
    temperature = 1.0,
    debug = false,      # Enable standard debug events
    raw_debug = false   # Enable ultra-verbose raw API I/O logging
}
```

### Debug Configuration

**Standard Debug** (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):
- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from Anthropic API before any processing

**Example**:
```yaml
providers:
  - module: provider-anthropic
    config:
      debug: true      # Enable debug events
      raw_debug: true  # Enable raw API I/O capture
      default_model: claude-sonnet-4-5
```

## Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

```python
# In amplifier configuration
[provider]
name = "anthropic"
default_model = "claude-sonnet-4-5"
```

## Features

- Streaming support
- Tool use (function calling)
- Vision capabilities (on supported models)
- Token counting and management
- **Message validation** before API calls (defense in depth)

## Message Validation

Before sending messages to Anthropic API, the provider validates tool_use/tool_result consistency:

- Every tool_use block has a matching tool_result block
- Tool pairs are adjacent (tool_result immediately follows tool_use)
- No orphaned tool_results from failed operations

This defense-in-depth validation catches conversation state corruption before it reaches the API, providing clear error messages instead of cryptic 400 errors from Anthropic

## Dependencies

- `amplifier-core>=1.0.0`
- `anthropic>=0.25.0`

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
