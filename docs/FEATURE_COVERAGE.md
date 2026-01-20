# Feature Coverage: Claude CLI Provider vs Anthropic API Provider

This document compares the Claude CLI provider (this module) with the official Anthropic API provider.

## Architecture Comparison

| Aspect | Claude CLI Provider | Anthropic API Provider |
|--------|---------------------|------------------------|
| **Integration** | CLI subprocess | Python SDK |
| **Authentication** | Claude subscription | API key (`ANTHROPIC_API_KEY`) |
| **Billing** | Subscription (fixed cost) | Per-token usage |
| **Dependencies** | Zero (CLI binary only) | `anthropic` SDK |
| **Tool Control** | Full Control (orchestrator) | Full Control (orchestrator) |
| **Streaming** | `stream-json` mode | Native SDK streaming |

## Feature Matrix

| Feature | CLI Provider | API Provider | Notes |
|---------|:------------:|:------------:|-------|
| **Basic Completion** | ✅ | ✅ | Both support basic chat completion |
| **Streaming** | ✅ | ✅ | CLI uses `stream-json`, API uses SDK streaming |
| **Tool Calling** | ✅ | ✅ | Both via Full Control architecture |
| **Session Continuity** | ✅ | ✅ | CLI: `--resume`, API: conversation context |
| **Extended Thinking** | ✅ | ✅ | ThinkingBlock parsing implemented |
| **Multi-turn Conversations** | ✅ | ✅ | |
| **System Prompts** | ✅ | ✅ | |
| **Model Selection** | ✅ | ✅ | sonnet, opus, haiku |

### Features with Limitations

| Feature | CLI Provider | API Provider | CLI Limitation |
|---------|:------------:|:------------:|----------------|
| **Beta Headers** | ⚠️ | ✅ | Requires API key; rejected without it |
| **1M Context Window** | ⚠️ | ✅ | Beta header required (see above) |
| **Interleaved Thinking** | ⚠️ | ✅ | Changes response structure; consumes token budget |
| **JSON Schema Output** | ❌ | ✅ | `--json-schema` not available via CLI |
| **Image Input** | ❌ | ✅ | CLI doesn't support image blocks |
| **PDF Support** | ❌ | ✅ | CLI doesn't support document blocks |
| **Custom Headers** | ❌ | ✅ | CLI manages headers internally |

## Known Limitations

### CLI Subprocess Model

1. **Process Overhead**: Each request spawns a new process (unless using `--resume`)
2. **Stdin Size Limits**: Large system prompts are passed via stdin to avoid `ARG_MAX` limits
3. **Error Propagation**: CLI errors come via stderr; less structured than API exceptions

### Beta Features

**Beta headers require API authentication.** The Claude CLI rejects beta flags (like `--betas`) when used with subscription authentication:

```
Error: Custom betas are rejected by Claude CLI without API key
```

This means features that require beta access (1M context window, interleaved thinking) are **not available** when using subscription authentication.

### Response Structure

**Multiple text blocks**: Claude CLI can return multiple `text` blocks in a single response. The provider accumulates these correctly (fixed in `ee6b860`), but this behavior differs from the API SDK which typically returns a single content block.

**Interleaved thinking** (when enabled via API) changes response structure significantly:
- Thinking blocks consume output tokens
- Less budget remains for visible response text
- Can cause responses to appear "truncated" to users

### Tool Calling

**Tool definitions injected via system prompt**: Unlike the API where tools are first-class parameters, the CLI provider injects tool specs as JSON in the system prompt and parses `<tool_use>` blocks from response text.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for detailed documentation on:
- Text-based tool calling tradeoffs
- Prompt structure and XML tag reference
- Tool call parsing and validation
- Session caching for prompt efficiency

## Development Lessons (Post-Mortem)

From attempting to add features that were subsequently reverted:

### What We Learned

1. **CLI behavior is undocumented**: Many CLI flags and their interactions aren't publicly documented. Trial-and-error is required.

2. **Unit tests are insufficient**: Tests that mock CLI responses pass, but real CLI invocations reveal:
   - Beta rejection without API key
   - Response structure changes with certain flags
   - Multi-block text responses

3. **One feature per commit**: A large feature commit (6 features, 1157 lines) created cascading bugs that required 5 fix attempts before full revert.

4. **Conservative defaults**: Enabling new behaviors by default (like interleaved thinking) can break existing users.

### Recommended Approach

When adding CLI provider features:

1. **Test against real CLI** before committing
2. **Document observed CLI behavior** as you discover it
3. **Ship one feature at a time** with integration test
4. **Default new features to OFF** with explicit opt-in
5. **Stop after 2 cascading fixes** - re-investigate the design

## Events Emitted

| Event | When |
|-------|------|
| `llm:request` | Before CLI invocation |
| `llm:response` | After successful response |
| `content_block:start` | When streaming block starts |
| `content_block:delta` | For each content chunk |
| `content_block:end` | When streaming block ends |

## Configuration Reference

```yaml
providers:
  - module: provider-claude
    name: claude
    config:
      default_model: sonnet    # sonnet, opus, haiku
      timeout: 300.0           # Request timeout in seconds
      debug: false             # Enable debug logging
```

## See Also

- [README.md](../README.md) - Quick start guide
- [Anthropic API Documentation](https://docs.anthropic.com/) - Official API reference
- [Claude Code CLI](https://claude.ai/install.sh) - CLI installation
