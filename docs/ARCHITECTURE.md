# Architecture: Prompt Structure and Tool Calling

This document describes how the Claude CLI provider constructs prompts and handles tool calling.

## Design Decision: Text-Based Tool Calling

The Claude CLI provider uses **text-based tool calling** rather than the native Anthropic API `tools` parameter.

### Why Not Native API Tools?

The Claude Code CLI doesn't expose the native `tools` parameter—it's a simplified interface that takes prompts via stdin. This provider works around that limitation by:

1. Injecting tool definitions into the system prompt as JSON
2. Instructing Claude to emit `<tool_use>` XML blocks in responses
3. Parsing those blocks to extract tool calls

### Tradeoffs

| Native API Tools | Text-Based Tools (This Provider) |
|------------------|----------------------------------|
| ✅ Structured `tool_use` content blocks | ❌ Regex parsing of XML from text |
| ✅ Schema validation via `strict: true` | ❌ No automatic validation |
| ✅ Advanced features (tool search, programmatic calling) | ❌ Not available |
| ❌ Requires API key + per-token billing | ✅ Uses Claude subscription |
| ❌ Must implement session caching | ✅ CLI handles caching automatically |

**The tradeoff is intentional**: leverage Claude Code's authentication and session management at the cost of native tool use.

---

## Prompt Structure

### First Turn (New Session)

When starting a new session, the full prompt has this structure:



### Resumed Turns (Session Continuation)

When resuming a session (`--resume session_id`), the CLI has cached history. Only the current turn is sent:



---

## XML Tag Reference

| Tag | Purpose | When Used |
|-----|---------|----------|
| `<system-reminder>` | System context (bundle, hooks, tools) | Always for system content |
| `<system-reminder source="...">` | Attributed system context | Hooks, tools |
| `<user>` | User's actual message | Wrapping user input |
| `<assistant>` | Assistant's previous response | In conversation history |
| `<tool_use>` | Claude's tool call request | In assistant responses |
| `<tool_result>` | Tool execution result | After tool execution |
| `<context_file>` | Developer-injected context (@mentions) | From developer role messages |
| `<tools>` | Tool definitions JSON | Inside tools-context reminder |

### Hook Content Detection

User messages that start with `<system-reminder` are detected as hook-injected content and rendered **outside** `<user>` tags:



This creates cleaner prompt structure:



---

## Tool Call Flow

### 1. Tool Definition Injection

Tools are converted to JSON and injected in the system prompt:



### 2. Claude Emits Tool Calls

Claude responds with tool calls embedded in text:



### 3. Tool Call Extraction

The provider parses tool calls using regex:

[\s\S]*?

**Validation steps:**
1. Skip non-JSON content (documentation text)
2. Parse JSON and extract tool name, id, arguments
3. Validate tool name against `_valid_tool_names`
4. Filter out invalid tools (feed back to Claude next turn)

### 4. Tool Result Formatting

Results are formatted as JSON in `<tool_result>` tags:



---

## Session Caching

The provider automatically tracks Claude session IDs for prompt caching:



Token savings from caching (example session):

| Turn | Cache Created | Cache Read | Savings |
|------|---------------|------------|---------|
| 1 | 24,707 | 0 | - |
| 2 | 1,101 | 27,164 | ~96% |

---

## Defensive Features

### Missing Tool Result Repair

Detects orphaned tool calls (calls without results) and injects synthetic errors:



Prevents infinite loops from context corruption.

### Invalid Tool Filtering

Tools not in `_valid_tool_names` are rejected and feedback is provided:



---

## See Also

- [FEATURE_COVERAGE.md](FEATURE_COVERAGE.md) — Feature comparison with Anthropic API provider
- [Anthropic Tool Use Docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Official tool use documentation
