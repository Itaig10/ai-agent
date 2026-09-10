# AI Agent TUI

A provider-agnostic terminal chat application written in Python. It uses your
local ChatGPT-authenticated Codex installation by default, with an optional
direct OpenAI Responses API provider. The architecture is designed so a LiteLLM
provider can be added later without changing the TUI or conversation logic.

## Requirements

- Python 3.11 or newer
- Codex installed and authenticated with ChatGPT

Check the local Codex login:

```bash
codex login status
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Codex uses the account-default model unless `AI_AGENT_MODEL` is set. To use the
direct OpenAI API instead:

```bash
export AI_AGENT_PROVIDER="openai"
export AI_AGENT_MODEL="gpt-5.5"
export OPENAI_API_KEY="your-api-key"
```

Run the application:

```bash
ai-agent
```

Inside the app, enter a message and press **Enter**. Use **Ctrl+C** to quit and
**Ctrl+L** to clear the current conversation.

## Development

Run the dependency-free unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Architecture

```text
TUI -> ChatSession -> ChatProvider -> Codex app server (default)
                         |-------> OpenAI Responses API
                         `-------> LiteLLM (future adapter)
```

Provider implementations live in `src/ai_agent/providers/`. A provider only
needs to implement the `ChatProvider` protocol. The Codex provider creates an
ephemeral session with a read-only sandbox and no command approvals.
