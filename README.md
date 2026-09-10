# AI Agent TUI

A provider-agnostic terminal chat application written in Python. The first
provider uses the OpenAI Responses API; the architecture is designed so a
LiteLLM provider can be added later without changing the TUI or conversation
logic.

## Requirements

- Python 3.11 or newer
- An OpenAI API key

A ChatGPT subscription and an OpenAI API account are separate. Create an API
key in the OpenAI Platform and keep it out of Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
export OPENAI_API_KEY="your-api-key"
```

Optionally choose a model or provider:

```bash
export AI_AGENT_MODEL="gpt-5.5"
export AI_AGENT_PROVIDER="openai"
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
TUI -> ChatSession -> ChatProvider -> OpenAI Responses API
                              `----> LiteLLM (future adapter)
```

Provider implementations live in `src/ai_agent/providers/`. A provider only
needs to implement the `ChatProvider` protocol.
