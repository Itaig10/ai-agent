# AI Agent TUI

A provider-agnostic terminal chat application written in Python. It uses your
local ChatGPT-authenticated Codex installation by default, with an optional
direct OpenAI Responses API provider. The architecture is designed so a LiteLLM
provider can access additional hosted and local models without changing the TUI
or conversation logic.

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
account-default reasoning effort unless `AI_AGENT_EFFORT` is set to `low`,
`medium`, `high`, or `xhigh`. To use the direct OpenAI API instead:

```bash
export AI_AGENT_PROVIDER="openai"
export AI_AGENT_MODEL="gpt-5.5"
export AI_AGENT_EFFORT="medium"
export OPENAI_API_KEY="your-api-key"
```

To use the LiteLLM Python SDK, choose a provider-prefixed model and set that
provider's normal credential environment variable:

```bash
export AI_AGENT_PROVIDER="litellm"
export AI_AGENT_MODEL="anthropic/claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="your-api-key"
```

For a custom endpoint or LiteLLM proxy, optionally set
`AI_AGENT_LITELLM_API_BASE` and `AI_AGENT_LITELLM_API_KEY`. Set
`AI_AGENT_CONTEXT_WINDOW` to the selected model's context size to enable the
context percentage in the status bar.

Run the application:

```bash
ai-agent
```

Inside the app, enter a message and press **Enter**. Use **Ctrl+C** to quit and
**Ctrl+L** to clear the current conversation. The lower workspace has **Tools**,
**Plan**, **Diff**, and **Notifications** tabs; jump to them with **Ctrl+T**,
**Ctrl+P**, **Ctrl+D**, and **Ctrl+N**.

Responses stream into a temporary assistant panel as they are generated. Use
the provider and model controls at the top (or `/provider` and `/model`) to
switch runtimes without restarting the TUI. The existing conversation is
preserved and is supplied to the new provider.

Named conversations are stored as JSON under `.ai-agent/conversations/`, which
is ignored by Git. Use `/save NAME`, `/load NAME`, and `/sessions` to manage
them. Credentials are never written to conversation files.

Use the **Effort** dropdown beside the prompt to change reasoning effort for
subsequent turns without restarting the app.

The **Tool activity** panel shows commands and file changes as they start and
finish, including command output, exit status, and duration. Actions requiring
more access pause the turn and open an approval dialog with **Allow once**,
**Allow for session**, **Deny**, and **Cancel turn** choices. Pressing **Esc**
denies the request. The same choices are available from the keyboard with
**A**, **S**, **D**, and **C**.

Persistent command-permission rules can allow, ask about, or deny exact command
prefixes before execution. For example, `/permission allow git status` permits
`git status` and its additional arguments, while `/permission deny git push`
blocks pushes. The longest matching argument prefix wins; equally specific deny
rules take precedence. `/permissions` shows the active rules and default policy.
These rules are stored in `.ai-agent/command-policy.json` and are enforced for
native agent command approvals, `/bg`, and autofix test commands. Compound shell
expressions and active shell expansions always require manual approval, even if
their leading command matches an allow rule. The default `ask` policy also
prompts before directly entered `/bg` and autofix commands.

Every foreground command, `/bg` command, and autofix test run also gets a
persistent row in the command-output area below the activity log. Running and
failed commands are expanded automatically; successful commands collapse to a
status-and-duration header. Select a row and press **Enter**, or click it, to
expand or collapse its full output. Displayed output is bounded by the current
output limit and rendered as literal text so terminal markup cannot alter the
interface.

The Codex provider has live web search enabled and displays each search in the
tool activity tab. The direct OpenAI provider also exposes the native web-search
tool to the model. Web access and citations depend on the selected model and
provider.

For multi-step Codex tasks, the **Plan** tab shows pending, active, and completed
steps as they change. The **Diff** tab shows the latest turn-level unified diff
with additions and deletions highlighted for review.

Each non-empty turn diff is also saved as a workspace-local checkpoint under
`.ai-agent/checkpoints/`. `/undo` safely reverses the latest active checkpoint
after first verifying that the patch still matches the working tree. It refuses
to overwrite conflicting later edits. Use `/checkpoints` to see checkpoint IDs
and `/restore ID` to reapply an undone checkpoint.

The safe patch editor applies externally prepared unified diffs in two phases.
Run `/patch preview FILE` to validate paths, size, and worktree compatibility
before seeing the patch in the **Diff** tab. `/patch apply` asks for confirmation,
rechecks for intervening conflicts, applies the exact preview, and validates
Python, JSON, and TOML syntax. A validation failure is rolled back automatically;
a successful patch creates a normal `/undo` checkpoint. Use `/patch discard` or
`/patch status` to manage the pending preview.

`/autofix` runs the detected project test suite and gives a fresh implementation
agent a bounded number of repair attempts. The default is two fixes and a
five-minute timeout per test run. Customize it with
`/autofix --retries 3 --timeout 2m`, or provide an explicit argument-safe command
after `--`, such as `/autofix -- python -m unittest tests.test_session`. It never
uses a shell, refuses more than five repair attempts, and keeps subsequent chat
prompts queued until the workflow finishes. Use `/cancel ID` to stop it.

Long commands can run without blocking the chat using `/bg COMMAND [ARGS...]`.
Arguments are executed directly without a shell, so pipes, redirection, and
other shell syntax are intentionally unavailable. Output is streamed into a
bounded 100 KB tail, and cancelling a command terminates its whole process
group. `/bg-agent PROMPT` starts an independent model request with the current
provider and model. Use `/jobs` to inspect jobs and `/cancel ID` to stop a
running one.

Local `/bg` and autofix processes share configurable resource limits. Use
`/limits` to inspect them and commands such as `/limit set timeout 2m`,
`/limit set output 256KB`, `/limit set cpu 30s`, `/limit set memory 1GB`,
`/limit set file-size 100MB`, or `/limit set processes 32` to change them.
Optional CPU, memory, file-size, and process limits accept `off`; `/limit reset`
restores the defaults. Settings persist in `.ai-agent/execution-limits.json`.
Operating-system CPU, memory, file-size, and process caps are available on
POSIX systems. These local limits do not replace the Codex provider's own
sandbox and limits for commands it runs internally.

Specialized subagents can investigate work in parallel through
`/agent ROLE PROMPT`. The built-in roles are `researcher`, `coder`, `reviewer`,
and `tester`; `/agents` describes them. Subagents use an independent provider
session and are deliberately read-only to prevent simultaneous agents from
overwriting each other's work. Completed reports appear in the chat and the
background-job history.

`/review [SCOPE]` starts a fresh, independent provider session that reviews the
current tracked and untracked changes without modifying them. The review checks
correctness, security, regressions, missing tests, and unnecessary complexity,
and reports actionable findings with severity and file locations. An optional
scope such as `/review authentication` focuses the pass.

The prompt remains available while the main agent is working. Additional
messages are placed in a FIFO queue and begin automatically after the current
turn. The status line shows the pending count. `/queue` lists pending prompts,
`/queue remove ID` removes one, and `/queue clear` removes all of them. Use
`/enqueue PROMPT` to add an item explicitly.

Persistent goals let the agent continue taking and verifying steps until an
objective is complete. Start one with `/goal start --timeout 30m OBJECTIVE`;
durations accept seconds (`s`), minutes (`m`), hours (`h`), or days (`d`). The
timeout defaults to 30 minutes and stops further attempts when reached. Goal
status, attempt count, latest result, and remaining time appear in the Plan tab
and status line. Active goals and progress are saved under `.ai-agent/goals/`
and resume automatically after restarting the app. `/goal stop` ends a goal
manually, while `/goals` lists goal history.
Goal orchestration prompts and status markers stay outside the normal chat
history, and the native provider context is reset when a goal ends.

Reusable project skills are discovered from `skills/NAME/SKILL.md` and optional
local-only `.ai-agent/skills/NAME/SKILL.md` files. `/skills` lists them,
`/skill show NAME` displays one, and `/skill run NAME TASK` applies it to a
request. Skills with `auto_activate: true` are selected when a configured trigger
appears as a complete word or phrase in a prompt. Skill instructions are sent to
the provider but are not stored as the user's conversation message. Enable and
disable choices persist in `.ai-agent/skills.json`; use `/skills reload` after
editing definitions.

Each `SKILL.md` uses a small, validated YAML-frontmatter subset:

```markdown
---
name: testing
description: Design focused regression tests
triggers: [add tests, test coverage]
auto_activate: false
---
Inspect the implementation and existing tests before making changes.
```

Supporting scripts and resources can live beside `SKILL.md`; the skill prompt
supplies that directory to the provider. Skill files are size-limited, cannot
escape their configured root, and invalid definitions are reported without
preventing valid skills from loading. A starter `testing` skill is included.

The **Notifications** tab records approval requests, checkpoint events, agent
errors, and background-job completion or failure. Important events also appear
as temporary TUI notifications.

Typing `/` in the prompt shows inline, case-insensitive completions for commands
and live values such as installed skills, registered tools, available themes,
permission actions, and resource-limit names. The list updates immediately when
those registries change.

Use `/themes` and `/theme NAME` to inspect and select Textual color themes.
`/accessibility` reports display preferences; `contrast on|off`,
`motion full|reduced`, and `density compact|comfortable` adjust contrast,
animation, and spacing. Appearance settings persist in
`.ai-agent/appearance.json`.

The bottom status line displays the provider, model, and exact context-window
usage. When usage reaches 80%, the Codex provider automatically compacts the
thread and shows the number of completed compactions in the status line.
After the first request it also shows session token usage, average latency, and
cost when the provider reports one. `/metrics` shows the expanded input/output
breakdown. Cost is normally available through LiteLLM; Codex and direct OpenAI
show it as unavailable rather than guessing from a stale price table.

## Slash commands

```text
/save NAME                 Save this conversation
/load NAME                 Load a saved conversation
/sessions                  List saved conversations
/provider NAME [MODEL]     Switch provider and optionally model
/model MODEL               Change model on the current provider
/tools                     Show the tool registry
/tool enable|disable NAME  Toggle a registered tool
/permissions               Show command rules and the default policy
/permission default ACTION Set the default allow, ask, or deny action
/permission ACTION COMMAND Add an argument-prefix command rule
/permission remove ID      Remove a command-permission rule
/permission reset          Reset command permissions
/limits                    Show local command resource limits
/limit set NAME VALUE      Set timeout, output, CPU, memory, file, or process cap
/limit reset               Restore default resource limits
/themes                    List installed TUI themes
/theme NAME                Select and persist a TUI theme
/accessibility [NAME VALUE]
                            Show or change contrast, motion, and density
/skills [reload]           List or reload project skills
/skill show NAME           Show skill metadata and instructions
/skill run NAME TASK       Run a task with a named skill
/skill enable|disable NAME Persistently toggle a skill
/metrics                   Show detailed session usage
/plan                      Open the live plan tab
/diff                      Open the diff review tab
/undo                      Safely reverse the latest checkpoint
/checkpoints               List active and undone checkpoints
/restore NAME              Reapply an undone checkpoint
/patch preview FILE        Validate and show a unified diff
/patch apply               Confirm, apply, validate, and checkpoint it
/patch discard             Discard the pending patch preview
/patch status              Show the pending patch summary
/autofix [OPTIONS] [-- COMMAND]
                            Run bounded automatic test-and-fix attempts
/review [SCOPE]             Independently review workspace changes
/bg COMMAND [ARGS...]      Run a command in the background
/bg-agent PROMPT           Run an independent agent request
/agent ROLE PROMPT         Delegate work to a specialist subagent
/agents                    List available subagent roles
/jobs                      List background jobs and states
/cancel ID                 Cancel a running background job
/enqueue PROMPT             Add a prompt to the FIFO queue
/queue                     List queued prompts
/queue remove ID           Remove one queued prompt
/queue clear               Remove all queued prompts
/goal start [--timeout 30m] OBJECTIVE
                            Start a persistent, time-limited goal
/goal status               Show current goal progress and remaining time
/goal stop                 Stop the active goal
/goals                     List saved goal history
/notifications             Open the notification center
/clear                     Clear the conversation
/help                      Show command help
```

## Tool plugins

`ToolRegistry` contains the built-in filesystem, command, Git, and web-search
capabilities. New client-side Codex tools can be registered with a name,
description, JSON input schema, and async handler. Enabled handlers are exposed
through the Codex dynamic-tool protocol and require user approval by default.
Use `requires_approval=False` only for handlers that are safe to invoke without
confirmation.

Codex implements filesystem, command, and Git access through one native
workspace-execution runtime. For hard enforcement, toggling any one of those
entries toggles all three together and restarts the ephemeral Codex context.
Web search and client-side plugins remain independently toggleable.

## Development

Run the dependency-free unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Architecture

```text
TUI -> ChatSession -> ChatProvider -> Codex app server (default)
                         |-------> OpenAI Responses API
                         `-------> LiteLLM Python SDK
```

Provider implementations live in `src/ai_agent/providers/`. A provider only
needs to implement the `ChatProvider` protocol. The Codex provider creates an
ephemeral session with filesystem tools enabled. The agent can list, search,
read, create, and edit files inside the directory where `ai-agent` is started.
It can also run non-destructive local commands such as tests, linters,
formatters, and project scripts. Git integration supports status, diffs, logs,
file history, and branch inspection. Codex starts in a read-only sandbox, so
file writes, Git mutations, destructive commands, package installation, and
network access require a blocking approval decision. Paths outside the workspace
remain unavailable.

Filesystem, command, Git, activity, approval, live-plan, and diff events
currently belong to the Codex provider. The direct OpenAI provider also supports
native web search. LiteLLM currently supports chat, reasoning-effort selection,
and usage reporting; provider-specific tool calling is not yet enabled.
