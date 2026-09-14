import asyncio
import inspect
import os
import re
import shlex
import signal
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from ai_agent.appearance import AppearanceStore
from ai_agent.background import BackgroundJob, BackgroundTaskManager
from ai_agent.checkpoints import CheckpointStore
from ai_agent.conversations import ConversationStore
from ai_agent.events import (
    ApprovalRequest,
    DiffUpdate,
    NotificationEvent,
    PlanUpdate,
    TextDelta,
    ToolActivity,
)
from ai_agent.execution import (
    CommandPolicyStore,
    ExecutionLimitStore,
    native_command_is_simple,
)
from ai_agent.goals import (
    Goal,
    GoalStore,
    build_goal_prompt,
    clean_goal_reply,
    goal_reply_complete,
    parse_timeout,
)
from ai_agent.messages import ChatMessage
from ai_agent.patches import PatchManager
from ai_agent.prompt_queue import PromptQueue
from ai_agent.providers.base import ChatProvider
from ai_agent.quality import (
    AutoFixWorkflow,
    build_review_prompt,
    collect_workspace_diff,
    parse_autofix_args,
)
from ai_agent.session import ChatSession
from ai_agent.skills import SkillRegistry, build_skill_prompt
from ai_agent.subagents import SUBAGENT_ROLES, build_subagent_prompt
from ai_agent.suggestions import SlashCommandSuggester
from ai_agent.tools import ToolRegistry


ProviderFactory = Callable[[str, str | None], ChatProvider]
ModelFetcher = Callable[[str], Awaitable[Sequence[str]]]


class ApprovalScreen(ModalScreen[str]):
    """Modal decision prompt for a provider tool request."""

    BINDINGS = [
        ("a", "accept", "Allow once"),
        ("s", "accept_session", "Allow session"),
        ("d", "deny", "Deny"),
        ("c", "cancel_turn", "Cancel turn"),
        ("escape", "deny", "Deny"),
    ]
    CSS = """
    ApprovalScreen {
        align: center middle;
        background: $background 70%;
    }

    #approval-dialog {
        width: 76;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    #approval-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #approval-detail, #approval-reason {
        height: auto;
        margin-bottom: 1;
    }

    #approval-actions {
        height: auto;
        align-horizontal: right;
    }

    #approval-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static(self.request.title, id="approval-title", markup=False)
            yield Static(self.request.detail, id="approval-detail", markup=False)
            if self.request.reason:
                yield Static(
                    f"Reason: {self.request.reason}",
                    id="approval-reason",
                    markup=False,
                )
            with Horizontal(id="approval-actions"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Cancel turn", id="cancel")
                yield Button("Allow once", id="accept", variant="success")
                yield Button(
                    "Allow for session",
                    id="accept-session",
                    variant="primary",
                )

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        decisions = {
            "deny": "decline",
            "cancel": "cancel",
            "accept": "accept",
            "accept-session": "acceptForSession",
        }
        self.dismiss(decisions[event.button.id or "deny"])

    def action_deny(self) -> None:
        self.dismiss("decline")

    def action_accept(self) -> None:
        self.dismiss("accept")

    def action_accept_session(self) -> None:
        self.dismiss("acceptForSession")

    def action_cancel_turn(self) -> None:
        self.dismiss("cancel")


class AgentApp(App[None]):
    """Terminal user interface for an AI chat session."""

    TITLE = "AI Agent"
    SUB_TITLE = "Provider-agnostic terminal assistant"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+t", "show_tools", "Tools"),
        ("ctrl+p", "show_plan", "Plan"),
        ("ctrl+d", "show_diff", "Diff"),
        ("ctrl+n", "show_notifications", "Notifications"),
        ("ctrl+m", "focus_model", "Model"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #chat {
        height: 2fr;
        margin: 0 1;
        padding: 1 2;
        border: round $primary;
    }

    #streaming {
        display: none;
        height: auto;
        max-height: 8;
        margin: 0 1;
        padding: 1 2;
        border: round $accent;
    }

    #runtime-controls {
        height: 3;
        margin: 0 1;
    }

    #provider-label, #model-label {
        width: auto;
        padding: 1 1 0 0;
    }

    #provider {
        width: 16;
        margin-right: 2;
    }

    #model {
        width: 1fr;
    }

    #switch-provider {
        width: 12;
        margin-left: 1;
    }

    #work-panel {
        height: 1fr;
        min-height: 7;
        margin: 0 1;
    }

    #tools, #plan, #diff, #notifications {
        height: 1fr;
        padding: 0 1;
    }

    #tools {
        min-height: 4;
    }

    #command-outputs {
        height: 1fr;
        min-height: 5;
        padding: 0 1;
        border-top: solid $primary-darken-2;
    }

    #command-output-help {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    .command-output-body {
        height: auto;
        color: $text;
    }

    .-high-contrast #chat, .-high-contrast #work-panel,
    .-high-contrast #streaming {
        border: heavy $foreground;
    }

    .-high-contrast #status {
        background: $foreground;
        color: $background;
        text-style: bold;
    }

    .-compact #chat, .-compact #tools, .-compact #plan,
    .-compact #diff, .-compact #notifications {
        padding: 0 1;
    }

    #prompt-container {
        height: auto;
        margin: 0 1 1 1;
    }

    #prompt {
        width: 1fr;
    }

    #effort-label {
        width: auto;
        height: 3;
        padding: 1 1 0 2;
    }

    #effort {
        width: 18;
    }
    """

    def __init__(
        self,
        session: ChatSession,
        *,
        provider_factory: ProviderFactory | None = None,
        model_fetcher: ModelFetcher | None = None,
        tool_registry: ToolRegistry | None = None,
        conversation_store: ConversationStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        patch_manager: PatchManager | None = None,
        quality_workflow: AutoFixWorkflow | None = None,
        prompt_queue: PromptQueue | None = None,
        goal_store: GoalStore | None = None,
        skill_registry: SkillRegistry | None = None,
        command_policy: CommandPolicyStore | None = None,
        limit_store: ExecutionLimitStore | None = None,
        appearance_store: AppearanceStore | None = None,
    ) -> None:
        super().__init__()
        self.session = session
        self.provider_factory = provider_factory
        self.model_fetcher = model_fetcher
        self._available_models: dict[str, tuple[str, ...]] = {}
        self.tool_registry = (
            tool_registry
            or getattr(session.provider, "tool_registry", None)
            or ToolRegistry()
        )
        self.conversation_store = conversation_store or ConversationStore()
        self.checkpoint_store = checkpoint_store or CheckpointStore()
        self.patch_manager = patch_manager or PatchManager()
        self.command_policy = command_policy or CommandPolicyStore()
        self.limit_store = limit_store or ExecutionLimitStore()
        self.appearance_store = appearance_store or AppearanceStore()
        self.quality_workflow = quality_workflow or AutoFixWorkflow(
            limits=self.limit_store.limits
        )
        self.prompt_queue = prompt_queue or PromptQueue()
        self.goal_store = goal_store or GoalStore()
        self.skill_registry = skill_registry or SkillRegistry()
        self.background = BackgroundTaskManager(self._background_finished)
        self._streamed_text = ""
        self._active_diff = ""
        self._shutting_down = False
        self._busy = False
        self._active_goal: Goal | None = None
        self._goal_worker = None
        self._command_cards: dict[str, tuple[Collapsible, Static]] = {}
        self._session_command_allows: set[tuple[str, ...]] = set()
        self._default_animation_level = self.animation_level
        self._bind_provider_handlers()

    def _bind_provider_handlers(self) -> None:
        self._bind_handlers(self.session.provider)

    def _bind_handlers(self, provider: ChatProvider) -> None:
        set_handlers = getattr(provider, "set_event_handlers", None)
        if set_handlers is not None:
            handlers = {
                "activity": self._show_activity,
                "approval": self._request_approval,
                "plan": self._show_plan,
                "diff": self._show_diff,
                "text_delta": self._show_text_delta,
            }
            parameters = inspect.signature(set_handlers).parameters
            accepts_any = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            set_handlers(
                **{
                    name: handler
                    for name, handler in handlers.items()
                    if accepts_any or name in parameters
                }
            )

    def compose(self) -> ComposeResult:
        provider_options = [
            ("Codex", "codex"),
            ("OpenAI", "openai"),
            ("LiteLLM", "litellm"),
        ]
        if self.session.provider.name not in {value for _, value in provider_options}:
            provider_options.append(
                (self.session.provider.name.title(), self.session.provider.name)
            )
        yield Header()
        with Horizontal(id="runtime-controls"):
            yield Label("Provider:", id="provider-label")
            yield Select(
                provider_options,
                value=self.session.provider.name,
                allow_blank=False,
                id="provider",
            )
            yield Label("Model:", id="model-label")
            yield Input(
                value=(
                    ""
                    if self.session.provider.model == "account default"
                    else self.session.provider.model
                ),
                placeholder="account default",
                id="model",
            )
            yield Button(
                "Switch",
                id="switch-provider",
                variant="primary",
                disabled=self.provider_factory is None,
            )
        yield RichLog(id="chat", wrap=True, markup=True)
        yield Static("", id="streaming", markup=True)
        with TabbedContent(id="work-panel"):
            with TabPane("Tools", id="tools-tab"):
                yield RichLog(id="tools", wrap=True, markup=True)
                with VerticalScroll(id="command-outputs"):
                    yield Static(
                        "Command output — select a row or press Enter to "
                        "expand/collapse",
                        id="command-output-help",
                    )
            with TabPane("Plan", id="plan-tab"):
                yield RichLog(id="plan", wrap=True, markup=True)
            with TabPane("Diff", id="diff-tab"):
                yield RichLog(id="diff", wrap=False, markup=True)
            with TabPane("Notifications", id="notifications-tab"):
                yield RichLog(id="notifications", wrap=True, markup=True)
        with Horizontal(id="prompt-container"):
            yield Input(
                placeholder="Ask anything… (/ for commands)",
                suggester=SlashCommandSuggester(self._slash_suggestions),
                id="prompt",
            )
            yield Label("Effort:", id="effort-label")
            yield Select(
                [
                    ("Default", "default"),
                    ("Low", "low"),
                    ("Medium", "medium"),
                    ("High", "high"),
                    ("XHigh", "xhigh"),
                ],
                value=getattr(self.session.provider, "effort", None) or "default",
                allow_blank=False,
                id="effort",
            )
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._apply_appearance()
        for label, store in (
            ("Command permissions", self.command_policy),
            ("Execution limits", self.limit_store),
            ("Appearance settings", self.appearance_store),
        ):
            if store.load_error:
                self._record_notification(
                    NotificationEvent(
                        f"{label} reset",
                        f"{store.load_error}. Safe defaults are active.",
                        "warning",
                    )
                )
        self.query_one("#chat", RichLog).write(
            "[bold cyan]AI Agent[/bold cyan]\nEnter a message to begin."
        )
        self.query_one("#tools", RichLog).write(
            "[bold]Tool activity[/bold] "
            "[dim]Commands and file changes appear here.[/dim]\n"
            f"[dim]{self.skill_registry.enabled_count} project skill(s) enabled.[/dim]"
        )
        self.query_one("#plan", RichLog).write(
            "[bold]Task plan[/bold] [dim]Multi-step work appears here.[/dim]"
        )
        self.query_one("#diff", RichLog).write(
            "[bold]Review[/bold] [dim]File changes appear here as a unified diff.[/dim]"
        )
        self.query_one("#notifications", RichLog).write(
            "[bold]Notifications[/bold] "
            "[dim]Approvals and background-job results appear here.[/dim]"
        )
        self.query_one("#prompt", Input).focus()
        if (
            self.model_fetcher is not None
            and self.session.provider.name == "litellm"
        ):
            self.fetch_models(self.session.provider.name)
        active_goal = self.goal_store.active()
        if active_goal is not None:
            if active_goal.remaining_seconds <= 0:
                active_goal.status = "timed_out"
                self.goal_store.save(active_goal)
                self._record_notification(
                    NotificationEvent(
                        "Goal timed out",
                        f"{active_goal.id} expired while the app was closed.",
                        "warning",
                    )
                )
            else:
                self._start_goal(active_goal, resumed=True)

    def _apply_appearance(self) -> None:
        settings = self.appearance_store.settings
        if settings.theme not in self.available_themes:
            unavailable = settings.theme
            fallback = (
                "textual-dark"
                if "textual-dark" in self.available_themes
                else next(iter(self.available_themes))
            )
            try:
                self.appearance_store.update("theme", fallback)
            except OSError as error:
                settings.theme = fallback
                self._record_notification(
                    NotificationEvent(
                        "Theme fallback could not be saved",
                        str(error),
                        "warning",
                    )
                )
            self._record_notification(
                NotificationEvent(
                    "Theme unavailable",
                    f"{unavailable!r} is unavailable; using {fallback!r}.",
                    "warning",
                )
            )
        self.theme = settings.theme
        self.set_class(settings.high_contrast, "-high-contrast")
        self.set_class(settings.density == "compact", "-compact")
        self.animation_level = (
            "none" if settings.reduced_motion else self._default_animation_level
        )

    def _slash_suggestions(self) -> list[str]:
        commands = [
            "/accessibility",
            "/agent ",
            "/agents",
            "/autofix ",
            "/bg ",
            "/bg-agent ",
            "/cancel ",
            "/checkpoints",
            "/clear",
            "/diff",
            "/enqueue ",
            "/goal start --timeout 30m ",
            "/goal status",
            "/goal stop",
            "/goals",
            "/help",
            "/jobs",
            "/limit reset",
            "/limit set ",
            "/limits",
            "/load ",
            "/metrics",
            "/model ",
            "/models",
            "/models refresh",
            "/notifications",
            "/patch apply",
            "/patch discard",
            "/patch preview ",
            "/patch status",
            "/permission default ask",
            "/permission remove ",
            "/permission reset",
            "/permissions",
            "/plan",
            "/provider codex",
            "/queue",
            "/queue clear",
            "/queue remove ",
            "/restore ",
            "/review ",
            "/save ",
            "/sessions",
            "/skills",
            "/skills reload",
            "/theme ",
            "/themes",
            "/tools",
            "/undo",
        ]
        commands.extend(
            f"/tool {action} {plugin.name}"
            for plugin, _, _ in self.tool_registry.entries()
            for action in ("enable", "disable")
        )
        commands.extend(
            f"/skill {action} {skill.name}{' ' if action == 'run' else ''}"
            for skill, _ in self.skill_registry.entries()
            for action in ("show", "run", "enable", "disable")
        )
        commands.extend(f"/theme {theme}" for theme in self.available_themes)
        commands.extend(
            f"/limit set {name} " for name in self.limit_store.NAMES
        )
        commands.extend(
            f"/permission {action} " for action in ("allow", "ask", "deny")
        )
        return commands

    @on(Input.Submitted, "#prompt")
    async def submit_prompt(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        event.input.clear()
        if prompt.startswith("/"):
            await self._handle_command(prompt)
            return
        if self._busy:
            item = self.prompt_queue.enqueue(prompt)
            self._write_system(f"Queued prompt {item.id}: {item.text}")
            self.query_one("#status", Static).update(self._status_text())
            return
        self._start_user_prompt(prompt)

    def _start_user_prompt(
        self,
        prompt: str,
        *,
        queue_id: int | None = None,
    ) -> None:
        skill = self.skill_registry.match(prompt)
        provider_prompt = None
        if skill is not None:
            provider_prompt = build_skill_prompt(skill, prompt)
            self._write_system(f"Auto-activated skill {skill.name!r}.")
        self._start_prompt(
            prompt,
            provider_prompt=provider_prompt,
            queue_id=queue_id,
        )

    def _start_prompt(
        self,
        prompt: str,
        *,
        provider_prompt: str | None = None,
        queue_id: int | None = None,
    ) -> None:
        self._busy = True
        self.query_one("#switch-provider", Button).disabled = True
        chat = self.query_one("#chat", RichLog)
        if queue_id is not None:
            chat.write(f"\n[dim]Starting queued prompt {queue_id}[/dim]")
        chat.write(f"\n[bold green]You[/bold green]\n{escape(prompt)}")
        self._streamed_text = ""
        self._active_diff = ""
        streaming = self.query_one("#streaming", Static)
        streaming.update("[bold cyan]Assistant[/bold cyan]\n[dim]Thinking…[/dim]")
        streaming.styles.display = "block"
        self.query_one("#status", Static).update(self._status_text())
        self.get_reply(prompt, provider_prompt)

    @work
    async def get_reply(self, prompt: str, provider_prompt: str | None = None) -> None:
        chat = self.query_one("#chat", RichLog)
        prompt_input = self.query_one("#prompt", Input)

        try:
            reply = await self.session.send(prompt, provider_prompt=provider_prompt)
            chat.write(f"\n[bold cyan]Assistant[/bold cyan]\n{escape(reply)}")
        except Exception as error:
            chat.write(f"\n[bold red]Error[/bold red]\n{error}")
            self._record_notification(
                NotificationEvent("Agent request failed", str(error), "error")
            )
        finally:
            self._save_active_checkpoint()
            self.query_one("#streaming", Static).styles.display = "none"
            self._busy = False
            if self.prompt_queue.list() and not self._shutting_down:
                self._start_next_queued_prompt()
            else:
                self.query_one("#switch-provider", Button).disabled = (
                    self.provider_factory is None
                )
                self.query_one("#status", Static).update(self._status_text())
            prompt_input.focus()

    def _start_next_queued_prompt(self) -> None:
        item = self.prompt_queue.pop()
        if item is None:
            return
        self._record_notification(
            NotificationEvent("Queued prompt started", f"Prompt {item.id}")
        )
        self._start_user_prompt(item.text, queue_id=item.id)

    def _status_text(self) -> str:
        provider = self.session.provider
        effort = getattr(provider, "effort", None) or "default"
        supports_context = getattr(
            provider,
            "context_window_supported",
            hasattr(provider, "context_percent"),
        )
        percent = getattr(provider, "context_percent", None)
        if not supports_context:
            context = "unavailable"
        else:
            context = "measuring…" if percent is None else f"{percent:.1f}%"
        compacted = getattr(provider, "compaction_count", 0)
        compact_text = f" | Compacted: {compacted}" if compacted else ""
        threshold = getattr(provider, "auto_compact_threshold", None)
        auto_compact = (
            f" | Auto-compact: {threshold:.0%}" if threshold is not None else ""
        )
        tls = (
            " | TLS: verification disabled"
            if getattr(provider, "verify_tls", True) is False
            else ""
        )
        queue = f" | Queue: {len(self.prompt_queue)}" if self.prompt_queue else ""
        skills = f" | Skills: {self.skill_registry.enabled_count}"
        goal = (
            f" | Goal: {self._active_goal.attempts + 1}"
            if self._active_goal is not None
            and self._active_goal.status == "active"
            else ""
        )
        return (
            f"Provider: {provider.name} | Model: {provider.model} | Effort: {effort} | "
            f"Context: {context}{compact_text}{auto_compact}{tls}{goal}{queue}{skills}"
            f"{self._metrics_text()}"
        )

    def _metrics_text(self) -> str:
        metrics = getattr(self.session.provider, "session_metrics", None)
        if metrics is None or metrics.requests == 0:
            return ""
        cost = (
            f" | Cost: ${metrics.cost_usd:.4f}"
            if metrics.cost_usd is not None
            else ""
        )
        return (
            f" | Tokens: {metrics.total_tokens:,}"
            f" | Avg: {metrics.average_latency_ms}ms{cost}"
        )

    @on(Select.Changed, "#effort")
    async def change_effort(self, event: Select.Changed) -> None:
        selected = str(event.value)
        effort = None if selected == "default" else selected
        setter = getattr(self.session.provider, "set_effort", None)
        if setter is None:
            self.notify(
                "This provider does not support reasoning effort",
                severity="error",
            )
            return

        select = self.query_one("#effort", Select)
        select.disabled = True
        try:
            await setter(effort)
            self.query_one("#status", Static).update(self._status_text())
            self.notify(f"Reasoning effort: {selected}")
        except Exception as error:
            self.notify(str(error), severity="error")
        finally:
            select.disabled = False

    async def _show_activity(self, activity: ToolActivity) -> None:
        icons = {
            "running": "▶",
            "inProgress": "▶",
            "completed": "✓",
            "failed": "✗",
            "declined": "⊘",
            "approval": "?",
            "accept": "✓",
            "acceptForSession": "✓",
            "decline": "⊘",
            "cancel": "■",
        }
        icon = icons.get(activity.status, "•")
        duration = (
            f" [dim]({activity.duration_ms / 1000:.2f}s)[/dim]"
            if activity.duration_ms is not None
            else ""
        )
        detail = escape(activity.detail)
        line = f"{icon} [bold]{escape(activity.title)}[/bold]{duration}"
        if detail:
            line += f"\n  [dim]{detail}[/dim]"
        if activity.output:
            output = escape(activity.output[-1200:])
            line += f"\n  {output}"
        self.query_one("#tools", RichLog).write(line)
        if activity.kind == "command" and activity.status in {
            "cancelled",
            "completed",
            "failed",
            "inProgress",
            "running",
        }:
            await self._upsert_command_output(activity)

    async def _upsert_command_output(
        self,
        activity: ToolActivity,
        *,
        key: str | None = None,
    ) -> None:
        normalized_detail = re.sub(
            r"\s+\[exit\s+-?\d+\]\s*$",
            "",
            activity.detail,
        )
        key = key or f"{activity.kind}:{activity.title}:{normalized_detail}"
        icon = {
            "running": "▶",
            "inProgress": "▶",
            "completed": "✓",
            "failed": "✗",
            "cancelled": "■",
            "declined": "⊘",
        }.get(activity.status, "•")
        duration = (
            f" ({activity.duration_ms / 1000:.2f}s)"
            if activity.duration_ms is not None
            else ""
        )
        title = f"{icon} {activity.title} — {activity.status}{duration}"
        output_limit = self.limit_store.limits.output_bytes
        output = activity.output[-output_limit:]
        if len(activity.output) > len(output):
            output = "[earlier output truncated]\n" + output
        body_text = f"Command:\n{activity.detail or activity.title}"
        if output:
            body_text += f"\n\nOutput:\n{output}"
        elif activity.status in {"running", "inProgress"}:
            body_text += "\n\nOutput will appear when available."
        card_entry = self._command_cards.get(key)
        collapsed = activity.status == "completed"
        if card_entry is None:
            body = Static(body_text, markup=False, classes="command-output-body")
            card = Collapsible(
                body,
                title=title,
                collapsed=collapsed,
                classes="command-output",
            )
            await self.query_one("#command-outputs", VerticalScroll).mount(card)
            self._command_cards[key] = (card, body)
        else:
            card, body = card_entry
            card.title = title
            card.collapsed = collapsed
            body.update(body_text)

    async def _request_approval(self, request: ApprovalRequest) -> str:
        if request.kind == "command":
            source = str(request.detail)
            try:
                command = shlex.split(source)
            except ValueError:
                command = []
            if command:
                decision = self.command_policy.evaluate(command)
                if decision.action == "deny":
                    rule = f" by rule {decision.rule.id}" if decision.rule else ""
                    self._record_notification(
                        NotificationEvent(
                            "Command blocked",
                            f"{request.detail}{rule}",
                            "warning",
                        )
                    )
                    return "decline"
                if decision.action == "allow" and native_command_is_simple(source):
                    rule = f" by rule {decision.rule.id}" if decision.rule else ""
                    self._record_notification(
                        NotificationEvent(
                            "Command allowed",
                            f"{request.detail}{rule}",
                        )
                    )
                    return "accept"
        self._record_notification(
            NotificationEvent("Approval required", request.title, "warning")
        )
        return await self.push_screen_wait(ApprovalScreen(request))

    async def _authorize_direct_command(
        self,
        command: list[str],
        label: str,
    ) -> None:
        decision = self.command_policy.evaluate(command)
        if decision.action == "deny":
            rule = f" by rule {decision.rule.id}" if decision.rule else ""
            raise ValueError(f"Blocked {label}{rule}: {shlex.join(command)}")
        if decision.action == "allow":
            return
        normalized = tuple(command)
        if normalized in self._session_command_allows:
            return
        reason = (
            f"Matched permission rule {decision.rule.id}."
            if decision.rule is not None
            else "The default command policy requires approval."
        )
        response = await self._request_approval(
            ApprovalRequest(
                kind="direct-command",
                title=f"Approve {label}",
                detail=shlex.join(command),
                reason=reason,
            )
        )
        if response not in {"accept", "acceptForSession"}:
            raise ValueError(f"Cancelled {label}: {shlex.join(command)}")
        if response == "acceptForSession":
            self._session_command_allows.add(normalized)

    async def _show_text_delta(self, delta: TextDelta) -> None:
        self._streamed_text += delta.text
        self.query_one("#streaming", Static).update(
            f"[bold cyan]Assistant[/bold cyan]\n{escape(self._streamed_text)}"
        )

    async def _show_plan(self, update: PlanUpdate) -> None:
        plan = self.query_one("#plan", RichLog)
        plan.clear()
        if update.explanation:
            plan.write(f"[dim]{escape(update.explanation)}[/dim]\n")
        if not update.steps:
            plan.write("[dim]No active plan.[/dim]")
            return
        icons = {"pending": "○", "inProgress": "◐", "completed": "●"}
        styles = {
            "pending": "dim",
            "inProgress": "bold yellow",
            "completed": "green",
        }
        for step in update.steps:
            icon = icons.get(step.status, "○")
            style = styles.get(step.status, "")
            text = f"{icon} {escape(step.text)}"
            plan.write(f"[{style}]{text}[/{style}]" if style else text)

    async def _show_diff(self, update: DiffUpdate) -> None:
        self._active_diff = update.diff
        self._render_diff(update.diff)

    def _render_diff(self, diff: str) -> None:
        review = self.query_one("#diff", RichLog)
        review.clear()
        if not diff.strip():
            review.write("[dim]No file changes in this turn.[/dim]")
            return
        for line in diff.splitlines():
            escaped = escape(line)
            if line.startswith("+++") or line.startswith("---"):
                review.write(f"[bold]{escaped}[/bold]")
            elif line.startswith("+"):
                review.write(f"[green]{escaped}[/green]")
            elif line.startswith("-"):
                review.write(f"[red]{escaped}[/red]")
            elif line.startswith("@@"):
                review.write(f"[cyan]{escaped}[/cyan]")
            else:
                review.write(escaped)

    def _select_work_tab(self, tab_id: str) -> None:
        self.query_one("#work-panel", TabbedContent).active = tab_id

    def action_show_tools(self) -> None:
        self._select_work_tab("tools-tab")

    def action_show_plan(self) -> None:
        self._select_work_tab("plan-tab")

    def action_show_diff(self) -> None:
        self._select_work_tab("diff-tab")

    def action_show_notifications(self) -> None:
        self._select_work_tab("notifications-tab")

    def action_focus_model(self) -> None:
        self.query_one("#model", Input).focus()

    @on(Select.Changed, "#provider")
    def discover_selected_provider_models(self, event: Select.Changed) -> None:
        provider_name = str(event.value)
        cached = self._available_models.get(provider_name)
        if cached is not None:
            self._apply_model_suggestions(provider_name, cached)
        elif self.model_fetcher is not None and provider_name == "litellm":
            self.fetch_models(provider_name)

    @work(exclusive=True, group="model-discovery")
    async def fetch_models(self, provider_name: str, *, announce: bool = False) -> None:
        await self._discover_models(provider_name, announce=announce)

    async def _discover_models(
        self,
        provider_name: str,
        *,
        announce: bool,
    ) -> tuple[str, ...]:
        if self.model_fetcher is None:
            if announce:
                self._write_system("Model discovery is unavailable in this embedding.")
            return ()
        model_input = self.query_one("#model", Input)
        if str(self.query_one("#provider", Select).value) == provider_name:
            model_input.placeholder = "fetching available models…"
        try:
            fetched = await self.model_fetcher(provider_name)
            models = tuple(
                sorted(
                    {
                        model.strip()
                        for model in fetched
                        if isinstance(model, str) and model.strip()
                    },
                    key=str.casefold,
                )
            )
        except Exception as error:
            if str(self.query_one("#provider", Select).value) == provider_name:
                model_input.placeholder = "enter model name"
            message = f"Could not fetch {provider_name} models: {error}"
            self._record_notification(
                NotificationEvent("Model discovery failed", message, "warning")
            )
            if announce:
                self._write_system(message, error=True)
            return ()

        self._available_models[provider_name] = models
        self._apply_model_suggestions(provider_name, models)
        if announce:
            report = "\n".join(models) if models else "No models were returned."
            self._write_system(
                f"Available {provider_name} models ({len(models)}):\n{report}"
            )
        elif models:
            self.notify(f"Discovered {len(models)} {provider_name} model(s)")
        return models

    def _apply_model_suggestions(
        self,
        provider_name: str,
        models: Sequence[str],
    ) -> None:
        if str(self.query_one("#provider", Select).value) != provider_name:
            return
        model_input = self.query_one("#model", Input)
        model_input.suggester = SuggestFromList(models, case_sensitive=False)
        model_input.placeholder = (
            f"type to match {len(models)} available model(s)"
            if models
            else "enter model name"
        )

    @on(Button.Pressed, "#switch-provider")
    async def switch_provider_from_controls(self) -> None:
        provider = str(self.query_one("#provider", Select).value)
        model_text = self.query_one("#model", Input).value.strip()
        await self._switch_provider(provider, model_text or None)

    async def _switch_provider(self, provider_name: str, model: str | None) -> None:
        if self.provider_factory is None:
            self._write_system("Provider switching is unavailable in this embedding.")
            return
        try:
            provider = self.provider_factory(provider_name, model)
        except Exception as error:
            self._write_system(f"Could not switch provider: {error}", error=True)
            return
        old_provider = self.session.provider
        previous_effort = getattr(old_provider, "effort", None)
        set_effort = getattr(provider, "set_effort", None)
        if previous_effort is not None and set_effort is not None:
            await set_effort(previous_effort)
        close = getattr(old_provider, "close", None)
        if close is not None:
            await close()
        self.session.replace_provider(provider)
        self._bind_provider_handlers()
        self.query_one("#provider", Select).value = provider.name
        self.query_one("#model", Input).value = (
            "" if provider.model == "account default" else provider.model
        )
        effort = getattr(provider, "effort", None) or "default"
        self.query_one("#effort", Select).value = effort
        self.query_one("#status", Static).update(self._status_text())
        self._write_system(
            f"Switched to {provider.name} / {provider.model}; conversation preserved."
        )

    async def _handle_command(self, command_text: str) -> None:
        try:
            parts = shlex.split(command_text)
        except (OSError, RuntimeError, ValueError) as error:
            self._write_system(str(error), error=True)
            return
        command = parts[0].lower()
        args = parts[1:]
        try:
            blocked_while_busy = {
                "/clear",
                "/load",
                "/model",
                "/provider",
                "/restore",
                "/autofix",
                "/patch",
                "/review",
                "/skill",
                "/tool",
                "/undo",
            }
            if self._busy and command in blocked_while_busy:
                raise ValueError(
                    f"{command} is unavailable during an active turn; queue it or wait"
                )
            if command == "/help":
                self._write_system(
                    "/save NAME · /load NAME · /sessions · "
                    "/provider NAME [MODEL] · /model MODEL · /models [refresh] · "
                    "/tools · "
                    "/tool enable|disable NAME · /metrics · "
                    "/plan · /diff · /undo · /checkpoints · /restore NAME · "
                    "/patch preview FILE|apply|discard|status · "
                    "/autofix [--retries N] [--timeout 5m] [-- COMMAND] · "
                    "/review [SCOPE] · "
                    "/skills [reload] · /skill show|run|enable|disable NAME · "
                    "/permissions · /permission ACTION COMMAND · /limits · "
                    "/limit set NAME VALUE · /themes · /theme NAME · "
                    "/accessibility [contrast|motion|density VALUE] · "
                    "/bg COMMAND · /bg-agent PROMPT · /jobs · /cancel ID · "
                    "/agent ROLE PROMPT · /agents · /enqueue PROMPT · "
                    "/queue [remove ID|clear] · /notifications · /clear · /help"
                    " · /goal start [--timeout 30m] OBJECTIVE · /goal status|stop"
                )
            elif command == "/save":
                if len(args) != 1:
                    raise ValueError("Usage: /save NAME")
                provider = self.session.provider
                path = self.conversation_store.save(
                    args[0],
                    self.session.messages,
                    provider=provider.name,
                    model=provider.model,
                )
                self._write_system(f"Saved conversation to {path}")
            elif command == "/load":
                if len(args) != 1:
                    raise ValueError("Usage: /load NAME")
                messages, metadata = self.conversation_store.load(args[0])
                saved_provider = str(metadata.get("provider") or "")
                saved_model = str(metadata.get("model") or "") or None
                if saved_model == "account default":
                    saved_model = None
                if saved_provider and self.provider_factory is not None:
                    await self._switch_provider(saved_provider, saved_model)
                self.session.restore(messages)
                self._render_conversation(f"Loaded {args[0]!r}")
            elif command == "/sessions":
                names = self.conversation_store.list()
                self._write_system(
                    "Saved conversations: " + (", ".join(names) if names else "none")
                )
            elif command in {"/provider", "/model"}:
                if command == "/provider":
                    if not args:
                        self.action_focus_model()
                        self._write_system(
                            "Choose a provider and model above, then select Switch."
                        )
                    else:
                        if len(args) > 2:
                            raise ValueError("Usage: /provider NAME [MODEL]")
                        model = args[1] if len(args) > 1 else None
                        await self._switch_provider(args[0].lower(), model)
                else:
                    if len(args) != 1:
                        raise ValueError("Usage: /model MODEL")
                    await self._switch_provider(self.session.provider.name, args[0])
            elif command == "/models":
                if args not in ([], ["refresh"]):
                    raise ValueError("Usage: /models [refresh]")
                provider_name = str(self.query_one("#provider", Select).value)
                cached = self._available_models.get(provider_name)
                if cached is not None and not args:
                    report = "\n".join(cached) if cached else "No models were returned."
                    self._write_system(
                        f"Available {provider_name} models ({len(cached)}):\n{report}"
                    )
                else:
                    await self._discover_models(provider_name, announce=True)
            elif command == "/tools":
                self._show_tool_registry()
            elif command == "/permissions":
                if args:
                    raise ValueError("Usage: /permissions")
                self._write_system(self._permission_report())
            elif command == "/permission":
                self._handle_permission_command(args)
            elif command == "/limits":
                if args:
                    raise ValueError("Usage: /limits")
                self._write_system(self.limit_store.report())
            elif command == "/limit":
                self._handle_limit_command(args)
            elif command == "/themes":
                if args:
                    raise ValueError("Usage: /themes")
                self._write_system(
                    "Available themes:\n" + "\n".join(sorted(self.available_themes))
                )
            elif command == "/theme":
                if len(args) != 1:
                    raise ValueError("Usage: /theme NAME")
                if args[0] not in self.available_themes:
                    raise ValueError(f"Unknown theme {args[0]!r}; use /themes")
                self.appearance_store.update("theme", args[0])
                self._apply_appearance()
                self._write_system(f"Theme changed to {args[0]}.")
            elif command == "/accessibility":
                self._handle_accessibility_command(args)
            elif command == "/skills":
                if not args:
                    self._write_system(self._skills_report())
                elif args == ["reload"]:
                    count = self.skill_registry.reload()
                    self._write_system(
                        f"Reloaded {count} skill(s).\n{self._skills_report()}"
                    )
                    self.query_one("#status", Static).update(self._status_text())
                else:
                    raise ValueError("Usage: /skills [reload]")
            elif command == "/skill":
                self._handle_skill_command(args)
            elif command == "/tool":
                if len(args) != 2 or args[0] not in {"enable", "disable"}:
                    raise ValueError("Usage: /tool enable|disable NAME")
                enabled = args[0] == "enable"
                changed = self.tool_registry.set_enabled(args[1], enabled)
                if self.session.provider.name == "codex":
                    await self._reset_provider_context()
                label = ", ".join(changed)
                suffix = (
                    " (shared Codex workspace runtime)" if len(changed) > 1 else ""
                )
                self._write_system(
                    f"Tools {label} {'enabled' if enabled else 'disabled'}{suffix}."
                )
            elif command == "/metrics":
                self._write_system(self._metrics_report())
            elif command == "/plan":
                self.action_show_plan()
            elif command == "/diff":
                self.action_show_diff()
            elif command == "/notifications":
                self.action_show_notifications()
            elif command == "/checkpoints":
                checkpoints = self.checkpoint_store.list()
                report = "\n".join(
                    f"{checkpoint.name} — "
                    f"{'undone' if checkpoint.undone else 'active'}"
                    for checkpoint in checkpoints
                )
                self._write_system(report or "No checkpoints saved yet.")
            elif command == "/undo":
                if args:
                    raise ValueError("Usage: /undo")
                checkpoint = await self.checkpoint_store.undo_latest()
                message = f"Undid checkpoint {checkpoint.name}."
                self._write_system(message)
                self._record_notification(
                    NotificationEvent("Checkpoint undone", message)
                )
            elif command == "/restore":
                if len(args) != 1:
                    raise ValueError("Usage: /restore NAME")
                checkpoint = await self.checkpoint_store.restore(args[0])
                message = f"Restored checkpoint {checkpoint.name}."
                self._write_system(message)
                self._record_notification(
                    NotificationEvent("Checkpoint restored", message)
                )
            elif command == "/patch":
                await self._handle_patch_command(args)
            elif command == "/autofix":
                if self.provider_factory is None:
                    raise ValueError("Autofix requires a provider factory")
                retries, timeout_seconds, test_command = parse_autofix_args(args)
                await self._authorize_direct_command(
                    list(test_command),
                    "autofix test command",
                )
                self._busy = True
                self._active_diff = ""
                self.query_one("#switch-provider", Button).disabled = True
                job = self.background.start(
                    f"Autofix: {shlex.join(test_command)[:100]}",
                    lambda: self._run_autofix(
                        test_command,
                        retries=retries,
                        timeout_seconds=timeout_seconds,
                    ),
                    kind="autofix",
                )
                self._write_system(
                    f"Started autofix job {job.id} with at most {retries} fix "
                    f"attempt(s)."
                )
                await self._upsert_command_output(
                    ToolActivity(
                        "test",
                        "running",
                        f"Autofix job {job.id}",
                        shlex.join(test_command),
                    ),
                    key=f"background:{job.id}",
                )
                self.query_one("#status", Static).update(self._status_text())
            elif command == "/review":
                if self.provider_factory is None:
                    raise ValueError("Independent review requires a provider factory")
                scope = " ".join(args).strip() or "all changes"
                job = self.background.start(
                    f"Independent review: {scope[:80]}",
                    lambda: self._run_review(scope),
                    kind="review",
                )
                self._write_system(f"Started independent review job {job.id}.")
            elif command == "/bg":
                if not args:
                    raise ValueError("Usage: /bg COMMAND [ARGS...]")
                await self._authorize_direct_command(args, "background command")
                job = self.background.start(
                    shlex.join(args),
                    lambda: self._run_background_command(args),
                    kind="command",
                )
                await self._upsert_command_output(
                    ToolActivity(
                        "command",
                        "running",
                        f"Background command {job.id}",
                        job.title,
                    ),
                    key=f"background:{job.id}",
                )
                self._write_system(f"Started background job {job.id}: {job.title}")
            elif command == "/bg-agent":
                if not args:
                    raise ValueError("Usage: /bg-agent PROMPT")
                if self.provider_factory is None:
                    raise ValueError("Background agent jobs require a provider factory")
                prompt = " ".join(args)
                job = self.background.start(
                    f"Agent: {prompt[:80]}",
                    lambda: self._run_background_agent(prompt),
                )
                self._write_system(f"Started background job {job.id}: {job.title}")
            elif command == "/agent":
                if len(args) < 2:
                    raise ValueError("Usage: /agent ROLE PROMPT")
                if self.provider_factory is None:
                    raise ValueError("Subagents require a provider factory")
                task = " ".join(args[1:])
                role, prompt = build_subagent_prompt(args[0], task)
                job = self.background.start(
                    f"{role.name.title()} subagent: {task[:70]}",
                    lambda: self._run_background_agent(prompt),
                    kind="subagent",
                )
                self._write_system(f"Started subagent job {job.id}: {job.title}")
            elif command == "/agents":
                roles = "\n".join(
                    f"{role.name} — {role.description}"
                    for role in SUBAGENT_ROLES.values()
                )
                self._write_system(f"Available subagents:\n{roles}")
            elif command == "/jobs":
                self._write_system(self._jobs_report())
            elif command == "/cancel":
                if len(args) != 1 or not args[0].isdigit():
                    raise ValueError("Usage: /cancel ID")
                job = self.background.cancel(int(args[0]))
                self._write_system(f"Cancelling background job {job.id}.")
            elif command == "/enqueue":
                if not args:
                    raise ValueError("Usage: /enqueue PROMPT")
                item = self.prompt_queue.enqueue(" ".join(args))
                self._write_system(f"Queued prompt {item.id}: {item.text}")
                if not self._busy:
                    self._start_next_queued_prompt()
                else:
                    self.query_one("#status", Static).update(self._status_text())
            elif command == "/queue":
                if not args:
                    self._write_system(self._queue_report())
                elif args == ["clear"]:
                    count = self.prompt_queue.clear()
                    self._write_system(f"Removed {count} queued prompt(s).")
                    self.query_one("#status", Static).update(self._status_text())
                elif len(args) == 2 and args[0] == "remove" and args[1].isdigit():
                    item = self.prompt_queue.remove(int(args[1]))
                    self._write_system(f"Removed queued prompt {item.id}.")
                    self.query_one("#status", Static).update(self._status_text())
                else:
                    raise ValueError("Usage: /queue [remove ID|clear]")
            elif command == "/goal":
                await self._handle_goal_command(args)
            elif command == "/goals":
                self._write_system(self._goals_report())
            elif command == "/clear":
                await self.action_clear_chat()
            else:
                raise ValueError(f"Unknown command {command!r}; use /help")
        except (OSError, ValueError) as error:
            self._write_system(str(error), error=True)

    def _show_tool_registry(self) -> None:
        provider = self.session.provider.name
        lines = ["[bold]Tool registry[/bold]"]
        for plugin, enabled, supported in self.tool_registry.entries(provider):
            state = "enabled" if enabled else "disabled"
            availability = "" if supported else f"; unavailable for {provider}"
            lines.append(
                f"{'✓' if enabled and supported else '○'} "
                f"[bold]{plugin.name}[/bold] "
                f"[dim]({state}{availability}) — {escape(plugin.description)}[/dim]"
            )
        tools = self.query_one("#tools", RichLog)
        tools.clear()
        for line in lines:
            tools.write(line)
        self.action_show_tools()

    def _permission_report(self) -> str:
        rules = self.command_policy.list()
        lines = [f"Default agent-command policy: {self.command_policy.default_action}"]
        lines.extend(
            f"{rule.id}: {rule.action} — {shlex.join(rule.command)}"
            for rule in rules
        )
        if not rules:
            lines.append("No command-specific rules.")
        return "\n".join(lines)

    def _handle_permission_command(self, args: list[str]) -> None:
        if len(args) == 2 and args[0] == "default":
            self.command_policy.set_default(args[1])
            self._write_system(f"Default command policy set to {args[1]}.")
            return
        if len(args) >= 2 and args[0] in {"allow", "ask", "deny"}:
            rule = self.command_policy.add(args[0], args[1:])
            self._write_system(
                f"Added permission rule {rule.id}: {rule.action} — "
                f"{shlex.join(rule.command)}"
            )
            return
        if len(args) == 2 and args[0] == "remove" and args[1].isdigit():
            rule = self.command_policy.remove(int(args[1]))
            self._write_system(f"Removed permission rule {rule.id}.")
            return
        if args == ["reset"]:
            self.command_policy.reset()
            self._session_command_allows.clear()
            self._write_system("Command permission rules reset.")
            return
        raise ValueError(
            "Usage: /permission default allow|ask|deny | "
            "/permission allow|ask|deny COMMAND [ARGS...] | "
            "/permission remove ID | /permission reset"
        )

    def _handle_limit_command(self, args: list[str]) -> None:
        if len(args) == 3 and args[0] == "set":
            self.limit_store.set(args[1], args[2])
            self._write_system(self.limit_store.report())
            return
        if args == ["reset"]:
            self.limit_store.reset()
            self._write_system("Execution limits reset to defaults.")
            return
        raise ValueError("Usage: /limit set NAME VALUE | /limit reset")

    def _handle_accessibility_command(self, args: list[str]) -> None:
        if not args:
            self._write_system(self.appearance_store.report())
            return
        if len(args) != 2 or args[0] not in {"contrast", "motion", "density"}:
            raise ValueError(
                "Usage: /accessibility contrast on|off | motion full|reduced | "
                "density compact|comfortable"
            )
        self.appearance_store.update(args[0], args[1])
        self._apply_appearance()
        self._write_system(self.appearance_store.report())

    def _handle_skill_command(self, args: list[str]) -> None:
        if len(args) == 2 and args[0] == "show":
            skill = self.skill_registry.get(args[1])
            triggers = ", ".join(skill.triggers) or "none"
            self._write_system(
                f"Skill: {skill.name}\nDescription: {skill.description}\n"
                f"Enabled: {self.skill_registry.is_enabled(skill.name)}\n"
                f"Auto-activate: {skill.auto_activate}\nTriggers: {triggers}\n"
                f"Path: {skill.path}\n\n{skill.instructions}"
            )
            return
        if len(args) == 2 and args[0] in {"enable", "disable"}:
            enabled = args[0] == "enable"
            skill = self.skill_registry.set_enabled(args[1], enabled)
            self._write_system(
                f"Skill {skill.name!r} {'enabled' if enabled else 'disabled'}."
            )
            self.query_one("#status", Static).update(self._status_text())
            return
        if len(args) >= 3 and args[0] == "run":
            skill = self.skill_registry.get(args[1])
            if not self.skill_registry.is_enabled(skill.name):
                raise ValueError(f"Skill {skill.name!r} is disabled")
            task = " ".join(args[2:]).strip()
            provider_prompt = build_skill_prompt(skill, task)
            self._write_system(f"Running skill {skill.name!r}.")
            self._start_prompt(task, provider_prompt=provider_prompt)
            return
        raise ValueError(
            "Usage: /skill show NAME | /skill run NAME TASK | "
            "/skill enable|disable NAME"
        )

    def _skills_report(self) -> str:
        entries = self.skill_registry.entries()
        lines = [
            f"{'✓' if enabled else '○'} {skill.name} — {skill.description}"
            + (" [auto]" if skill.auto_activate else "")
            for skill, enabled in entries
        ]
        if self.skill_registry.errors:
            lines.append("Load errors:")
            lines.extend(
                f"- {error.path}: {error.message}"
                for error in self.skill_registry.errors
            )
        return "Available skills:\n" + ("\n".join(lines) if lines else "none")

    def _metrics_report(self) -> str:
        metrics = getattr(self.session.provider, "session_metrics", None)
        if metrics is None or metrics.requests == 0:
            return "No model usage recorded in this session yet."
        cost = (
            f"${metrics.cost_usd:.6f}"
            if metrics.cost_usd is not None
            else "unavailable"
        )
        return (
            f"Requests: {metrics.requests} | Input: {metrics.input_tokens:,} | "
            f"Output: {metrics.output_tokens:,} | Total: {metrics.total_tokens:,} | "
            f"Average latency: {metrics.average_latency_ms}ms | Cost: {cost}"
        )

    async def _handle_patch_command(self, args: list[str]) -> None:
        if len(args) == 2 and args[0] == "preview":
            preview = await self.patch_manager.preview_file(Path(args[1]))
            self._render_diff(preview.diff)
            self.action_show_diff()
            paths = ", ".join(preview.paths)
            self._write_system(
                f"Patch preview ready ({preview.summary}): {paths}. "
                "Use /patch apply to apply it or /patch discard to cancel."
            )
            return
        if args == ["apply"]:
            preview = self.patch_manager.pending
            if preview is None:
                raise ValueError("There is no previewed patch to apply")
            decision = await self._request_approval(
                ApprovalRequest(
                    kind="patch",
                    title="Apply previewed patch",
                    detail=f"{preview.summary}: {', '.join(preview.paths)}",
                    reason="The patch will be syntax-checked and can be undone.",
                )
            )
            if decision not in {"accept", "acceptForSession"}:
                self._write_system("Patch application cancelled; preview preserved.")
                return
            applied = await self.patch_manager.apply_pending()
            checkpoint = self.checkpoint_store.save(applied.diff)
            self._render_diff(applied.diff)
            self.action_show_diff()
            checkpoint_text = (
                f" Checkpoint {checkpoint.name} was saved for /undo."
                if checkpoint is not None
                else ""
            )
            message = f"Applied patch ({applied.summary}).{checkpoint_text}"
            self._write_system(message)
            self._record_notification(NotificationEvent("Patch applied", message))
            return
        if args == ["discard"]:
            discarded = self.patch_manager.discard()
            self._write_system(f"Discarded patch preview from {discarded.source}.")
            return
        if args == ["status"]:
            pending = self.patch_manager.pending
            message = (
                f"Pending patch from {pending.source} ({pending.summary}): "
                f"{', '.join(pending.paths)}"
                if pending is not None
                else "No patch is currently previewed."
            )
            self._write_system(message)
            return
        raise ValueError("Usage: /patch preview FILE | /patch apply|discard|status")

    def _save_active_checkpoint(self) -> None:
        diff, self._active_diff = self._active_diff, ""
        if not diff.strip():
            return
        try:
            checkpoint = self.checkpoint_store.save(diff)
        except OSError as error:
            self._record_notification(
                NotificationEvent("Checkpoint failed", str(error), "error")
            )
            return
        if checkpoint is not None:
            self._record_notification(
                NotificationEvent(
                    "Checkpoint saved",
                    f"{checkpoint.name} can be reverted with /undo.",
                )
            )

    async def _run_background_command(self, args: list[str]) -> str:
        limits = self.limit_store.limits
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=Path.cwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **limits.subprocess_options(),
        )
        output = bytearray()

        async def collect_output() -> None:
            if process.stdout is None:
                raise RuntimeError("Background command output is unavailable")
            while chunk := await process.stdout.read(8192):
                output.extend(chunk)
                overflow = len(output) - limits.output_bytes
                if overflow > 0:
                    del output[:overflow]
            await process.wait()

        try:
            await asyncio.wait_for(
                collect_output(),
                timeout=limits.wall_time_seconds,
            )
        except asyncio.TimeoutError as error:
            await self._terminate_background_process(process)
            raise RuntimeError(
                f"Command timed out after {limits.wall_time_seconds:g}s"
            ) from error
        except asyncio.CancelledError:
            await self._terminate_background_process(process)
            raise
        decoded = output.decode(errors="replace")
        if process.returncode != 0:
            detail = decoded.strip() or f"exit status {process.returncode}"
            raise RuntimeError(detail)
        return decoded.strip() or "Command completed with no output."

    @staticmethod
    async def _terminate_background_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        if os.name == "posix":
            # The process-group ID remains the parent's PID even if the parent
            # exits before a child that inherited its output pipe.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        else:
            if process.returncode is not None:
                return
            with suppress(ProcessLookupError):
                process.terminate()
        if process.returncode is not None:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
            return
        except asyncio.TimeoutError:
            pass
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()

    async def _run_background_agent(self, prompt: str) -> str:
        if self.provider_factory is None:
            raise RuntimeError("Background agent jobs require a provider factory")
        current = self.session.provider
        model = None if current.model == "account default" else current.model
        provider = self.provider_factory(current.name, model)
        try:
            return await provider.complete([ChatMessage(role="user", content=prompt)])
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    async def _run_autofix(
        self,
        command: tuple[str, ...],
        *,
        retries: int,
        timeout_seconds: float,
    ) -> str:
        provider = self._fresh_provider()
        self._bind_handlers(provider)

        async def fix(prompt: str) -> str:
            return await provider.complete([ChatMessage(role="user", content=prompt)])

        try:
            report = await self.quality_workflow.run(
                command,
                fix,
                retries=retries,
                timeout_seconds=timeout_seconds,
                progress=self._show_quality_progress,
            )
            rendered = report.render()
            if not report.passed:
                raise RuntimeError(rendered)
            return rendered
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    async def _run_review(self, scope: str) -> str:
        diff = await collect_workspace_diff()
        prompt = build_review_prompt(diff, scope)
        provider = self._fresh_provider()
        try:
            return await provider.complete([ChatMessage(role="user", content=prompt)])
        finally:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    def _fresh_provider(self) -> ChatProvider:
        if self.provider_factory is None:
            raise RuntimeError("This workflow requires a provider factory")
        current = self.session.provider
        model = None if current.model == "account default" else current.model
        return self.provider_factory(current.name, model)

    async def _show_quality_progress(self, message: str) -> None:
        await self._show_activity(
            ToolActivity("test", "running", "Autofix", message)
        )

    async def _background_finished(self, job: BackgroundJob) -> None:
        if self._shutting_down:
            return
        duration = f"{(job.duration_ms or 0) / 1000:.2f}s"
        detail = job.output if job.status == "completed" else job.error
        if job.status == "cancelled":
            detail = "Cancelled by user."
        self.query_one("#tools", RichLog).write(
            f"\n[bold]Background job {job.id}: {escape(job.title)}[/bold]\n"
            f"Status: {job.status} ({duration})\n{escape(detail[-4000:])}"
        )
        if job.kind in {"autofix", "command"}:
            await self._upsert_command_output(
                ToolActivity(
                    kind="test" if job.kind == "autofix" else "command",
                    status=job.status,
                    title=(
                        f"Autofix job {job.id}"
                        if job.kind == "autofix"
                        else f"Background command {job.id}"
                    ),
                    detail=job.title,
                    output=detail,
                    duration_ms=job.duration_ms,
                ),
                key=f"background:{job.id}",
            )
        if job.kind in {"autofix", "review", "subagent"}:
            heading = "Independent review" if job.kind == "review" else job.title
            content = job.output if job.status == "completed" else detail
            self.query_one("#chat", RichLog).write(
                f"\n[bold magenta]{escape(heading)}[/bold magenta]\n"
                f"{escape(content)}"
            )
        severity = "error" if job.status == "failed" else "information"
        self._record_notification(
            NotificationEvent(
                f"Background job {job.id} {job.status}",
                job.title,
                severity,
            )
        )
        if job.kind == "autofix":
            self._save_active_checkpoint()
            self._busy = False
            if self.prompt_queue.list():
                self._start_next_queued_prompt()
            else:
                self.query_one("#switch-provider", Button).disabled = (
                    self.provider_factory is None
                )
                self.query_one("#status", Static).update(self._status_text())

    def _jobs_report(self) -> str:
        jobs = self.background.list()
        if not jobs:
            return "No background jobs."
        return "\n".join(
            f"{job.id}: {job.status} [{job.kind}] — {job.title}"
            + (
                f" ({(job.duration_ms or 0) / 1000:.2f}s)"
                if job.duration_ms is not None
                else ""
            )
            for job in jobs
        )

    def _queue_report(self) -> str:
        items = self.prompt_queue.list()
        if not items:
            return "The prompt queue is empty."
        return "\n".join(f"{item.id}: {item.text}" for item in items)

    async def _handle_goal_command(self, args: list[str]) -> None:
        action = args[0].lower() if args else "status"
        if (action == "status" and len(args) == 1) or not args:
            goal = self.goal_store.active() or self._active_goal
            self._write_system(
                self._goal_report(goal) if goal is not None else "No active goal."
            )
            return
        if action == "start":
            if self._busy:
                raise ValueError(
                    "Wait for the active turn to finish before starting a goal"
                )
            if self.goal_store.active() is not None:
                raise ValueError(
                    "A goal is already active; use /goal status or /goal stop"
                )
            objective_parts = list(args[1:])
            timeout_seconds = parse_timeout("30m")
            if "--timeout" in objective_parts:
                index = objective_parts.index("--timeout")
                if index + 1 >= len(objective_parts):
                    raise ValueError("--timeout requires a duration such as 30m")
                timeout_seconds = parse_timeout(objective_parts[index + 1])
                del objective_parts[index : index + 2]
            objective = " ".join(objective_parts)
            goal = self.goal_store.create(objective, timeout_seconds)
            self._start_goal(goal)
            return
        if action == "stop" and len(args) == 1:
            goal = self.goal_store.active()
            if goal is None:
                raise ValueError("No active goal to stop")
            goal.status = "stopped"
            self.goal_store.save(goal)
            if self._goal_worker is not None:
                self._goal_worker.cancel()
            self._record_notification(
                NotificationEvent("Goal stopped", goal.objective, "warning")
            )
            self._write_system(f"Stopped goal {goal.id}.")
            return
        raise ValueError(
            "Usage: /goal start [--timeout 30m] OBJECTIVE | /goal status | /goal stop"
        )

    def _start_goal(self, goal: Goal, *, resumed: bool = False) -> None:
        self._active_goal = goal
        self._busy = True
        self.query_one("#switch-provider", Button).disabled = True
        verb = "Resuming" if resumed else "Started"
        self._write_system(
            f"{verb} goal {goal.id} with {self._remaining_goal_time(goal)} remaining:\n"
            f"{goal.objective}"
        )
        self._render_goal_plan(goal)
        self.action_show_plan()
        self.query_one("#status", Static).update(self._status_text())
        self._goal_worker = self.run_goal(goal)

    @work
    async def run_goal(self, goal: Goal) -> None:
        try:
            while goal.status == "active":
                remaining = goal.remaining_seconds
                if remaining <= 0:
                    self._finish_goal(
                        goal,
                        "timed_out",
                        "Goal deadline reached",
                        severity="warning",
                    )
                    break
                self._streamed_text = ""
                self._active_diff = ""
                streaming = self.query_one("#streaming", Static)
                streaming.update(
                    f"[bold cyan]Goal attempt {goal.attempts + 1}[/bold cyan]\n"
                    "[dim]Working…[/dim]"
                )
                streaming.styles.display = "block"
                try:
                    reply = await asyncio.wait_for(
                        self.session.complete_transient(build_goal_prompt(goal)),
                        timeout=remaining,
                    )
                except TimeoutError:
                    self._save_active_checkpoint()
                    self._finish_goal(
                        goal,
                        "timed_out",
                        "Goal deadline reached",
                        severity="warning",
                    )
                    break
                except Exception as error:
                    goal.attempts += 1
                    goal.last_result = f"Attempt failed: {error}"
                    self.goal_store.save(goal)
                    self._render_goal_plan(goal)
                    self._record_notification(
                        NotificationEvent(
                            "Goal attempt failed; retrying",
                            str(error),
                            "warning",
                        )
                    )
                    backoff = min(30, 2 ** min(goal.attempts, 5))
                    await asyncio.sleep(min(backoff, max(0, goal.remaining_seconds)))
                    continue
                self._save_active_checkpoint()
                goal.attempts += 1
                goal.last_result = clean_goal_reply(reply)
                self.goal_store.save(goal)
                self.query_one("#chat", RichLog).write(
                    f"\n[bold magenta]Goal attempt {goal.attempts}[/bold magenta]\n"
                    f"{escape(goal.last_result)}"
                )
                self._render_goal_plan(goal)
                if goal_reply_complete(reply):
                    self._finish_goal(goal, "completed", "Goal completed and verified")
                    break
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._save_active_checkpoint()
            self._finish_goal(goal, "failed", str(error), severity="error")
        finally:
            if not self._shutting_down:
                await self._reset_provider_context()
                self.query_one("#streaming", Static).styles.display = "none"
                self._busy = False
                self._active_goal = None
                self._goal_worker = None
                if self.prompt_queue.list():
                    self._start_next_queued_prompt()
                else:
                    self.query_one("#switch-provider", Button).disabled = (
                        self.provider_factory is None
                    )
                    self.query_one("#status", Static).update(self._status_text())
                self.query_one("#prompt", Input).focus()

    def _finish_goal(
        self,
        goal: Goal,
        status: str,
        message: str,
        *,
        severity: str = "information",
    ) -> None:
        goal.status = status
        self.goal_store.save(goal)
        self._render_goal_plan(goal)
        title = status.replace("_", " ").title()
        self._write_system(f"{title}: {message}.")
        self._record_notification(NotificationEvent(title, goal.objective, severity))

    def _render_goal_plan(self, goal: Goal) -> None:
        plan = self.query_one("#plan", RichLog)
        plan.clear()
        plan.write(f"[bold]Goal[/bold] {escape(goal.objective)}")
        plan.write(
            f"Status: [bold]{escape(goal.status)}[/bold] | "
            f"Attempts: {goal.attempts} | Remaining: {self._remaining_goal_time(goal)}"
        )
        if goal.last_result:
            latest = escape(goal.last_result[-2000:])
            plan.write(f"\n[dim]Latest result[/dim]\n{latest}")

    def _goal_report(self, goal: Goal) -> str:
        return (
            f"Goal {goal.id} | Status: {goal.status} | Attempts: {goal.attempts} | "
            f"Remaining: {self._remaining_goal_time(goal)}\n{goal.objective}"
        )

    def _goals_report(self) -> str:
        goals = self.goal_store.list()
        if not goals:
            return "No saved goals."
        return "\n".join(
            f"{goal.id}: {goal.status} ({goal.attempts} attempts) — {goal.objective}"
            for goal in goals
        )

    @staticmethod
    def _remaining_goal_time(goal: Goal) -> str:
        seconds = max(0, round(goal.remaining_seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def _record_notification(self, event: NotificationEvent) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.query_one("#notifications", RichLog).write(
            f"[dim]{timestamp}[/dim] [bold]{escape(event.title)}[/bold]\n"
            f"  {escape(event.message)}"
        )
        self.notify(event.message, title=event.title, severity=event.severity)

    def _render_conversation(self, title: str) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        chat.write(f"[bold cyan]{escape(title)}[/bold cyan]")
        for message in self.session.messages:
            label = "You" if message.role == "user" else "Assistant"
            color = "green" if message.role == "user" else "cyan"
            chat.write(
                f"\n[bold {color}]{label}[/bold {color}]\n{escape(message.content)}"
            )

    def _write_system(self, message: str, *, error: bool = False) -> None:
        color = "red" if error else "yellow"
        self.query_one("#chat", RichLog).write(
            f"\n[bold {color}]System[/bold {color}]\n{escape(message)}"
        )

    async def _reset_provider_context(self) -> None:
        reset = getattr(self.session.provider, "reset", None)
        if reset is None:
            return
        try:
            await reset()
        except Exception as error:
            self._record_notification(
                NotificationEvent(
                    "Provider context reset failed",
                    str(error),
                    "error",
                )
            )

    async def action_clear_chat(self) -> None:
        if self._busy:
            self._write_system(
                "Cannot clear the conversation during an active turn; wait for it "
                "to finish."
            )
            return
        self.session.clear()
        await self._reset_provider_context()
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        chat.write("[bold cyan]Conversation cleared.[/bold cyan]")

    async def on_unmount(self) -> None:
        self._shutting_down = True
        if self._goal_worker is not None:
            self._goal_worker.cancel()
        await self.background.shutdown()
        close = getattr(self.session.provider, "close", None)
        if close is not None:
            await close()
