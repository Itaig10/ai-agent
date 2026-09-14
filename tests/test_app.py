import asyncio
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from textual.containers import VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    RichLog,
    Select,
    Static,
    TabbedContent,
)

from ai_agent.app import AgentApp, ApprovalScreen
from ai_agent.appearance import AppearanceStore
from ai_agent.checkpoints import CheckpointStore
from ai_agent.conversations import ConversationStore
from ai_agent.events import (
    ApprovalRequest,
    DiffUpdate,
    PlanStep,
    PlanUpdate,
    ToolActivity,
)
from ai_agent.execution import CommandPolicyStore, ExecutionLimitStore
from ai_agent.goals import GoalStore
from ai_agent.messages import ChatMessage
from ai_agent.patches import PatchManager
from ai_agent.quality import AutoFixReport, TestRun
from ai_agent.session import ChatSession
from ai_agent.skills import SkillRegistry
from ai_agent.tools import ToolRegistry


class FakeInteractiveProvider:
    name = "fake"
    model = "fake-model"
    effort = "medium"
    context_percent = 25.0
    auto_compact_threshold = 0.8
    compaction_count = 0

    async def complete(self, messages: list[ChatMessage]) -> str:
        return f"reply to {messages[-1].content}"

    def set_event_handlers(
        self,
        *,
        activity: object,
        approval: object,
        plan: object,
        diff: object,
        text_delta: object,
    ) -> None:
        self.activity_handler = activity
        self.approval_handler = approval
        self.plan_handler = plan
        self.diff_handler = diff
        self.text_delta_handler = text_delta


class QueuedProvider(FakeInteractiveProvider):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.second_finished = asyncio.Event()
        self.prompts: list[str] = []

    async def complete(self, messages: list[ChatMessage]) -> str:
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            self.started.set()
            await self.release.wait()
        else:
            self.second_finished.set()
        return f"reply to {prompt}"


class GoalProvider(FakeInteractiveProvider):
    def __init__(self) -> None:
        self.attempts = 0
        self.completed = asyncio.Event()
        self.reset_count = 0

    async def complete(self, messages: list[ChatMessage]) -> str:
        self.attempts += 1
        if self.attempts == 1:
            return "Implemented the change.\n<goal-status>continue</goal-status>"
        self.completed.set()
        return "Tests pass.\n<goal-status>complete</goal-status>"

    async def reset(self) -> None:
        self.reset_count += 1


class ResettableProvider(FakeInteractiveProvider):
    def __init__(self) -> None:
        self.reset_count = 0

    async def reset(self) -> None:
        self.reset_count += 1


class ClosableProvider(FakeInteractiveProvider):
    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


class HangingGoalProvider(FakeInteractiveProvider):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, messages: list[ChatMessage]) -> str:
        self.started.set()
        await asyncio.Event().wait()
        return "unreachable"


class SkillProvider(FakeInteractiveProvider):
    def __init__(self) -> None:
        self.received: list[ChatMessage] = []
        self.completed = asyncio.Event()

    async def complete(self, messages: list[ChatMessage]) -> str:
        self.received = list(messages)
        self.completed.set()
        return "skill completed"


class AgentAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_panel_and_controls_render(self) -> None:
        provider = FakeInteractiveProvider()
        provider.verify_tls = False
        app = AgentApp(ChatSession(provider))

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            self.assertIsNotNone(app.query_one("#tools", RichLog))
            self.assertIsNotNone(
                app.query_one("#command-outputs", VerticalScroll)
            )
            self.assertIsNotNone(app.query_one("#plan", RichLog))
            self.assertIsNotNone(app.query_one("#diff", RichLog))
            self.assertIsNotNone(app.query_one("#notifications", RichLog))
            self.assertEqual(app.query_one("#effort", Select).value, "medium")
            status = app.query_one("#status", Static).render()
            self.assertIn("Context: 25.0%", str(status))
            self.assertIn("TLS: verification disabled", str(status))
            self.assertTrue(callable(provider.activity_handler))
            self.assertTrue(callable(provider.approval_handler))
            self.assertTrue(callable(provider.plan_handler))
            self.assertTrue(callable(provider.diff_handler))
            self.assertTrue(callable(provider.text_delta_handler))

            await provider.plan_handler(
                PlanUpdate(
                    steps=(PlanStep("Run tests", "inProgress"),),
                    explanation="Checking the change",
                )
            )
            await provider.diff_handler(
                DiffUpdate("--- a/example.py\n+++ b/example.py\n+added\n-removed")
            )
            await pilot.pause()

            app.action_show_plan()
            self.assertEqual(
                app.query_one("#work-panel", TabbedContent).active,
                "plan-tab",
            )
            app.action_show_diff()
            self.assertEqual(
                app.query_one("#work-panel", TabbedContent).active,
                "diff-tab",
            )
            app.action_show_notifications()
            self.assertEqual(
                app.query_one("#work-panel", TabbedContent).active,
                "notifications-tab",
            )

            app.push_screen(
                ApprovalScreen(
                    ApprovalRequest(
                        kind="command",
                        title="Approve command",
                        detail="python -m unittest",
                    )
                )
            )
            await pilot.pause()
            self.assertEqual(
                app.screen.query_one("#accept", Button).label.plain,
                "Allow once",
            )
            await pilot.press("escape")

    async def test_command_output_is_collapsible_and_updates_in_place(self) -> None:
        provider = FakeInteractiveProvider()
        app = AgentApp(ChatSession(provider))

        async with app.run_test(size=(120, 40)) as pilot:
            await provider.activity_handler(
                ToolActivity(
                    "command",
                    "running",
                    "Command",
                    "python -m unittest",
                )
            )
            await pilot.pause()
            card = app.query_one("#command-outputs").query_one(Collapsible)
            self.assertFalse(card.collapsed)

            await provider.activity_handler(
                ToolActivity(
                    "command",
                    "completed",
                    "Command",
                    "python -m unittest  [exit 0]",
                    "Ran 153 tests\nOK",
                    1250,
                )
            )
            await pilot.pause()

            cards = list(app.query_one("#command-outputs").query(Collapsible))
            self.assertEqual(len(cards), 1)
            self.assertTrue(cards[0].collapsed)
            self.assertIn("completed", str(cards[0].title))
            self.assertIn("1.25s", str(cards[0].title))
            body = cards[0].query_one(".command-output-body", Static)
            self.assertIn("Ran 153 tests", str(body.render()))
            title = cards[0].query_one("CollapsibleTitle")
            title.focus()
            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(cards[0].collapsed)
            await pilot.press("enter")
            await pilot.pause()
            self.assertTrue(cards[0].collapsed)

    async def test_failed_command_output_stays_expanded_and_is_plain_text(self) -> None:
        provider = FakeInteractiveProvider()
        app = AgentApp(ChatSession(provider))

        async with app.run_test(size=(120, 40)) as pilot:
            await provider.activity_handler(
                ToolActivity(
                    "command",
                    "failed",
                    "Command",
                    "unsafe-looking output",
                    "[bold red]literal failure[/bold red]",
                    20,
                )
            )
            await pilot.pause()

            card = app.query_one("#command-outputs").query_one(Collapsible)
            body = card.query_one(".command-output-body", Static)
            self.assertFalse(card.collapsed)
            self.assertFalse(body._render_markup)
            self.assertIn("[bold red]", body.render().plain)

    async def test_approval_content_is_rendered_without_markup(self) -> None:
        provider = FakeInteractiveProvider()
        app = AgentApp(ChatSession(provider))
        request = ApprovalRequest(
            kind="command",
            title="[conceal]hidden title[/conceal]",
            detail="[link=https://example.test]misleading command[/link]",
            reason="[bold red]urgent[/bold red]",
        )

        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ApprovalScreen(request))
            await pilot.pause()

            for selector in (
                "#approval-title",
                "#approval-detail",
                "#approval-reason",
            ):
                widget = app.screen.query_one(selector, Static)
                self.assertFalse(widget._render_markup)
                self.assertIn("[", widget.render().plain)
            await pilot.press("escape")

    async def test_command_policy_auto_allows_and_denies_native_requests(self) -> None:
        with TemporaryDirectory() as directory:
            policy = CommandPolicyStore(Path(directory) / "policy.json")
            policy.add("allow", ["git", "status"])
            policy.add("deny", ["rm"])
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                command_policy=policy,
            )

            async with app.run_test(size=(120, 40)):
                allowed = await app._request_approval(
                    ApprovalRequest("command", "Command", "git status")
                )
                denied = await app._request_approval(
                    ApprovalRequest("command", "Command", "rm important.txt")
                )

                self.assertEqual(allowed, "accept")
                self.assertEqual(denied, "decline")

    async def test_compound_native_command_cannot_inherit_allow_rule(self) -> None:
        with TemporaryDirectory() as directory:
            policy = CommandPolicyStore(Path(directory) / "policy.json")
            policy.add("allow", ["git", "status"])
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                command_policy=policy,
            )

            async with app.run_test(size=(120, 40)):
                approval = AsyncMock(return_value="decline")
                with patch.object(app, "push_screen_wait", approval):
                    decision = await app._request_approval(
                        ApprovalRequest(
                            "command",
                            "Command",
                            "git status && echo unexpected",
                        )
                    )

                self.assertEqual(decision, "decline")
                approval.assert_awaited_once()

    async def test_default_ask_policy_prompts_for_direct_commands(self) -> None:
        with TemporaryDirectory() as directory:
            policy = CommandPolicyStore(Path(directory) / "policy.json")
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                command_policy=policy,
            )

            async with app.run_test(size=(120, 40)):
                approval = AsyncMock(return_value="accept")
                with patch.object(app, "push_screen_wait", approval):
                    await app._authorize_direct_command(
                        ["git", "status"],
                        "command",
                    )

                approval.assert_awaited_once()

    async def test_direct_command_allow_for_session_is_remembered(self) -> None:
        with TemporaryDirectory() as directory:
            policy = CommandPolicyStore(Path(directory) / "policy.json")
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                command_policy=policy,
            )

            async with app.run_test(size=(120, 40)):
                approval = AsyncMock(return_value="acceptForSession")
                with patch.object(app, "push_screen_wait", approval):
                    await app._authorize_direct_command(["git", "status"], "command")
                    await app._authorize_direct_command(["git", "status"], "command")

                approval.assert_awaited_once()

    async def test_denied_direct_command_never_starts(self) -> None:
        with TemporaryDirectory() as directory:
            policy = CommandPolicyStore(Path(directory) / "policy.json")
            policy.add("deny", ["python3"])
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                command_policy=policy,
            )

            async with app.run_test(size=(120, 40)):
                await app._handle_command('/bg python3 -c "print(1)"')

                self.assertEqual(app.background.list(), [])

    async def test_clear_resets_provider_native_context(self) -> None:
        provider = ResettableProvider()
        session = ChatSession(provider)
        session.messages.append(ChatMessage(role="user", content="remember this"))
        app = AgentApp(session)

        async with app.run_test(size=(120, 40)) as pilot:
            await app.action_clear_chat()
            await pilot.pause()

            self.assertEqual(session.messages, [])
            self.assertEqual(provider.reset_count, 1)

    async def test_clear_is_refused_while_a_turn_is_active(self) -> None:
        provider = ResettableProvider()
        session = ChatSession(provider)
        session.messages.append(ChatMessage(role="user", content="keep this"))
        app = AgentApp(session)

        async with app.run_test(size=(120, 40)):
            app._busy = True
            await app.action_clear_chat()

            self.assertEqual(
                session.messages,
                [ChatMessage(role="user", content="keep this")],
            )
            self.assertEqual(provider.reset_count, 0)

    async def test_disabling_codex_workspace_tool_resets_shared_runtime(self) -> None:
        provider = ResettableProvider()
        provider.name = "codex"
        registry = ToolRegistry()
        app = AgentApp(
            ChatSession(provider),
            tool_registry=registry,
        )

        async with app.run_test(size=(120, 40)):
            await app._handle_command("/tool disable git")

            self.assertEqual(provider.reset_count, 1)
            self.assertFalse(registry.codex_workspace_tools_enabled())

    async def test_save_and_load_slash_commands(self) -> None:
        provider = FakeInteractiveProvider()
        session = ChatSession(provider)
        session.messages = []

        with TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory))
            app = AgentApp(session, conversation_store=store)
            async with app.run_test(size=(120, 40)) as pilot:
                await app._handle_command("/save demo")
                session.messages.append(ChatMessage(role="user", content="temporary"))
                await app._handle_command("/load demo")
                await pilot.pause()

                self.assertEqual(session.messages, [])
                self.assertEqual(store.list(), ["demo"])

    async def test_provider_picker_switches_runtime_provider(self) -> None:
        provider = FakeInteractiveProvider()

        def factory(name: str, model: str | None) -> FakeInteractiveProvider:
            replacement = FakeInteractiveProvider()
            replacement.name = name
            replacement.model = model or "account default"
            return replacement

        app = AgentApp(ChatSession(provider), provider_factory=factory)
        async with app.run_test(size=(120, 40)) as pilot:
            await app._switch_provider("openai", "new-model")
            await pilot.pause()

            self.assertEqual(app.session.provider.name, "openai")
            self.assertEqual(app.session.provider.model, "new-model")
            self.assertEqual(app.query_one("#provider", Select).value, "openai")

    async def test_litellm_models_are_discovered_in_background(self) -> None:
        provider = FakeInteractiveProvider()
        provider.name = "litellm"
        provider.model = "model-b"
        fetched = asyncio.Event()

        async def fetch_models(provider_name: str) -> tuple[str, ...]:
            self.assertEqual(provider_name, "litellm")
            fetched.set()
            return ("model-b", "model-a", "model-b")

        app = AgentApp(
            ChatSession(provider),
            model_fetcher=fetch_models,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await asyncio.wait_for(fetched.wait(), timeout=1)
            await pilot.pause()

            self.assertEqual(
                app._available_models["litellm"],
                ("model-a", "model-b"),
            )
            model_input = app.query_one("#model", Input)
            self.assertIn("2 available", model_input.placeholder)
            self.assertEqual(
                await model_input.suggester.get_suggestion("model-a"),
                "model-a",
            )

    async def test_models_command_uses_cached_discovery(self) -> None:
        provider = FakeInteractiveProvider()
        provider.name = "litellm"
        calls = 0

        async def fetch_models(_provider_name: str) -> tuple[str, ...]:
            nonlocal calls
            calls += 1
            return ("first-model", "second-model")

        app = AgentApp(
            ChatSession(provider),
            model_fetcher=fetch_models,
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await app._discover_models("litellm", announce=False)
            calls_after_discovery = calls
            await app._handle_command("/models")
            await pilot.pause()

            self.assertEqual(calls, calls_after_discovery)

    async def test_provider_switch_closes_previous_provider(self) -> None:
        original = ClosableProvider()
        replacement = ClosableProvider()
        replacement.name = "openai"

        def factory(_name: str, _model: str | None) -> ClosableProvider:
            return replacement

        app = AgentApp(ChatSession(original), provider_factory=factory)
        async with app.run_test(size=(120, 40)):
            await app._switch_provider("openai", None)

            self.assertEqual(original.close_count, 1)
            self.assertIs(app.session.provider, replacement)

    async def test_background_command_completes(self) -> None:
        provider = FakeInteractiveProvider()
        app = AgentApp(ChatSession(provider))
        app.command_policy.default_action = "allow"

        async with app.run_test(size=(120, 40)) as pilot:
            await app._handle_command('/bg python3 -c "print(\'ready\')"')
            job = app.background.list()[0]
            await job.task
            await pilot.pause()

            self.assertEqual(job.status, "completed")
            self.assertEqual(job.output, "ready")
            card = app.query_one("#command-outputs").query_one(Collapsible)
            self.assertTrue(card.collapsed)
            body = card.query_one(".command-output-body", Static)
            self.assertIn("ready", body.render().plain)

    async def test_background_command_keeps_only_bounded_output(self) -> None:
        app = AgentApp(ChatSession(FakeInteractiveProvider()))

        async with app.run_test(size=(120, 40)):
            output = await app._run_background_command(
                [
                    "python3",
                    "-c",
                    "import sys; sys.stdout.write('x' * 120000 + 'TAIL')",
                ]
            )

            self.assertLessEqual(len(output.encode()), 100_000)
            self.assertTrue(output.endswith("TAIL"))

    async def test_background_command_reports_nonzero_exit_output(self) -> None:
        app = AgentApp(ChatSession(FakeInteractiveProvider()))

        async with app.run_test(size=(120, 40)):
            with self.assertRaisesRegex(RuntimeError, "failure detail"):
                await app._run_background_command(
                    [
                        "python3",
                        "-c",
                        "import sys; print('failure detail'); sys.exit(7)",
                    ]
                )

    async def test_background_command_honors_wall_time_limit(self) -> None:
        with TemporaryDirectory() as directory:
            limits = ExecutionLimitStore(Path(directory) / "limits.json")
            limits.set("timeout", "0.05s")
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                limit_store=limits,
            )

            async with app.run_test(size=(120, 40)):
                with self.assertRaisesRegex(RuntimeError, "timed out after 0.05s"):
                    await app._run_background_command(
                        ["python3", "-c", "import time; time.sleep(1)"]
                    )

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    async def test_cancelling_background_command_stops_child_processes(self) -> None:
        provider = FakeInteractiveProvider()
        app = AgentApp(ChatSession(provider))

        with TemporaryDirectory() as directory:
            started = Path(directory) / "started"
            survived = Path(directory) / "survived"
            child = (
                "import time; from pathlib import Path; "
                f"Path({str(started)!r}).write_text('started'); "
                "time.sleep(0.5); "
                f"Path({str(survived)!r}).write_text('survived')"
            )
            parent = (
                "import subprocess, sys; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}])"
            )

            async with app.run_test(size=(120, 40)):
                task = asyncio.create_task(
                    app._run_background_command(["python3", "-c", parent])
                )
                for _ in range(100):
                    if started.exists():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.exists())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.6)

                self.assertFalse(survived.exists())

    async def test_prompts_submitted_while_busy_run_in_fifo_order(self) -> None:
        provider = QueuedProvider()
        app = AgentApp(ChatSession(provider))

        async with app.run_test(size=(120, 40)) as pilot:
            prompt = app.query_one("#prompt", Input)
            prompt.value = "first"
            await pilot.press("enter")
            await provider.started.wait()

            self.assertFalse(prompt.disabled)
            prompt.value = "second"
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(len(app.prompt_queue), 1)

            provider.release.set()
            await provider.second_finished.wait()
            await pilot.pause()

            self.assertEqual(provider.prompts, ["first", "second"])
            self.assertEqual(len(app.prompt_queue), 0)

    async def test_subagent_runs_as_a_background_job(self) -> None:
        provider = FakeInteractiveProvider()

        def factory(name: str, model: str | None) -> FakeInteractiveProvider:
            return FakeInteractiveProvider()

        app = AgentApp(ChatSession(provider), provider_factory=factory)
        async with app.run_test(size=(120, 40)) as pilot:
            await app._handle_command("/agent reviewer inspect the parser")
            job = app.background.list()[0]
            await job.task
            await pilot.pause()

            self.assertEqual(job.kind, "subagent")
            self.assertEqual(job.status, "completed")
            self.assertIn("reviewer subagent", job.output)

    async def test_autofix_runs_as_bounded_background_job(self) -> None:
        provider = FakeInteractiveProvider()
        workflow = SimpleNamespace(
            run=AsyncMock(
                return_value=AutoFixReport(
                    (TestRun(("test",), 0, "OK", 10),),
                    (),
                )
            )
        )

        def factory(name: str, model: str | None) -> FakeInteractiveProvider:
            return FakeInteractiveProvider()

        app = AgentApp(
            ChatSession(provider),
            provider_factory=factory,
            quality_workflow=workflow,
        )
        app.command_policy.default_action = "allow"
        async with app.run_test(size=(120, 40)) as pilot:
            await app._handle_command("/autofix --retries 3 -- test")
            job = app.background.list()[0]
            self.assertTrue(app._busy)
            await job.task
            await pilot.pause()

            self.assertEqual(job.kind, "autofix")
            self.assertEqual(job.status, "completed")
            self.assertFalse(app._busy)
            workflow.run.assert_awaited_once()
            self.assertEqual(workflow.run.await_args.kwargs["retries"], 3)

    async def test_review_uses_fresh_provider_and_workspace_diff(self) -> None:
        provider = FakeInteractiveProvider()
        reviewer = FakeInteractiveProvider()
        prompts: list[str] = []

        async def review_complete(messages: list[ChatMessage]) -> str:
            prompts.append(messages[-1].content)
            return "No findings. Risk: low."

        reviewer.complete = review_complete

        def factory(name: str, model: str | None) -> FakeInteractiveProvider:
            return reviewer

        app = AgentApp(ChatSession(provider), provider_factory=factory)
        with patch(
            "ai_agent.app.collect_workspace_diff",
            AsyncMock(return_value="diff --git a/a.py b/a.py\n+change\n"),
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                await app._handle_command("/review security")
                job = app.background.list()[0]
                await job.task
                await pilot.pause()

                self.assertEqual(job.kind, "review")
                self.assertEqual(job.status, "completed")
                self.assertIn("Requested scope: security", prompts[0])
                self.assertIn("+change", prompts[0])

    async def test_patch_command_previews_applies_and_saves_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            subprocess.run(
                ["git", "init"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )
            target = workspace / "note.txt"
            target.write_text("before\n", encoding="utf-8")
            patch_file = workspace / "change.patch"
            patch_file.write_text(
                "diff --git a/note.txt b/note.txt\n"
                "--- a/note.txt\n"
                "+++ b/note.txt\n"
                "@@ -1 +1 @@\n"
                "-before\n"
                "+after\n",
                encoding="utf-8",
            )
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                patch_manager=PatchManager(workspace),
                checkpoint_store=CheckpointStore(
                    workspace / ".ai-agent" / "checkpoints",
                    workspace=workspace,
                ),
            )
            app._request_approval = AsyncMock(return_value="accept")

            async with app.run_test(size=(120, 40)) as pilot:
                await app._handle_command(f"/patch preview {patch_file}")
                self.assertIsNotNone(app.patch_manager.pending)
                await app._handle_command("/patch apply")
                await pilot.pause()

                self.assertEqual(target.read_text(encoding="utf-8"), "after\n")
                self.assertIsNone(app.patch_manager.pending)
                self.assertEqual(len(app.checkpoint_store.list()), 1)

    async def test_auto_activated_skill_injects_hidden_instructions(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            registry = self._skill_registry(workspace, auto_activate=True)
            provider = SkillProvider()
            session = ChatSession(provider)
            app = AgentApp(session, skill_registry=registry)

            async with app.run_test(size=(120, 40)) as pilot:
                prompt = app.query_one("#prompt", Input)
                prompt.value = "Please add tests for parser"
                await pilot.press("enter")
                await provider.completed.wait()
                await pilot.pause()

                self.assertIn("Skill instructions:", provider.received[-1].content)
                self.assertIn("Write focused tests", provider.received[-1].content)
                self.assertEqual(
                    session.messages[0],
                    ChatMessage(role="user", content="Please add tests for parser"),
                )

    async def test_skill_run_command_uses_named_skill(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            registry = self._skill_registry(workspace, auto_activate=False)
            provider = SkillProvider()
            app = AgentApp(ChatSession(provider), skill_registry=registry)

            async with app.run_test(size=(120, 40)) as pilot:
                await app._handle_command("/skill run testing inspect parser")
                await provider.completed.wait()
                await pilot.pause()

                self.assertIn("project skill 'testing'", provider.received[-1].content)
                self.assertTrue(
                    provider.received[-1].content.endswith("inspect parser")
                )

    async def test_theme_accessibility_and_dynamic_suggestions(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            appearance = AppearanceStore(workspace / "appearance.json")
            skills = self._skill_registry(workspace, auto_activate=False)
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                appearance_store=appearance,
                skill_registry=skills,
            )

            async with app.run_test(size=(120, 40)):
                prompt = app.query_one("#prompt", Input)
                suggestion = await prompt.suggester.get_suggestion("/skill r")
                self.assertEqual(suggestion, "/skill run testing ")

                await app._handle_command("/theme nord")
                await app._handle_command("/accessibility contrast on")
                await app._handle_command("/accessibility motion reduced")
                await app._handle_command("/accessibility density compact")

                self.assertEqual(app.theme, "nord")
                self.assertTrue(app.has_class("-high-contrast"))
                self.assertTrue(app.has_class("-compact"))
                self.assertEqual(app.animation_level, "none")

            restored = AppearanceStore(workspace / "appearance.json")
            self.assertEqual(restored.settings.theme, "nord")
            self.assertTrue(restored.settings.high_contrast)

    async def test_unavailable_saved_theme_is_replaced_with_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "appearance.json"
            appearance = AppearanceStore(path)
            appearance.update("theme", "removed-theme")
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                appearance_store=appearance,
            )

            async with app.run_test(size=(120, 40)):
                self.assertEqual(app.theme, "textual-dark")
                self.assertEqual(appearance.settings.theme, "textual-dark")

            restored = AppearanceStore(path)
            self.assertEqual(restored.settings.theme, "textual-dark")

    async def test_limit_commands_update_shared_autofix_limits(self) -> None:
        with TemporaryDirectory() as directory:
            limits = ExecutionLimitStore(Path(directory) / "limits.json")
            app = AgentApp(
                ChatSession(FakeInteractiveProvider()),
                limit_store=limits,
            )

            async with app.run_test(size=(120, 40)):
                await app._handle_command("/limit set output 8kb")
                await app._handle_command("/limit set memory 256mb")

                self.assertEqual(limits.limits.output_bytes, 8192)
                self.assertEqual(limits.limits.memory_mb, 256)
                self.assertIs(app.quality_workflow.limits, limits.limits)

    @staticmethod
    def _skill_registry(workspace: Path, *, auto_activate: bool) -> SkillRegistry:
        directory = workspace / "skills" / "testing"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            "---\n"
            "name: testing\n"
            "description: Add focused tests\n"
            "triggers: [add tests]\n"
            f"auto_activate: {'true' if auto_activate else 'false'}\n"
            "---\n"
            "Write focused tests.\n",
            encoding="utf-8",
        )
        return SkillRegistry(workspace)

    async def test_goal_continues_until_provider_marks_it_complete(self) -> None:
        provider = GoalProvider()
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))
            app = AgentApp(ChatSession(provider), goal_store=store)

            async with app.run_test(size=(120, 40)) as pilot:
                await app._handle_command("/goal start --timeout 5s finish parser")
                await provider.completed.wait()
                await pilot.pause()

                goal = store.list()[0]
                self.assertEqual(goal.status, "completed")
                self.assertEqual(goal.attempts, 2)
                self.assertFalse(app._busy)
                self.assertEqual(app.session.messages, [])
                self.assertEqual(provider.reset_count, 1)

    async def test_goal_timeout_stops_an_active_provider_request(self) -> None:
        provider = HangingGoalProvider()
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))
            app = AgentApp(ChatSession(provider), goal_store=store)

            async with app.run_test(size=(120, 40)) as pilot:
                await app._handle_command("/goal start --timeout 0.05s wait forever")
                await provider.started.wait()
                await asyncio.sleep(0.1)
                await pilot.pause()

                goal = store.list()[0]
                self.assertEqual(goal.status, "timed_out")
                self.assertFalse(app._busy)

    async def test_active_goal_resumes_when_app_mounts(self) -> None:
        provider = GoalProvider()
        with TemporaryDirectory() as directory:
            store = GoalStore(Path(directory))
            saved = store.create("resume this work", 5)
            saved.attempts = 3
            saved.last_result = "Previous progress"
            store.save(saved)
            app = AgentApp(ChatSession(provider), goal_store=store)

            async with app.run_test(size=(120, 40)) as pilot:
                await provider.completed.wait()
                await pilot.pause()

                restored = store.load(saved.id)
                self.assertEqual(restored.status, "completed")
                self.assertEqual(restored.attempts, 5)


if __name__ == "__main__":
    unittest.main()
