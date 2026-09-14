import asyncio
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ai_agent.events import ApprovalRequest, DiffUpdate, PlanUpdate
from ai_agent.messages import ChatMessage
from ai_agent.providers.codex import CodexProvider
from ai_agent.tools import ToolPlugin, ToolRegistry


class CodexProviderTests(unittest.TestCase):
    @patch("ai_agent.providers.codex.shutil.which", return_value=None)
    def test_missing_executable_has_actionable_error(self, _which: object) -> None:
        with self.assertRaisesRegex(ValueError, "CODEX_PATH"):
            CodexProvider()

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_uses_account_default_model(self, _which: object) -> None:
        provider = CodexProvider()

        self.assertEqual(provider.model, "account default")
        self.assertIsNone(provider.requested_model)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_explicit_model_is_preserved(self, _which: object) -> None:
        provider = CodexProvider(model="example-model")

        self.assertEqual(provider.model, "example-model")
        self.assertEqual(provider.requested_model, "example-model")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_reasoning_effort_is_validated(self, _which: object) -> None:
        self.assertEqual(CodexProvider(effort="high").effort, "high")
        with self.assertRaisesRegex(ValueError, "effort must be"):
            CodexProvider(effort="extreme")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_effort_can_change_on_running_thread(self, _which: object) -> None:
        provider = CodexProvider(effort="low")
        provider._thread_id = "thread-1"
        provider._request = AsyncMock(return_value={})

        asyncio.run(provider.set_effort("high"))

        self.assertEqual(provider.effort, "high")
        provider._request.assert_awaited_once_with(
            "thread/settings/update",
            {"threadId": "thread-1", "effort": "high"},
        )

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_initial_effort_is_applied_to_turn_not_thread(self, _which: object) -> None:
        provider = CodexProvider(effort="high")
        provider._thread_id = "thread-1"

        self.assertNotIn("effort", provider._thread_params())
        self.assertEqual(provider._turn_params("hello")["effort"], "high")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_complete_requires_a_final_user_message(self, _which: object) -> None:
        provider = CodexProvider()

        for messages in (
            [],
            [ChatMessage(role="assistant", content="answer")],
        ):
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ValueError, "final user message"):
                    asyncio.run(provider.complete(messages))

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_successful_completion_records_metrics(self, _which: object) -> None:
        provider = CodexProvider(effort="medium")
        provider._thread_id = "thread-1"
        provider._ensure_started = AsyncMock()
        provider._request = AsyncMock(return_value={"turn": {"id": "turn-1"}})
        provider._read_turn = AsyncMock(return_value="answer")

        reply = asyncio.run(
            provider.complete([ChatMessage(role="user", content="hello")])
        )

        self.assertEqual(reply, "answer")
        request = provider._request.await_args.args
        self.assertEqual(request[0], "turn/start")
        self.assertEqual(request[1]["effort"], "medium")
        self.assertEqual(provider.session_metrics.requests, 1)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_cancelled_turn_is_aborted(self, _which: object) -> None:
        provider = CodexProvider()
        provider._thread_id = "thread-1"
        provider._ensure_started = AsyncMock()
        provider._request = AsyncMock(return_value={"turn": {"id": "turn-1"}})
        provider._read_turn = AsyncMock(side_effect=asyncio.CancelledError())
        provider._abort_turn = AsyncMock()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(
                provider.complete([ChatMessage(role="user", content="hello")])
            )

        provider._abort_turn.assert_awaited_once_with("turn-1")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_timed_out_turn_is_aborted(self, _which: object) -> None:
        provider = CodexProvider()
        provider._thread_id = "thread-1"
        provider._ensure_started = AsyncMock()
        provider._request = AsyncMock(return_value={"turn": {"id": "turn-1"}})
        provider._read_turn = AsyncMock(side_effect=asyncio.TimeoutError())
        provider._abort_turn = AsyncMock()

        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                provider.complete([ChatMessage(role="user", content="hello")])
            )

        provider._abort_turn.assert_awaited_once_with("turn-1")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_abort_uses_turn_interrupt_then_closes(self, _which: object) -> None:
        provider = CodexProvider()
        provider._thread_id = "thread-1"
        provider._process = SimpleNamespace(returncode=None)
        provider._request = AsyncMock(return_value={})
        provider.close = AsyncMock()

        asyncio.run(provider._abort_turn("turn-1"))

        provider._request.assert_awaited_once_with(
            "turn/interrupt",
            {"threadId": "thread-1", "turnId": "turn-1"},
        )
        provider.close.assert_awaited_once_with()

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_reset_discards_native_thread_state(self, _which: object) -> None:
        provider = CodexProvider()
        provider._thread_id = "thread-1"
        provider._needs_history_seed = False
        provider.context_tokens = 50
        provider.context_window = 100

        asyncio.run(provider.reset())

        self.assertIsNone(provider._thread_id)
        self.assertTrue(provider._needs_history_seed)
        self.assertEqual(provider.context_tokens, 0)
        self.assertIsNone(provider.context_window)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_disabling_workspace_tools_disables_native_runtime(
        self, _which: object
    ) -> None:
        registry = ToolRegistry()
        registry.set_enabled("filesystem", False)
        provider = CodexProvider(tool_registry=registry)

        command = provider._server_command()
        params = provider._thread_params()

        self.assertIn("shell_tool", command)
        self.assertIn("unified_exec", command)
        self.assertFalse(params["config"]["features"]["shell_tool"])
        self.assertFalse(params["config"]["features"]["unified_exec"])
        self.assertIn("tools are disabled", params["developerInstructions"])

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_disabled_workspace_runtime_auto_declines_native_approval(
        self, _which: object
    ) -> None:
        registry = ToolRegistry()
        registry.set_enabled("command", False)
        approval = AsyncMock(return_value="accept")
        provider = CodexProvider(
            tool_registry=registry,
            approval_handler=approval,
        )
        provider._send = AsyncMock()

        asyncio.run(
            provider._handle_server_request(
                {
                    "id": 8,
                    "method": "item/commandExecution/requestApproval",
                    "params": {"command": "git status"},
                }
            )
        )

        approval.assert_not_awaited()
        provider._send.assert_awaited_once_with(
            {"id": 8, "result": {"decision": "decline"}}
        )

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_filesystem_tools_are_scoped_to_workspace(self, _which: object) -> None:
        provider = CodexProvider(cwd=Path("/tmp/example-workspace"))

        params = provider._thread_params()

        self.assertEqual(params["sandbox"], "read-only")
        self.assertEqual(params["approvalPolicy"], "on-request")
        self.assertEqual(params["cwd"], "/tmp/example-workspace")
        self.assertIn("filesystem tools", params["developerInstructions"])
        instructions = params["developerInstructions"]
        self.assertIn("run non-destructive local commands", instructions)
        self.assertIn("status, diff, log, show", instructions)
        self.assertIn("blocking approval dialogue", instructions)
        self.assertIn("access paths outside the workspace", instructions)
        self.assertEqual(params["config"]["web_search"], "live")
        self.assertIn("live web search", instructions)

    def test_formats_protocol_errors(self) -> None:
        self.assertEqual(
            CodexProvider._format_error({"message": "request failed"}),
            "request failed",
        )

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_context_percentage_uses_protocol_token_usage(self, _which: object) -> None:
        provider = CodexProvider()

        provider._update_token_usage(
            {
                "last": {"totalTokens": 20_000},
                "modelContextWindow": 100_000,
            }
        )

        self.assertEqual(provider.context_tokens, 20_000)
        self.assertEqual(provider.context_window, 100_000)
        self.assertEqual(provider.context_percent, 20.0)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_auto_compact_threshold_is_validated(self, _which: object) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            CodexProvider(auto_compact_threshold=1)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_compacts_when_context_reaches_threshold(self, _which: object) -> None:
        provider = CodexProvider()
        provider._thread_id = "thread-1"
        provider.context_tokens = 80_000
        provider.context_window = 100_000
        provider._request = AsyncMock(return_value={})
        provider._next_message = AsyncMock(
            side_effect=[
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "tokenUsage": {
                            "last": {"totalTokens": 12_000},
                            "modelContextWindow": 100_000,
                        }
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {"turn": {"status": "completed"}},
                },
            ]
        )

        asyncio.run(provider._maybe_compact())

        provider._request.assert_awaited_once_with(
            "thread/compact/start", {"threadId": "thread-1"}
        )
        self.assertEqual(provider.context_percent, 12.0)
        self.assertEqual(provider.compaction_count, 1)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_does_not_compact_below_threshold(self, _which: object) -> None:
        provider = CodexProvider()
        provider.context_tokens = 79_999
        provider.context_window = 100_000
        provider._request = AsyncMock()

        asyncio.run(provider._maybe_compact())

        provider._request.assert_not_awaited()

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_failed_turn_reports_protocol_error(self, _which: object) -> None:
        provider = CodexProvider()
        provider._next_message = AsyncMock(
            return_value={
                "method": "turn/completed",
                "params": {
                    "turnId": "turn-1",
                    "turn": {
                        "status": "failed",
                        "error": {"message": "model unavailable"},
                    },
                },
            }
        )

        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            asyncio.run(provider._read_turn("turn-1"))

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_request_surfaces_json_rpc_errors(self, _which: object) -> None:
        provider = CodexProvider()
        provider._send = AsyncMock()
        provider._read_message = AsyncMock(
            return_value={
                "id": 1,
                "error": {"message": "invalid request"},
            }
        )

        with self.assertRaisesRegex(RuntimeError, "invalid request"):
            asyncio.run(provider._request("thread/start", {}))

        provider._send.assert_awaited_once()

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_command_activity_has_output_and_duration(self, _which: object) -> None:
        provider = CodexProvider()

        activity = provider._activity_from_item(
            {
                "item": {
                    "type": "commandExecution",
                    "command": "python -m unittest",
                    "status": "completed",
                    "exitCode": 0,
                    "durationMs": 1250,
                    "aggregatedOutput": "OK",
                }
            },
            "completed",
        )

        self.assertEqual(activity.kind, "command")
        self.assertEqual(activity.status, "completed")
        self.assertIn("exit 0", activity.detail)
        self.assertEqual(activity.output, "OK")
        self.assertEqual(activity.duration_ms, 1250)

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_approval_request_pauses_for_handler_decision(self, _which: object) -> None:
        activities = []

        async def record(activity: object) -> None:
            activities.append(activity)

        async def approve(request: ApprovalRequest) -> str:
            self.assertEqual(request.detail, "git commit -m test")
            return "accept"

        provider = CodexProvider(
            activity_handler=record,
            approval_handler=approve,
        )
        provider._send = AsyncMock()

        asyncio.run(
            provider._handle_server_request(
                {
                    "id": 42,
                    "method": "item/commandExecution/requestApproval",
                    "params": {
                        "command": "git commit -m test",
                        "reason": "Creates a commit",
                    },
                }
            )
        )

        provider._send.assert_awaited_once_with(
            {"id": 42, "result": {"decision": "accept"}}
        )
        self.assertEqual(
            [activity.status for activity in activities],
            ["approval", "accept"],
        )

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_unknown_approval_decision_fails_closed(self, _which: object) -> None:
        async def invalid_decision(_request: ApprovalRequest) -> str:
            return "always"

        provider = CodexProvider(approval_handler=invalid_decision)
        provider._send = AsyncMock()

        asyncio.run(
            provider._handle_server_request(
                {
                    "id": 12,
                    "method": "item/commandExecution/requestApproval",
                    "params": {"command": "python -m unittest"},
                }
            )
        )

        provider._send.assert_awaited_once_with(
            {"id": 12, "result": {"decision": "decline"}}
        )

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_permission_approval_returns_requested_grant(self, _which: object) -> None:
        async def approve(_request: ApprovalRequest) -> str:
            return "acceptForSession"

        provider = CodexProvider(approval_handler=approve)
        provider._send = AsyncMock()
        permissions = {"network": {"enabled": True}}

        asyncio.run(
            provider._handle_server_request(
                {
                    "id": 9,
                    "method": "item/permissions/requestApproval",
                    "params": {
                        "permissions": permissions,
                        "reason": "Download a dependency",
                    },
                }
            )
        )

        provider._send.assert_awaited_once_with(
            {
                "id": 9,
                "result": {"permissions": permissions, "scope": "session"},
            }
        )

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_plan_and_diff_notifications_are_emitted(self, _which: object) -> None:
        plans: list[PlanUpdate] = []
        diffs: list[DiffUpdate] = []

        async def show_plan(update: PlanUpdate) -> None:
            plans.append(update)

        async def show_diff(update: DiffUpdate) -> None:
            diffs.append(update)

        provider = CodexProvider(plan_handler=show_plan, diff_handler=show_diff)
        asyncio.run(
            provider._emit_plan(
                {
                    "explanation": "Implementation",
                    "plan": [
                        {"step": "Add UI", "status": "inProgress"},
                        {"step": "Test", "status": "pending"},
                    ],
                }
            )
        )
        asyncio.run(provider._emit_diff({"diff": "+new line"}))

        self.assertEqual(plans[0].steps[0].text, "Add UI")
        self.assertEqual(plans[0].steps[0].status, "inProgress")
        self.assertEqual(diffs[0].diff, "+new line")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_turn_routes_plan_and_diff_updates(self, _which: object) -> None:
        plans: list[PlanUpdate] = []
        diffs: list[DiffUpdate] = []

        async def show_plan(update: PlanUpdate) -> None:
            plans.append(update)

        async def show_diff(update: DiffUpdate) -> None:
            diffs.append(update)

        provider = CodexProvider(plan_handler=show_plan, diff_handler=show_diff)
        provider._next_message = AsyncMock(
            side_effect=[
                {
                    "method": "turn/plan/updated",
                    "params": {
                        "turnId": "turn-1",
                        "plan": [{"step": "Inspect", "status": "completed"}],
                    },
                },
                {
                    "method": "turn/diff/updated",
                    "params": {"turnId": "turn-1", "diff": "+change"},
                },
                {
                    "method": "item/completed",
                    "params": {
                        "turnId": "turn-1",
                        "item": {
                            "id": "message-1",
                            "type": "agentMessage",
                            "text": "Done",
                        },
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "turnId": "turn-1",
                        "turn": {"status": "completed"},
                    },
                },
            ]
        )

        result = asyncio.run(provider._read_turn("turn-1"))

        self.assertEqual(result, "Done")
        self.assertEqual(plans[0].steps[0].text, "Inspect")
        self.assertEqual(diffs[0].diff, "+change")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_registered_dynamic_tool_uses_approval(self, _which: object) -> None:
        async def greet(arguments: dict[str, object]) -> str:
            return f"Hello {arguments['name']}"

        async def approve(_request: ApprovalRequest) -> str:
            return "accept"

        registry = ToolRegistry()
        registry.register(
            ToolPlugin(
                "greet",
                "Greet a user",
                frozenset({"codex"}),
                input_schema={"type": "object"},
                handler=greet,
            )
        )
        provider = CodexProvider(
            approval_handler=approve,
            tool_registry=registry,
        )
        provider._send = AsyncMock()

        asyncio.run(
            provider._handle_server_request(
                {
                    "id": 71,
                    "method": "item/tool/call",
                    "params": {"tool": "greet", "arguments": {"name": "Itai"}},
                }
            )
        )

        provider._send.assert_awaited_once_with(
            {
                "id": 71,
                "result": {
                    "contentItems": [{"type": "inputText", "text": "Hello Itai"}],
                    "success": True,
                },
            }
        )
        self.assertEqual(provider._thread_params()["dynamicTools"][0]["name"], "greet")

    @patch("ai_agent.providers.codex.shutil.which", return_value="/bin/codex")
    def test_restored_history_seeds_new_thread(self, _which: object) -> None:
        provider = CodexProvider()
        messages = [
            ChatMessage(role="user", content="First"),
            ChatMessage(role="assistant", content="Earlier answer"),
            ChatMessage(role="user", content="Continue"),
        ]

        prompt = provider._turn_prompt(messages)

        self.assertIn("Earlier answer", prompt)
        self.assertIn("User: Continue", prompt)
