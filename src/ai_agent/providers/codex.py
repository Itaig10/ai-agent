import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
import json
from pathlib import Path
import shutil
from time import monotonic
from typing import Any

from ai_agent.events import (
    ApprovalRequest,
    DiffUpdate,
    PlanStep,
    PlanUpdate,
    TextDelta,
    ToolActivity,
)
from ai_agent.messages import ChatMessage
from ai_agent.metrics import UsageMetrics
from ai_agent.tools import ToolRegistry


ActivityHandler = Callable[[ToolActivity], Awaitable[None]]
ApprovalHandler = Callable[[ApprovalRequest], Awaitable[str]]
PlanHandler = Callable[[PlanUpdate], Awaitable[None]]
DiffHandler = Callable[[DiffUpdate], Awaitable[None]]
TextDeltaHandler = Callable[[TextDelta], Awaitable[None]]


class CodexProvider:
    """Chat provider backed by the local Codex app-server process."""

    name = "codex"

    def __init__(
        self,
        *,
        command: str = "codex",
        model: str | None = None,
        effort: str | None = None,
        cwd: Path | None = None,
        timeout_seconds: float = 300,
        auto_compact_threshold: float = 0.8,
        activity_handler: ActivityHandler | None = None,
        approval_handler: ApprovalHandler | None = None,
        plan_handler: PlanHandler | None = None,
        diff_handler: DiffHandler | None = None,
        text_delta_handler: TextDeltaHandler | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        executable = shutil.which(command)
        if executable is None:
            raise ValueError(
                f"Codex executable {command!r} was not found. Install Codex or set "
                "CODEX_PATH."
            )

        self.command = executable
        self.requested_model = model
        self.model = model or "account default"
        self.effort = self._validate_effort(effort)
        self.cwd = (cwd or Path.cwd()).resolve()
        self.timeout_seconds = timeout_seconds
        if not 0 < auto_compact_threshold < 1:
            raise ValueError("auto_compact_threshold must be between 0 and 1")
        self.auto_compact_threshold = auto_compact_threshold
        self.context_tokens = 0
        self.context_window: int | None = None
        self.context_window_supported = True
        self.compaction_count = 0
        self.last_metrics = UsageMetrics()
        self.session_metrics = UsageMetrics()
        self.tool_registry = tool_registry or ToolRegistry()
        self._activity_handler = activity_handler
        self._approval_handler = approval_handler
        self._plan_handler = plan_handler
        self._diff_handler = diff_handler
        self._text_delta_handler = text_delta_handler
        self._active_items: dict[str, dict[str, Any]] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._thread_id: str | None = None
        self._needs_history_seed = True
        self._next_request_id = 1
        self._notifications: deque[dict[str, Any]] = deque()
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        if not messages or messages[-1].role != "user":
            raise ValueError("Codex requires a final user message")

        async with self._lock:
            turn_id: str | None = None
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    started = monotonic()
                    self.last_metrics = UsageMetrics()
                    await self._ensure_started()
                    prompt = self._turn_prompt(messages)
                    result = await self._request(
                        "turn/start",
                        self._turn_params(prompt),
                    )
                    self._needs_history_seed = False
                    turn_id = result["turn"]["id"]
                    reply = await self._read_turn(turn_id)
                    self.last_metrics.latency_ms = round(
                        (monotonic() - started) * 1000
                    )
                    self.last_metrics.requests = 1
                    self.session_metrics.add(self.last_metrics)
                    return reply
            except (asyncio.TimeoutError, asyncio.CancelledError):
                await self._abort_turn(turn_id)
                raise

    def _turn_params(self, prompt: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": prompt}],
        }
        if self.effort:
            params["effort"] = self.effort
        return params

    async def _abort_turn(self, turn_id: str | None) -> None:
        """Interrupt an active turn and discard its transport state."""

        try:
            if (
                turn_id is not None
                and self._thread_id is not None
                and self._process is not None
                and self._process.returncode is None
            ):
                with suppress(Exception):
                    await asyncio.wait_for(
                        self._request(
                            "turn/interrupt",
                            {"threadId": self._thread_id, "turnId": turn_id},
                        ),
                        timeout=2,
                    )
        finally:
            # A fresh process/thread prevents late notifications from a cancelled
            # turn leaking into the next request.
            await self.close()

    async def reset(self) -> None:
        """Discard native Codex conversation state while keeping configuration."""

        await self.close()

    async def close(self) -> None:
        process = self._process
        self._process = None
        self._thread_id = None
        self._needs_history_seed = True
        self.context_tokens = 0
        self.context_window = None
        self._active_items.clear()
        self._notifications.clear()
        self._stderr_lines.clear()
        self._next_request_id = 1
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None

    async def _ensure_started(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return

        self._process = await asyncio.create_subprocess_exec(
            *self._server_command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        await self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "ai-agent-tui",
                    "title": "AI Agent TUI",
                    "version": "0.1.0",
                }
            },
        )
        await self._send({"method": "initialized"})

        thread_params = self._thread_params()
        if self.requested_model:
            thread_params["model"] = self.requested_model

        thread_result = await self._request("thread/start", thread_params)
        self._thread_id = thread_result["thread"]["id"]

    def _server_command(self) -> list[str]:
        command = [self.command, "app-server", "--stdio"]
        if not self.tool_registry.codex_workspace_tools_enabled():
            # Both native shell implementations are disabled at process startup.
            # Switching a workspace tool restarts this provider process.
            command.extend(
                ["--disable", "shell_tool", "--disable", "unified_exec"]
            )
        return command

    def _turn_prompt(self, messages: Sequence[ChatMessage]) -> str:
        if not self._needs_history_seed or len(messages) == 1:
            return messages[-1].content
        history = "\n\n".join(
            f"{message.role.title()}: {message.content}" for message in messages[:-1]
        )
        return (
            "Continue this restored conversation. Treat the transcript as context:\n\n"
            f"{history}\n\nUser: {messages[-1].content}"
        )

    def _thread_params(self) -> dict[str, Any]:
        """Build a workspace-scoped Codex session with local tools enabled."""
        enabled = {
            name
            for name in ("filesystem", "command", "git", "web_search")
            if self.tool_registry.is_enabled(name, self.name)
        }
        workspace_tools = self.tool_registry.codex_workspace_tools_enabled()
        workspace_instructions = (
            "You may use filesystem tools to list, search, read, create, and edit "
            "files inside the current workspace when the user asks. You may also "
            "run non-destructive local commands in the workspace, including tests, "
            "linters, formatters, and project scripts. You may inspect Git "
            "repositories with status, diff, log, show, and branch-list commands. "
            if workspace_tools
            else "Filesystem, shell-command, and Git tools are disabled for this "
            "runtime. "
        )
        return {
            "approvalPolicy": "on-request",
            "cwd": str(self.cwd),
            "developerInstructions": (
                "Act as a general-purpose conversational assistant. "
                + workspace_instructions
                + "Use live web search for current, uncertain, or explicitly "
                "requested online information, and cite the sources in the response. "
                "Maintain a concise task plan for multi-step work and keep its "
                "statuses current. Only attempt file writes, destructive commands, "
                "package installation, network access, or Git mutations when "
                "explicitly "
                "requested by the user; the client will present a blocking approval "
                "dialogue before execution. Never access paths outside the workspace. "
                "Preserve existing files unless the user explicitly asks to change "
                "them, and explain the commands run and files changed."
            ),
            "config": {
                "web_search": "live" if "web_search" in enabled else "disabled",
                "features": {
                    "shell_tool": workspace_tools,
                    "unified_exec": workspace_tools,
                },
            },
            "dynamicTools": self.tool_registry.dynamic_specs(self.name),
            "ephemeral": True,
            "sandbox": "read-only",
        }

    def set_event_handlers(
        self,
        *,
        activity: ActivityHandler,
        approval: ApprovalHandler,
        plan: PlanHandler | None = None,
        diff: DiffHandler | None = None,
        text_delta: TextDeltaHandler | None = None,
    ) -> None:
        self._activity_handler = activity
        self._approval_handler = approval
        self._plan_handler = plan
        self._diff_handler = diff
        self._text_delta_handler = text_delta

    async def set_effort(self, effort: str | None) -> None:
        effort = self._validate_effort(effort)
        async with self._lock:
            self.effort = effort
            if self._thread_id is not None:
                await self._request(
                    "thread/settings/update",
                    {"threadId": self._thread_id, "effort": effort},
                )

    @staticmethod
    def _validate_effort(effort: str | None) -> str | None:
        if effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("effort must be low, medium, high, or xhigh")
        return effort

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        await self._send({"id": request_id, "method": method, "params": params})

        while True:
            message = await self._read_message()
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(
                        f"Codex {method} failed: {self._format_error(message['error'])}"
                    )
                return message.get("result", {})
            self._notifications.append(message)

    async def _read_turn(self, turn_id: str) -> str:
        final_text = ""
        streamed_text: list[str] = []

        while True:
            message = await self._next_message()
            method = message.get("method")
            params = message.get("params", {})

            if params.get("turnId") not in (None, turn_id):
                continue

            if method == "thread/tokenUsage/updated":
                self._update_token_usage(params.get("tokenUsage", {}))
            elif method == "turn/plan/updated":
                await self._emit_plan(params)
            elif method == "turn/diff/updated":
                await self._emit_diff(params)
            elif method == "item/started":
                item = params.get("item", {})
                if item.get("id"):
                    self._active_items[item["id"]] = item
                await self._emit_activity(self._activity_from_item(params, "running"))
            elif method == "item/agentMessage/delta":
                delta = params.get("delta", "")
                streamed_text.append(delta)
                if delta and self._text_delta_handler is not None:
                    await self._text_delta_handler(TextDelta(delta))
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage":
                    final_text = item.get("text", "")
                else:
                    await self._emit_activity(
                        self._activity_from_item(params, "completed")
                    )
                if item.get("id"):
                    self._active_items.pop(item["id"], None)
            elif method == "turn/completed":
                turn = params.get("turn", {})
                if turn.get("status") == "failed":
                    raise RuntimeError(
                        f"Codex turn failed: {self._format_error(turn.get('error'))}"
                    )
                await self._maybe_compact()
                return final_text or "".join(streamed_text)
            elif "id" in message and method:
                await self._handle_server_request(message)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        method = message.get("method", "")
        params = message.get("params", {})
        if method == "item/tool/call":
            await self._handle_dynamic_tool(message)
            return
        if method == "item/commandExecution/requestApproval":
            request = ApprovalRequest(
                kind="command",
                title="Approve command",
                detail=params.get("command") or "Unknown command",
                reason=params.get("reason") or "",
            )
        elif method == "item/fileChange/requestApproval":
            active_item = self._active_items.get(params.get("itemId", ""), {})
            changes = active_item.get("changes", [])
            changed_paths = ", ".join(
                f"{change.get('kind', 'update')}: {change.get('path', '?')}"
                for change in changes
            )
            request = ApprovalRequest(
                kind="file",
                title="Approve file changes",
                detail=(
                    changed_paths
                    or params.get("grantRoot")
                    or "Changes inside the workspace"
                ),
                reason=params.get("reason") or "",
            )
        elif method == "item/permissions/requestApproval":
            permissions = params.get("permissions", {})
            request = ApprovalRequest(
                kind="permissions",
                title="Approve additional permissions",
                detail=json.dumps(permissions, indent=2),
                reason=params.get("reason") or "",
            )
        else:
            raise RuntimeError(f"Codex requested unsupported client action {method!r}")

        await self._emit_activity(
            ToolActivity(request.kind, "approval", request.title, request.detail)
        )
        decision = "decline"
        workspace_request = method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "item/permissions/requestApproval",
        }
        if (
            self._approval_handler is not None
            and not (
                workspace_request
                and not self.tool_registry.codex_workspace_tools_enabled()
            )
        ):
            decision = await self._approval_handler(request)
        if decision not in {"accept", "acceptForSession", "decline", "cancel"}:
            decision = "decline"
        if method == "item/permissions/requestApproval":
            granted = (
                params.get("permissions", {})
                if decision.startswith("accept")
                else {}
            )
            result = {
                "permissions": granted,
                "scope": "session" if decision == "acceptForSession" else "turn",
            }
        else:
            result = {"decision": decision}
        await self._send({"id": message["id"], "result": result})
        await self._emit_activity(
            ToolActivity(request.kind, decision, request.title, request.detail)
        )

    async def _handle_dynamic_tool(self, message: dict[str, Any]) -> None:
        params = message.get("params", {})
        name = str(params.get("tool") or "")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        await self._emit_activity(
            ToolActivity("plugin", "running", f"Plugin: {name}", str(arguments))
        )
        try:
            plugin = self.tool_registry.get(name)
            if not self.tool_registry.is_enabled(name, self.name):
                raise ValueError(f"Tool {name!r} is disabled")
            if plugin.requires_approval:
                request = ApprovalRequest(
                    kind="plugin",
                    title=f"Approve plugin: {name}",
                    detail=json.dumps(arguments, indent=2),
                    reason="A registered client tool requested execution",
                )
                decision = (
                    await self._approval_handler(request)
                    if self._approval_handler is not None
                    else "decline"
                )
                if not decision.startswith("accept"):
                    raise PermissionError("User declined plugin execution")
            output = await self.tool_registry.execute(name, arguments)
            result = {
                "contentItems": [{"type": "inputText", "text": output}],
                "success": True,
            }
            status = "completed"
        except Exception as error:
            result = {
                "contentItems": [{"type": "inputText", "text": str(error)}],
                "success": False,
            }
            status = "failed"
        await self._send({"id": message["id"], "result": result})
        await self._emit_activity(
            ToolActivity(
                "plugin",
                status,
                f"Plugin: {name}",
                output=result["contentItems"][0]["text"],
            )
        )

    def _activity_from_item(
        self, params: dict[str, Any], lifecycle: str
    ) -> ToolActivity | None:
        item = params.get("item", {})
        item_type = item.get("type", "tool")
        duration = item.get("durationMs")
        if item_type == "commandExecution":
            status = item.get("status", lifecycle)
            exit_code = item.get("exitCode")
            detail = item.get("command", "")
            if exit_code is not None:
                detail = f"{detail}  [exit {exit_code}]"
            return ToolActivity(
                kind="command",
                status=status,
                title="Command",
                detail=detail,
                output=item.get("aggregatedOutput") or "",
                duration_ms=duration,
            )
        if item_type == "fileChange":
            changes = item.get("changes", [])
            detail = ", ".join(
                f"{change.get('kind', 'update')}: {change.get('path', '?')}"
                for change in changes
            )
            return ToolActivity(
                kind="file",
                status=item.get("status", lifecycle),
                title="File change",
                detail=detail,
                duration_ms=duration,
            )
        if item_type in {"mcpToolCall", "webSearch"}:
            return ToolActivity(
                kind=item_type,
                status=item.get("status", lifecycle),
                title="MCP tool" if item_type == "mcpToolCall" else "Web search",
                detail=str(item.get("name") or item.get("query") or ""),
                duration_ms=duration,
            )
        return None

    async def _emit_activity(self, activity: ToolActivity | None) -> None:
        if activity is not None and self._activity_handler is not None:
            await self._activity_handler(activity)

    async def _emit_plan(self, params: dict[str, Any]) -> None:
        if self._plan_handler is None:
            return
        steps = tuple(
            PlanStep(
                text=str(step.get("step", "")),
                status=str(step.get("status", "pending")),
            )
            for step in params.get("plan", [])
            if step.get("step")
        )
        await self._plan_handler(
            PlanUpdate(steps=steps, explanation=params.get("explanation") or "")
        )

    async def _emit_diff(self, params: dict[str, Any]) -> None:
        if self._diff_handler is not None:
            await self._diff_handler(DiffUpdate(diff=params.get("diff") or ""))

    @property
    def context_percent(self) -> float | None:
        if not self.context_window:
            return None
        return min(100.0, self.context_tokens / self.context_window * 100)

    def _update_token_usage(
        self, token_usage: dict[str, Any], *, record_metrics: bool = True
    ) -> None:
        last = token_usage.get("last", {})
        self.context_tokens = int(last.get("totalTokens", 0))
        if record_metrics:
            self.last_metrics.input_tokens = int(last.get("inputTokens", 0))
            self.last_metrics.output_tokens = int(last.get("outputTokens", 0))
            self.last_metrics.total_tokens = self.context_tokens
        window = token_usage.get("modelContextWindow")
        self.context_window = int(window) if window else None

    async def _maybe_compact(self) -> None:
        percent = self.context_percent
        if percent is None or percent < self.auto_compact_threshold * 100:
            return

        await self._request(
            "thread/compact/start",
            {"threadId": self._thread_id},
        )
        while True:
            message = await self._next_message()
            method = message.get("method")
            params = message.get("params", {})
            if method == "thread/tokenUsage/updated":
                self._update_token_usage(
                    params.get("tokenUsage", {}), record_metrics=False
                )
            elif method == "turn/completed":
                turn = params.get("turn", {})
                if turn.get("status") == "failed":
                    raise RuntimeError(
                        "Codex compaction failed: "
                        f"{self._format_error(turn.get('error'))}"
                    )
                self.compaction_count += 1
                return

    async def _next_message(self) -> dict[str, Any]:
        if self._notifications:
            return self._notifications.popleft()
        return await self._read_message()

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError("Codex app server stdin is unavailable")
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        process = self._require_process()
        if process.stdout is None:
            raise RuntimeError("Codex app server stdout is unavailable")
        line = await process.stdout.readline()
        if not line:
            details = "\n".join(self._stderr_lines)
            suffix = f"\n{details}" if details else ""
            raise RuntimeError(
                f"Codex app server exited unexpectedly ({process.returncode}).{suffix}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("Codex app server returned invalid JSON") from error

    async def _drain_stderr(self) -> None:
        process = self._require_process()
        if process.stderr is None:
            return
        while line := await process.stderr.readline():
            self._stderr_lines.append(line.decode(errors="replace").rstrip())

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise RuntimeError("Codex app server is not running")
        return self._process

    @staticmethod
    def _format_error(error: Any) -> str:
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(error or "unknown error")
