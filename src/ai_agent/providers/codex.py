import asyncio
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
import json
from pathlib import Path
import shutil
from typing import Any

from ai_agent.messages import ChatMessage


class CodexProvider:
    """Chat provider backed by the local Codex app-server process."""

    name = "codex"

    def __init__(
        self,
        *,
        command: str = "codex",
        model: str | None = None,
        cwd: Path | None = None,
        timeout_seconds: float = 300,
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
        self.cwd = (cwd or Path.cwd()).resolve()
        self.timeout_seconds = timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._thread_id: str | None = None
        self._next_request_id = 1
        self._notifications: deque[dict[str, Any]] = deque()
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        if not messages or messages[-1].role != "user":
            raise ValueError("Codex requires a final user message")

        async with self._lock:
            async with asyncio.timeout(self.timeout_seconds):
                await self._ensure_started()
                result = await self._request(
                    "turn/start",
                    {
                        "threadId": self._thread_id,
                        "input": [
                            {"type": "text", "text": messages[-1].content}
                        ],
                    },
                )
                turn_id = result["turn"]["id"]
                return await self._read_turn(turn_id)

    async def close(self) -> None:
        process = self._process
        self._process = None
        self._thread_id = None
        if process is not None and process.returncode is None:
            process.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2)
            if process.returncode is None:
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
            self.command,
            "app-server",
            "--stdio",
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

        thread_params: dict[str, Any] = {
            "approvalPolicy": "never",
            "cwd": str(self.cwd),
            "developerInstructions": (
                "Act as a general-purpose conversational assistant. Answer the "
                "user directly. Do not inspect files, execute commands, or modify "
                "the workspace unless the user explicitly asks you to."
            ),
            "ephemeral": True,
            "sandbox": "read-only",
        }
        if self.requested_model:
            thread_params["model"] = self.requested_model

        thread_result = await self._request("thread/start", thread_params)
        self._thread_id = thread_result["thread"]["id"]

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

            if method == "item/agentMessage/delta":
                streamed_text.append(params.get("delta", ""))
            elif method == "item/completed":
                item = params.get("item", {})
                if item.get("type") == "agentMessage":
                    final_text = item.get("text", "")
            elif method == "turn/completed":
                turn = params.get("turn", {})
                if turn.get("status") == "failed":
                    raise RuntimeError(
                        f"Codex turn failed: {self._format_error(turn.get('error'))}"
                    )
                return final_text or "".join(streamed_text)
            elif "id" in message and method:
                raise RuntimeError(
                    f"Codex requested unsupported client action {method!r}"
                )

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
