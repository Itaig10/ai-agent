import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A saved, reversible unified diff."""

    name: str
    path: Path
    undone: bool = False


class CheckpointStore:
    """Persist turn diffs and safely apply or reverse them."""

    _VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")

    def __init__(
        self,
        directory: Path | None = None,
        *,
        workspace: Path | None = None,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.directory = directory or self.workspace / ".ai-agent" / "checkpoints"

    def save(self, diff: str) -> Checkpoint | None:
        if not diff.strip():
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        name = f"{timestamp}-{uuid4().hex[:8]}"
        path = self.directory / f"{name}.patch"
        path.write_text(diff.rstrip() + "\n", encoding="utf-8")
        return Checkpoint(name=name, path=path)

    def list(self) -> list[Checkpoint]:
        if not self.directory.exists():
            return []
        checkpoints = [
            Checkpoint(path.stem, path, False)
            for path in self.directory.glob("*.patch")
            if not path.name.endswith(".undone.patch")
        ]
        checkpoints.extend(
            Checkpoint(path.name.removesuffix(".undone.patch"), path, True)
            for path in self.directory.glob("*.undone.patch")
        )
        return sorted(
            checkpoints,
            key=lambda checkpoint: (
                checkpoint.path.stat().st_mtime_ns,
                checkpoint.name,
            ),
            reverse=True,
        )

    async def undo_latest(self) -> Checkpoint:
        active = [checkpoint for checkpoint in self.list() if not checkpoint.undone]
        if not active:
            raise ValueError("There are no active checkpoints to undo")
        checkpoint = active[0]
        await self._git_apply(checkpoint.path, reverse=True)
        undone_path = checkpoint.path.with_name(f"{checkpoint.name}.undone.patch")
        checkpoint.path.replace(undone_path)
        return Checkpoint(checkpoint.name, undone_path, True)

    async def restore(self, name: str) -> Checkpoint:
        self._validate_name(name)
        path = self.directory / f"{name}.undone.patch"
        if not path.is_file():
            raise ValueError(f"Undone checkpoint {name!r} does not exist")
        await self._git_apply(path, reverse=False)
        restored_path = self.directory / f"{name}.patch"
        path.replace(restored_path)
        return Checkpoint(name, restored_path, False)

    async def _git_apply(self, path: Path, *, reverse: bool) -> None:
        direction = ["--reverse"] if reverse else []
        check = await self._run_git("apply", *direction, "--check", str(path))
        if check:
            action = "undo" if reverse else "restore"
            raise ValueError(f"Cannot {action} checkpoint safely: {check}")
        failure = await self._run_git("apply", *direction, str(path))
        if failure:
            raise ValueError(f"Could not apply checkpoint: {failure}")

    async def _run_git(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return ""
        message = (stderr or stdout).decode(errors="replace").strip()
        return message or f"git exited with status {process.returncode}"

    def _validate_name(self, name: str) -> None:
        if not self._VALID_NAME.fullmatch(name):
            raise ValueError("Invalid checkpoint name")
