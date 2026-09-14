import asyncio
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import tomllib


MAX_PATCH_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class PatchPreview:
    """A validated patch held in memory until the user applies it."""

    source: Path
    diff: str
    paths: tuple[str, ...]
    additions: int
    deletions: int

    @property
    def summary(self) -> str:
        files = len(self.paths)
        return (
            f"{files} file{'s' if files != 1 else ''}, "
            f"+{self.additions}/-{self.deletions}"
        )


class PatchManager:
    """Preview and safely apply textual unified diffs inside one workspace."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        max_patch_bytes: int = MAX_PATCH_BYTES,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.max_patch_bytes = max_patch_bytes
        self.pending: PatchPreview | None = None

    async def preview_file(self, source: Path) -> PatchPreview:
        if not source.is_absolute():
            source = self.workspace / source
        source = source.resolve()
        if not source.is_relative_to(self.workspace):
            raise ValueError("Patch files must be inside the workspace")
        if not source.is_file():
            raise ValueError(f"Patch file {source} does not exist")
        if source.stat().st_size > self.max_patch_bytes:
            raise ValueError(
                f"Patch exceeds the {self.max_patch_bytes:,}-byte safety limit"
            )
        try:
            diff = source.read_text(encoding="utf-8")
        except UnicodeError as error:
            raise ValueError("Patch must be UTF-8 text") from error
        preview = await self.preview(diff, source=source)
        self.pending = preview
        return preview

    async def preview(
        self,
        diff: str,
        *,
        source: Path | None = None,
    ) -> PatchPreview:
        if len(diff.encode("utf-8")) > self.max_patch_bytes:
            raise ValueError(
                f"Patch exceeds the {self.max_patch_bytes:,}-byte safety limit"
            )
        if "GIT binary patch" in diff or "Binary files " in diff:
            raise ValueError("Binary patches are not supported")
        if "mode 120000" in diff or "mode 160000" in diff:
            raise ValueError("Symlink and submodule patches are not supported")
        paths = self._paths(diff)
        self._validate_workspace_paths(paths)
        additions, deletions = self._line_counts(diff)
        await self._git_apply(diff, check=True)
        return PatchPreview(
            source=source or self.workspace / "<memory>",
            diff=diff,
            paths=paths,
            additions=additions,
            deletions=deletions,
        )

    async def apply_pending(self) -> PatchPreview:
        preview = self.pending
        if preview is None:
            raise ValueError("There is no previewed patch to apply")
        await self._git_apply(preview.diff, check=True)
        await self._git_apply(preview.diff)
        try:
            self._validate_syntax(preview.paths)
        except (OSError, SyntaxError, ValueError) as error:
            rollback = await self._git_apply(
                preview.diff,
                reverse=True,
                raise_on_error=False,
            )
            if rollback:
                raise RuntimeError(
                    f"Patch validation failed ({error}) and rollback failed: "
                    f"{rollback}"
                ) from error
            raise ValueError(
                f"Patch failed source syntax validation: {error}"
            ) from error
        self.pending = None
        return preview

    def discard(self) -> PatchPreview:
        if self.pending is None:
            raise ValueError("There is no previewed patch to discard")
        preview = self.pending
        self.pending = None
        return preview

    async def _git_apply(
        self,
        diff: str,
        *,
        check: bool = False,
        reverse: bool = False,
        raise_on_error: bool = True,
    ) -> str:
        arguments = ["git", "apply"]
        if check:
            arguments.append("--check")
        if reverse:
            arguments.append("--reverse")
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=self.workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(diff.encode("utf-8"))
        if process.returncode == 0:
            return ""
        message = (stderr or stdout).decode(errors="replace").strip()
        message = message or f"git apply exited with status {process.returncode}"
        if raise_on_error:
            raise ValueError(f"Patch cannot be applied safely: {message}")
        return message

    def _validate_syntax(self, paths: tuple[str, ...]) -> None:
        for relative in paths:
            path = self.workspace / relative
            if not path.is_file():
                continue
            if path.suffix not in {".json", ".py", ".toml"}:
                continue
            source = path.read_text(encoding="utf-8")
            if path.suffix == ".py":
                compile(source, str(path), "exec")
            elif path.suffix == ".json":
                json.loads(source)
            elif path.suffix == ".toml":
                tomllib.loads(source)

    def _validate_workspace_paths(self, paths: tuple[str, ...]) -> None:
        for relative in paths:
            target = self.workspace / relative
            if target.is_symlink():
                raise ValueError(f"Patch target {relative!r} is a symbolic link")
            parent = target.parent.resolve()
            if not parent.is_relative_to(self.workspace):
                raise ValueError(f"Patch target {relative!r} escapes the workspace")

    @classmethod
    def _paths(cls, diff: str) -> tuple[str, ...]:
        old_paths: list[str] = []
        new_paths: list[str] = []
        for line in diff.splitlines():
            if line.startswith("--- "):
                old_paths.append(cls._clean_header_path(line[4:]))
            elif line.startswith("+++ "):
                new_paths.append(cls._clean_header_path(line[4:]))
        if not old_paths or len(old_paths) != len(new_paths):
            raise ValueError("Patch must contain paired --- and +++ file headers")
        paths = []
        for old, new in zip(old_paths, new_paths, strict=True):
            candidate = new if new != "/dev/null" else old
            cls._validate_relative_path(candidate)
            paths.append(candidate)
        return tuple(dict.fromkeys(paths))

    @staticmethod
    def _clean_header_path(value: str) -> str:
        value = value.split("\t", 1)[0].strip()
        if value == "/dev/null":
            return value
        if value.startswith(('"', "'")):
            raise ValueError("Quoted patch paths are not supported")
        if value.startswith(("a/", "b/")):
            value = value[2:]
        return value

    @staticmethod
    def _validate_relative_path(value: str) -> None:
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] == ".git"
        ):
            raise ValueError(f"Unsafe patch path {value!r}")

    @staticmethod
    def _line_counts(diff: str) -> tuple[int, int]:
        additions = sum(
            line.startswith("+") and not line.startswith("+++")
            for line in diff.splitlines()
        )
        deletions = sum(
            line.startswith("-") and not line.startswith("---")
            for line in diff.splitlines()
        )
        return additions, deletions
