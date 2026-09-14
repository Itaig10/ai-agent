from datetime import UTC, datetime
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ai_agent.checkpoints import CheckpointStore


class CheckpointStoreTests(unittest.IsolatedAsyncioTestCase):
    def test_blank_diff_does_not_create_checkpoint_directory(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory) / "checkpoints"
            store = CheckpointStore(checkpoint_directory)

            self.assertIsNone(store.save("  \n"))
            self.assertFalse(checkpoint_directory.exists())

    def test_checkpoints_with_same_second_keep_creation_order(self) -> None:
        with TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory))
            fixed = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)
            identifiers = [
                SimpleNamespace(hex="ffffffff00000000"),
                SimpleNamespace(hex="0000000000000000"),
            ]
            with (
                patch("ai_agent.checkpoints.datetime") as clock,
                patch("ai_agent.checkpoints.uuid4", side_effect=identifiers),
            ):
                clock.now.return_value = fixed
                first = store.save("first")
                second = store.save("second")

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            os.utime(first.path, ns=(1, 1))
            os.utime(second.path, ns=(2, 2))

            self.assertEqual(store.list()[0].name, second.name)

    async def test_save_undo_and_restore(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._git(workspace, "init")
            self._git(workspace, "config", "user.name", "Test User")
            self._git(workspace, "config", "user.email", "test@example.com")
            file = workspace / "example.txt"
            file.write_text("before\n", encoding="utf-8")
            self._git(workspace, "add", "example.txt")
            self._git(workspace, "commit", "-m", "initial")

            file.write_text("after\n", encoding="utf-8")
            diff = self._git(workspace, "diff").stdout
            store = CheckpointStore(
                workspace / ".ai-agent" / "checkpoints",
                workspace=workspace,
            )
            saved = store.save(diff)

            self.assertIsNotNone(saved)
            undone = await store.undo_latest()
            self.assertEqual(file.read_text(encoding="utf-8"), "before\n")
            self.assertTrue(undone.undone)

            restored = await store.restore(undone.name)
            self.assertEqual(file.read_text(encoding="utf-8"), "after\n")
            self.assertFalse(restored.undone)

    async def test_undo_refuses_a_conflicting_worktree(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._git(workspace, "init")
            self._git(workspace, "config", "user.name", "Test User")
            self._git(workspace, "config", "user.email", "test@example.com")
            file = workspace / "example.txt"
            file.write_text("one\n", encoding="utf-8")
            self._git(workspace, "add", "example.txt")
            self._git(workspace, "commit", "-m", "initial")
            file.write_text("two\n", encoding="utf-8")
            diff = self._git(workspace, "diff").stdout
            store = CheckpointStore(workspace / "checkpoints", workspace=workspace)
            store.save(diff)
            file.write_text("conflict\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "safely"):
                await store.undo_latest()
            self.assertEqual(file.read_text(encoding="utf-8"), "conflict\n")

    async def test_undo_and_restore_validate_missing_checkpoints(self) -> None:
        with TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory))

            with self.assertRaisesRegex(ValueError, "no active checkpoints"):
                await store.undo_latest()
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint name"):
                await store.restore("../outside")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                await store.restore("missing")

    @staticmethod
    def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
