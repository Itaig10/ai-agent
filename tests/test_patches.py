import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent.patches import PatchManager


class PatchManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_apply_and_discard(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._initialize(workspace)
            target = workspace / "example.py"
            target.write_text("value = 1\n", encoding="utf-8")
            self._git(workspace, "add", "example.py")
            self._git(workspace, "commit", "-m", "initial")
            target.write_text("value = 2\nextra = True\n", encoding="utf-8")
            diff = self._git(workspace, "diff").stdout
            target.write_text("value = 1\n", encoding="utf-8")
            patch_file = workspace / "change.patch"
            patch_file.write_text(diff, encoding="utf-8")
            manager = PatchManager(workspace)

            preview = await manager.preview_file(Path("change.patch"))

            self.assertEqual(preview.paths, ("example.py",))
            self.assertEqual((preview.additions, preview.deletions), (2, 1))
            self.assertIn("+2/-1", preview.summary)
            applied = await manager.apply_pending()
            self.assertEqual(applied, preview)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "value = 2\nextra = True\n",
            )
            self.assertIsNone(manager.pending)

            manager.pending = preview
            discarded = manager.discard()
            self.assertEqual(discarded, preview)
            self.assertIsNone(manager.pending)

    async def test_python_syntax_failure_is_rolled_back(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._initialize(workspace)
            target = workspace / "example.py"
            target.write_text("value = 1\n", encoding="utf-8")
            self._git(workspace, "add", "example.py")
            self._git(workspace, "commit", "-m", "initial")
            target.write_text("if True print('bad')\n", encoding="utf-8")
            diff = self._git(workspace, "diff").stdout
            target.write_text("value = 1\n", encoding="utf-8")
            manager = PatchManager(workspace)
            manager.pending = await manager.preview(diff)

            with self.assertRaisesRegex(ValueError, "syntax validation"):
                await manager.apply_pending()

            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")
            self.assertIsNotNone(manager.pending)

    async def test_json_syntax_failure_is_rolled_back(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._initialize(workspace)
            target = workspace / "config.json"
            target.write_text('{"valid": true}\n', encoding="utf-8")
            self._git(workspace, "add", "config.json")
            self._git(workspace, "commit", "-m", "initial")
            target.write_text('{"invalid": }\n', encoding="utf-8")
            diff = self._git(workspace, "diff").stdout
            target.write_text('{"valid": true}\n', encoding="utf-8")
            manager = PatchManager(workspace)
            manager.pending = await manager.preview(diff)

            with self.assertRaisesRegex(ValueError, "syntax validation"):
                await manager.apply_pending()

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                '{"valid": true}\n',
            )

    async def test_conflict_is_rechecked_before_apply(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            self._initialize(workspace)
            target = workspace / "note.txt"
            target.write_text("before\n", encoding="utf-8")
            self._git(workspace, "add", "note.txt")
            self._git(workspace, "commit", "-m", "initial")
            target.write_text("after\n", encoding="utf-8")
            diff = self._git(workspace, "diff").stdout
            target.write_text("before\n", encoding="utf-8")
            manager = PatchManager(workspace)
            manager.pending = await manager.preview(diff)
            target.write_text("conflicting edit\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cannot be applied safely"):
                await manager.apply_pending()

            self.assertEqual(target.read_text(encoding="utf-8"), "conflicting edit\n")

    async def test_rejects_unsafe_binary_and_oversized_patches(self) -> None:
        with TemporaryDirectory() as directory:
            manager = PatchManager(Path(directory), max_patch_bytes=100)
            unsafe = "--- a/../outside\n+++ b/../outside\n@@ -0,0 +1 @@\n+bad\n"
            with self.assertRaisesRegex(ValueError, "Unsafe patch path"):
                await manager.preview(unsafe)
            with self.assertRaisesRegex(ValueError, "Binary patches"):
                await manager.preview("GIT binary patch\n")
            with self.assertRaisesRegex(ValueError, "safety limit"):
                await manager.preview("x" * 101)

    async def test_patch_file_must_be_inside_workspace(self) -> None:
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            manager = PatchManager(Path(directory))
            path = Path(outside) / "change.patch"
            path.write_text("not a patch", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inside the workspace"):
                await manager.preview_file(path)
            with self.assertRaisesRegex(ValueError, "no previewed patch"):
                manager.discard()

    async def test_rejects_symlink_and_submodule_modes(self) -> None:
        with TemporaryDirectory() as directory:
            manager = PatchManager(Path(directory))
            for mode in ("120000", "160000"):
                with self.subTest(mode=mode):
                    diff = (
                        "diff --git a/link b/link\n"
                        f"new file mode {mode}\n"
                        "--- /dev/null\n"
                        "+++ b/link\n"
                        "@@ -0,0 +1 @@\n"
                        "+target\n"
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "Symlink and submodule",
                    ):
                        await manager.preview(diff)

    @classmethod
    def _initialize(cls, workspace: Path) -> None:
        cls._git(workspace, "init")
        cls._git(workspace, "config", "user.name", "Test User")
        cls._git(workspace, "config", "user.email", "test@example.com")

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
