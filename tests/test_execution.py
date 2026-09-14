import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent.execution import (
    CommandPolicyStore,
    ExecutionLimitStore,
    ExecutionLimits,
    native_command_is_simple,
    parse_size,
)


class CommandPolicyStoreTests(unittest.TestCase):
    def test_longest_prefix_wins_and_equal_deny_is_safest(self) -> None:
        with TemporaryDirectory() as directory:
            store = CommandPolicyStore(Path(directory) / "policy.json")
            broad = store.add("allow", ["git"])
            specific = store.add("ask", ["git", "push"])

            self.assertEqual(store.evaluate(["git", "status"]).rule, broad)
            self.assertEqual(store.evaluate(["git", "push", "origin"]).rule, specific)
            deny = store.add("deny", ["git", "push"])
            self.assertEqual(store.evaluate(["git", "push"]).rule, deny)
            self.assertIsNone(store.evaluate(["git-lfs"]).rule)

    def test_rules_defaults_and_removal_persist(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            store = CommandPolicyStore(path)
            store.set_default("deny")
            first = store.add("allow", ["python", "-m", "unittest"])
            store.add("ask", ["git", "push"])
            store.remove(first.id)

            restored = CommandPolicyStore(path)

            self.assertEqual(restored.default_action, "deny")
            self.assertEqual(len(restored.list()), 1)
            self.assertEqual(restored.list()[0].command, ("git", "push"))
            self.assertEqual(restored.evaluate(["unknown"]).action, "deny")

    def test_invalid_actions_commands_and_ids_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = CommandPolicyStore(Path(directory) / "policy.json")
            with self.assertRaisesRegex(ValueError, "allow, ask, or deny"):
                store.add("sometimes", ["test"])
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                store.add("allow", [])
            with self.assertRaisesRegex(ValueError, "does not exist"):
                store.remove(99)

    def test_malformed_persisted_rules_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(
                json.dumps(
                    {
                        "default": "allow",
                        "rules": [{"id": 1, "action": "allow", "command": "rm"}],
                    }
                ),
                encoding="utf-8",
            )

            store = CommandPolicyStore(path)
            self.assertEqual(store.default_action, "ask")
            self.assertIsNotNone(store.load_error)

            path.write_text("[]", encoding="utf-8")
            store = CommandPolicyStore(path)
            self.assertEqual(store.list(), ())
            self.assertIsNotNone(store.load_error)

    def test_native_shell_complexity_requires_manual_approval(self) -> None:
        self.assertTrue(native_command_is_simple("git status"))
        self.assertTrue(native_command_is_simple("python -c 'print(1)'"))
        for command in (
            "git status && echo unexpected",
            "git status | tee status.txt",
            "echo $(touch unexpected)",
            "echo `touch unexpected`",
            "git status\nrm file",
        ):
            with self.subTest(command=command):
                self.assertFalse(native_command_is_simple(command))


class ExecutionLimitStoreTests(unittest.TestCase):
    def test_limits_parse_persist_report_and_reset(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "limits.json"
            store = ExecutionLimitStore(path)
            store.set("timeout", "2m")
            store.set("output", "32kb")
            store.set("cpu", "1.2s")
            store.set("memory", "512mb")
            store.set("file-size", "4mb")
            store.set("processes", "12")

            restored = ExecutionLimitStore(path)

            self.assertEqual(restored.limits.wall_time_seconds, 120)
            self.assertEqual(restored.limits.output_bytes, 32 * 1024)
            self.assertEqual(restored.limits.cpu_seconds, 2)
            self.assertEqual(restored.limits.memory_mb, 512)
            self.assertEqual(restored.limits.file_size_mb, 4)
            self.assertEqual(restored.limits.processes, 12)
            self.assertIn("memory: 512MB", restored.report())
            restored.reset()
            self.assertIsNone(restored.limits.memory_mb)
            self.assertEqual(restored.limits.output_bytes, 100_000)

    def test_optional_limits_can_be_disabled_and_required_ones_cannot(self) -> None:
        with TemporaryDirectory() as directory:
            store = ExecutionLimitStore(Path(directory) / "limits.json")
            store.set("memory", "64mb")
            store.set("memory", "off")
            self.assertIsNone(store.limits.memory_mb)
            with self.assertRaisesRegex(ValueError, "cannot be disabled"):
                store.set("timeout", "off")
            with self.assertRaisesRegex(ValueError, "Unknown limit"):
                store.set("network", "1")

    def test_invalid_limit_values_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            store = ExecutionLimitStore(Path(directory) / "limits.json")
            for name, value in (
                ("output", "10b"),
                ("processes", "0"),
                ("processes", "many"),
                ("memory", "0mb"),
            ):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        store.set(name, value)

    def test_malformed_limit_file_uses_safe_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "limits.json"
            path.write_text(
                json.dumps({"timeout": "2m", "output": "unbounded"}),
                encoding="utf-8",
            )

            store = ExecutionLimitStore(path)

            self.assertEqual(store.limits.wall_time_seconds, 300)
            self.assertEqual(store.limits.output_bytes, 100_000)
            self.assertIsNotNone(store.load_error)

    @unittest.skipUnless(os.name == "posix", "POSIX resource limits required")
    def test_subprocess_options_include_configured_resource_hook(self) -> None:
        limits = ExecutionLimits(cpu_seconds=2, memory_mb=256, processes=8)

        options = limits.subprocess_options()

        self.assertTrue(options["start_new_session"])
        self.assertTrue(callable(options["preexec_fn"]))
        self.assertEqual(limits.effective_timeout(900), 300)

    def test_parse_size_supports_units(self) -> None:
        self.assertEqual(parse_size("1kb"), 1024)
        self.assertEqual(parse_size("1.5MB"), 1_572_864)
        with self.assertRaisesRegex(ValueError, "must use"):
            parse_size("100")


if __name__ == "__main__":
    unittest.main()
