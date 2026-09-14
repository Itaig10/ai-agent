import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_agent.skills import SkillRegistry, build_skill_prompt, load_skill


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "A useful skill",
    triggers: tuple[str, ...] = (),
    auto_activate: bool = False,
    instructions: str = "Follow these focused instructions.",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    trigger_lines = "\n".join(f"  - {trigger}" for trigger in triggers)
    path = directory / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "triggers:\n"
        f"{trigger_lines}\n"
        f"auto_activate: {'true' if auto_activate else 'false'}\n"
        "---\n"
        f"{instructions}\n",
        encoding="utf-8",
    )
    return path


class SkillRegistryTests(unittest.TestCase):
    def test_discovers_and_loads_valid_skill(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = write_skill(
                workspace / "skills",
                "bug-fix",
                description="Fix reproducible defects",
                triggers=("fix bug", "regression"),
                auto_activate=True,
            )

            registry = SkillRegistry(workspace)
            skill = registry.get("BUG-FIX")

            self.assertEqual(skill.path, path)
            self.assertEqual(skill.triggers, ("fix bug", "regression"))
            self.assertTrue(skill.auto_activate)
            self.assertEqual(registry.enabled_count, 1)
            self.assertEqual(registry.errors, ())

    def test_matches_longest_enabled_auto_trigger(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            write_skill(
                root,
                "general",
                triggers=("test",),
                auto_activate=True,
            )
            write_skill(
                root,
                "regression",
                triggers=("add regression test",),
                auto_activate=True,
            )
            registry = SkillRegistry(Path(directory))

            self.assertEqual(
                registry.match("Please add regression test for this bug").name,
                "regression",
            )
            registry.set_enabled("regression", False)
            self.assertEqual(
                registry.match("Please add regression test for this bug").name,
                "general",
            )
            registry.set_enabled("general", False)
            self.assertIsNone(registry.match("Please add regression test"))

    def test_reload_preserves_enabled_state(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            write_skill(workspace / "skills", "testing")
            registry = SkillRegistry(workspace)
            registry.set_enabled("testing", False)

            count = registry.reload()

            self.assertEqual(count, 1)
            self.assertFalse(registry.is_enabled("testing"))
            restored = SkillRegistry(workspace)
            self.assertFalse(restored.is_enabled("testing"))

    def test_trigger_matching_respects_word_boundaries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            write_skill(
                root,
                "testing",
                triggers=("test",),
                auto_activate=True,
            )
            registry = SkillRegistry(Path(directory))

            self.assertIsNotNone(registry.match("test the parser"))
            self.assertIsNone(registry.match("enter the contest"))

    def test_invalid_skills_are_reported_without_blocking_valid_ones(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "skills"
            write_skill(root, "valid")
            invalid = root / "invalid"
            invalid.mkdir(parents=True)
            (invalid / "SKILL.md").write_text(
                "name: invalid\nmissing frontmatter",
                encoding="utf-8",
            )

            registry = SkillRegistry(workspace)

            self.assertEqual([skill.name for skill, _ in registry.entries()], ["valid"])
            self.assertEqual(len(registry.errors), 1)
            self.assertIn("frontmatter", registry.errors[0].message)

    def test_rejects_invalid_metadata_and_oversized_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mismatch = write_skill(root, "folder")
            text = mismatch.read_text(encoding="utf-8").replace(
                "name: folder",
                "name: another",
            )
            mismatch.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "directory name"):
                load_skill(mismatch)

            unknown = write_skill(root, "unknown")
            text = unknown.read_text(encoding="utf-8").replace(
                "description:",
                "unexpected: value\ndescription:",
            )
            unknown.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown skill metadata"):
                load_skill(unknown)

            large = write_skill(root, "large", instructions="x" * 100)
            with self.assertRaisesRegex(ValueError, "safety limit"):
                load_skill(large, max_bytes=20)

    def test_auto_activation_requires_triggers(self) -> None:
        with TemporaryDirectory() as directory:
            path = write_skill(
                Path(directory),
                "automatic",
                auto_activate=True,
            )

            with self.assertRaisesRegex(ValueError, "require at least one trigger"):
                load_skill(path)

    def test_build_prompt_scopes_skill_and_preserves_task(self) -> None:
        with TemporaryDirectory() as directory:
            path = write_skill(Path(directory), "testing")
            skill = load_skill(path)

            prompt = build_skill_prompt(skill, "Add parser tests")

            self.assertIn("project skill 'testing'", prompt)
            self.assertIn(str(skill.directory), prompt)
            self.assertIn(skill.instructions, prompt)
            self.assertTrue(prompt.endswith("User task:\nAdd parser tests"))
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                build_skill_prompt(skill, "   ")


if __name__ == "__main__":
    unittest.main()
