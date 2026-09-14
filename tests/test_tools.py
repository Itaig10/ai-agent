import asyncio
import unittest

from ai_agent.tools import ToolPlugin, ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def test_builtin_tools_can_be_toggled_by_name(self) -> None:
        registry = ToolRegistry()

        self.assertTrue(registry.is_enabled("web_search", "openai"))
        registry.set_enabled("web_search", False)

        self.assertFalse(registry.is_enabled("web_search", "openai"))

    def test_codex_workspace_tools_toggle_as_one_runtime(self) -> None:
        registry = ToolRegistry()

        changed = registry.set_enabled("git", False)

        self.assertEqual(changed, ("command", "filesystem", "git"))
        self.assertFalse(registry.codex_workspace_tools_enabled())
        for name in changed:
            self.assertFalse(registry.is_enabled(name, "codex"))

    def test_plugins_can_be_registered(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolPlugin(
                "example",
                "Example extension",
                frozenset({"litellm"}),
                default_enabled=False,
            )
        )

        entries = {plugin.name: enabled for plugin, enabled, _ in registry.entries()}
        self.assertFalse(entries["example"])

    def test_dynamic_plugin_is_exposed_and_executed(self) -> None:
        async def greet(arguments: dict[str, object]) -> str:
            return f"Hello {arguments['name']}"

        registry = ToolRegistry()
        registry.register(
            ToolPlugin(
                "greet",
                "Greet someone",
                frozenset({"codex"}),
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                handler=greet,
            )
        )

        specs = registry.dynamic_specs("codex")

        self.assertEqual(specs[0]["name"], "greet")
        self.assertEqual(
            asyncio.run(registry.execute("greet", {"name": "Itai"})),
            "Hello Itai",
        )

    def test_disabled_dynamic_plugin_is_not_exposed(self) -> None:
        async def handler(_arguments: dict[str, object]) -> str:
            return "done"

        registry = ToolRegistry()
        registry.register(
            ToolPlugin(
                "optional",
                "Optional tool",
                frozenset({"codex"}),
                handler=handler,
            )
        )
        registry.set_enabled("optional", False)

        self.assertEqual(registry.dynamic_specs("codex"), [])

    def test_duplicate_and_unknown_plugins_are_rejected(self) -> None:
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(ToolRegistry.BUILTINS[0])
        with self.assertRaisesRegex(ValueError, "Unknown tool"):
            registry.set_enabled("missing", True)
        with self.assertRaisesRegex(ValueError, "Unknown tool"):
            asyncio.run(registry.execute("missing", {}))

    def test_tool_without_handler_cannot_execute(self) -> None:
        registry = ToolRegistry()

        with self.assertRaisesRegex(ValueError, "no client handler"):
            asyncio.run(registry.execute("filesystem", {}))


if __name__ == "__main__":
    unittest.main()
