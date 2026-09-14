from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class ToolPlugin:
    """Metadata for one optional agent capability."""

    name: str
    description: str
    providers: frozenset[str]
    default_enabled: bool = True
    input_schema: dict[str, Any] | None = None
    handler: ToolHandler | None = None
    requires_approval: bool = True


class ToolRegistry:
    """Registry controlling provider tool availability."""

    # Codex exposes filesystem access, shell commands, and Git through the same
    # native execution runtime. They must therefore be enabled or disabled as a
    # unit; pretending they can be isolated would leave other entry points open.
    CODEX_WORKSPACE_TOOLS = frozenset({"filesystem", "command", "git"})

    BUILTINS = (
        ToolPlugin("filesystem", "Read and edit workspace files", frozenset({"codex"})),
        ToolPlugin("command", "Run local project commands", frozenset({"codex"})),
        ToolPlugin("git", "Inspect the Git repository", frozenset({"codex"})),
        ToolPlugin(
            "web_search",
            "Search the live web",
            frozenset({"codex", "openai"}),
        ),
    )

    def __init__(self, plugins: tuple[ToolPlugin, ...] | None = None) -> None:
        self._plugins = {plugin.name: plugin for plugin in plugins or self.BUILTINS}
        self._enabled = {
            plugin.name: plugin.default_enabled for plugin in self._plugins.values()
        }

    def register(self, plugin: ToolPlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Tool {plugin.name!r} is already registered")
        self._plugins[plugin.name] = plugin
        self._enabled[plugin.name] = plugin.default_enabled

    def set_enabled(self, name: str, enabled: bool) -> tuple[str, ...]:
        if name not in self._plugins:
            raise ValueError(f"Unknown tool {name!r}")
        affected = (
            self.CODEX_WORKSPACE_TOOLS
            if name in self.CODEX_WORKSPACE_TOOLS
            else frozenset({name})
        )
        changed = tuple(sorted(tool for tool in affected if tool in self._plugins))
        for tool in changed:
            self._enabled[tool] = enabled
        return changed

    def codex_workspace_tools_enabled(self) -> bool:
        """Return whether Codex's shared native execution runtime is enabled."""

        return all(
            self._enabled.get(name, False) for name in self.CODEX_WORKSPACE_TOOLS
        )

    def is_enabled(self, name: str, provider: str | None = None) -> bool:
        plugin = self._plugins.get(name)
        if plugin is None or not self._enabled.get(name, False):
            return False
        return provider is None or provider in plugin.providers

    def entries(
        self, provider: str | None = None
    ) -> list[tuple[ToolPlugin, bool, bool]]:
        return [
            (
                plugin,
                self._enabled[plugin.name],
                provider is None or provider in plugin.providers,
            )
            for plugin in self._plugins.values()
        ]

    def dynamic_specs(self, provider: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": plugin.name,
                "description": plugin.description,
                "inputSchema": plugin.input_schema or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "deferLoading": False,
            }
            for plugin in self._plugins.values()
            if plugin.handler is not None and self.is_enabled(plugin.name, provider)
        ]

    def get(self, name: str) -> ToolPlugin:
        try:
            return self._plugins[name]
        except KeyError as error:
            raise ValueError(f"Unknown tool {name!r}") from error

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        plugin = self.get(name)
        if plugin.handler is None:
            raise ValueError(f"Tool {name!r} has no client handler")
        return await plugin.handler(arguments)
