from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, Input, RichLog, Static

from ai_agent.session import ChatSession


class AgentApp(App[None]):
    """Terminal user interface for an AI chat session."""

    TITLE = "AI Agent"
    SUB_TITLE = "Provider-agnostic terminal assistant"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
    ]
    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #chat {
        height: 1fr;
        margin: 0 1;
        padding: 1 2;
        border: round $primary;
    }

    #prompt-container {
        height: auto;
        margin: 0 1 1 1;
    }

    #prompt {
        width: 1fr;
    }
    """

    def __init__(self, session: ChatSession) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Provider: {self.session.provider.name} | "
            f"Model: {self.session.provider.model}",
            id="status",
        )
        yield RichLog(id="chat", wrap=True, markup=True)
        with Container(id="prompt-container"):
            yield Input(placeholder="Ask anything…", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat", RichLog).write(
            "[bold cyan]AI Agent[/bold cyan]\nEnter a message to begin."
        )
        self.query_one("#prompt", Input).focus()

    @on(Input.Submitted, "#prompt")
    def submit_prompt(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        event.input.clear()
        event.input.disabled = True
        chat = self.query_one("#chat", RichLog)
        chat.write(f"\n[bold green]You[/bold green]\n{prompt}")
        chat.write("\n[dim]Thinking…[/dim]")
        self.get_reply(prompt)

    @work(exclusive=True)
    async def get_reply(self, prompt: str) -> None:
        chat = self.query_one("#chat", RichLog)
        prompt_input = self.query_one("#prompt", Input)

        try:
            reply = await self.session.send(prompt)
            chat.write(f"\n[bold cyan]Assistant[/bold cyan]\n{reply}")
        except Exception as error:
            chat.write(f"\n[bold red]Error[/bold red]\n{error}")
        finally:
            prompt_input.disabled = False
            prompt_input.focus()

    def action_clear_chat(self) -> None:
        self.session.clear()
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        chat.write("[bold cyan]Conversation cleared.[/bold cyan]")
