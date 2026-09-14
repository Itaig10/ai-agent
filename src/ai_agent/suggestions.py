from collections.abc import Callable, Iterable

from textual.suggester import Suggester


SuggestionFactory = Callable[[], Iterable[str]]


class SlashCommandSuggester(Suggester):
    """Offer dynamic, case-insensitive slash-command completions."""

    def __init__(self, suggestions: SuggestionFactory) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self._suggestions = suggestions

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        normalized = value.casefold()
        candidates = sorted(
            set(self._suggestions()),
            key=lambda suggestion: (len(suggestion), suggestion),
        )
        return next(
            (
                suggestion
                for suggestion in candidates
                if suggestion.casefold().startswith(normalized)
                and suggestion.casefold() != normalized
            ),
            None,
        )
