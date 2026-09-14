from dataclasses import dataclass


@dataclass(slots=True)
class UsageMetrics:
    """Token, cost, and latency totals for model requests."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int = 0
    requests: int = 0

    def add(self, other: "UsageMetrics") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.latency_ms += other.latency_ms
        self.requests += other.requests
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    @property
    def average_latency_ms(self) -> int:
        return round(self.latency_ms / self.requests) if self.requests else 0
