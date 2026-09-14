import unittest

from ai_agent.metrics import UsageMetrics


class UsageMetricsTests(unittest.TestCase):
    def test_add_accumulates_tokens_latency_requests_and_cost(self) -> None:
        totals = UsageMetrics()

        totals.add(
            UsageMetrics(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                cost_usd=0.02,
                latency_ms=100,
                requests=1,
            )
        )
        totals.add(
            UsageMetrics(
                input_tokens=5,
                output_tokens=2,
                total_tokens=7,
                cost_usd=0.03,
                latency_ms=200,
                requests=1,
            )
        )

        self.assertEqual(totals.input_tokens, 15)
        self.assertEqual(totals.output_tokens, 6)
        self.assertEqual(totals.total_tokens, 21)
        self.assertAlmostEqual(totals.cost_usd or 0, 0.05)
        self.assertEqual(totals.average_latency_ms, 150)

    def test_unknown_cost_stays_unknown_until_reported(self) -> None:
        totals = UsageMetrics()

        totals.add(UsageMetrics(requests=1))
        self.assertIsNone(totals.cost_usd)
        totals.add(UsageMetrics(cost_usd=0.0, requests=1))

        self.assertEqual(totals.cost_usd, 0.0)

    def test_average_latency_is_zero_without_requests(self) -> None:
        self.assertEqual(UsageMetrics(latency_ms=500).average_latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
