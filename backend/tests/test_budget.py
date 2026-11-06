import os
import unittest

os.environ.setdefault("MOCK_MODE", "1")

from app.core.tokens import (
    BudgetExceeded,
    Usage,
    UsageTracker,
    budget_remaining,
    budget_spent,
    charge,
    reset_budget,
    set_budget,
)


class BudgetTest(unittest.TestCase):
    def tearDown(self) -> None:
        reset_budget()

    def test_charge_accumulates_and_blocks(self) -> None:
        set_budget(0.01)
        charge(0.004)
        charge(0.005)
        self.assertAlmostEqual(budget_spent(), 0.009, places=6)
        self.assertAlmostEqual(budget_remaining(), 0.001, places=6)

        with self.assertRaises(BudgetExceeded):
            charge(0.002)

    def test_tracker_feeds_budget(self) -> None:
        set_budget(0.001)
        tracker = UsageTracker()
        tracker.add(Usage(model="qwen-plus", prompt_tokens=1000, completion_tokens=100))
        self.assertGreater(budget_spent(), 0)

        with self.assertRaises(BudgetExceeded):
            tracker.add(Usage(model="qwen-plus", prompt_tokens=5000, completion_tokens=1000))

    def test_no_limit_never_raises(self) -> None:
        set_budget(None)
        charge(100.0)
        self.assertIsNone(budget_remaining())


if __name__ == "__main__":
    unittest.main()
