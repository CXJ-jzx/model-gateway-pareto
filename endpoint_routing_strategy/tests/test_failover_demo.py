import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from endpoint_routing_strategy.experiments.selection.failover_demo import run_demo
from endpoint_routing_strategy.experiments.selection.runner import DEFAULT_CONFIG


class FailoverDemoTests(unittest.TestCase):
    def test_feedback_demo_produces_verified_reports_and_two_plots(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "new_demo"
            report = run_demo(DEFAULT_CONFIG, output)
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["attempted_endpoints"], ["primary", "stable_expensive"])
            self.assertEqual(report["actual_http_requests_sent"], 0)
            self.assertAlmostEqual(report["estimated_cumulative_cost"], .00075)
            self.assertEqual(report["status"], "succeeded")
            for name in ("session.json", "summary.json", "manifest.json", "input_snapshot.json",
                         "final_snapshot.json", "summary.md", "primary_pareto.svg", "backup_pareto.svg"):
                self.assertTrue((output / name).exists(), name)
            final = json.loads((output / "final_snapshot.json").read_text(encoding="utf-8"))
            original = json.loads((output / "input_snapshot.json").read_text(encoding="utf-8"))
            for first, last in zip(original["endpoints"], final["endpoints"]):
                increase = len(last["observations"]["nonstream"]) - len(first["observations"]["nonstream"])
                self.assertEqual(increase, int(first["offering"]["endpoint_id"] in report["attempted_endpoints"]))
            with self.assertRaises(FileExistsError):
                run_demo(DEFAULT_CONFIG, output)

    def test_missing_case_is_not_silently_replaced(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No eligible scenario"):
                run_demo(DEFAULT_CONFIG, Path(directory) / "demo", case_id="unknown")

    def test_single_frontier_without_backup_cannot_demo_failover(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one feasible backup"):
                run_demo(DEFAULT_CONFIG, Path(directory) / "demo", case_id="pareto_only_backup_pool")


if __name__ == "__main__":
    unittest.main()
