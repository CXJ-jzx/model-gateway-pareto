"""Verify documented module entry points after directory cleanup."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ProjectLayoutTests(unittest.TestCase):
    def help_command(self, *arguments):
        result = subprocess.run(
            [sys.executable, "-B", *arguments, "--help"], cwd=ROOT,
            env=dict(os.environ, PYTHONUTF8="1"), capture_output=True,
            text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)

    def test_current_module_entry_points(self):
        for module in ("endpoint_routing_strategy.data", "endpoint_routing_strategy.experiments.selection",
                       "endpoint_routing_strategy.experiments.selection.failover_demo",
                       "endpoint_routing_strategy.experiments.busy_check"):
            with self.subTest(module=module):
                self.help_command("-m", module)

    def test_legacy_module_entry_points(self):
        if not (ROOT / "endpoint_routing_strategy/experiments/legacy").is_dir():
            self.skipTest("Historical tools are intentionally excluded from the GitHub release")
        for module in ("endpoint_routing_strategy.experiments.legacy.router_strategy",
                       "endpoint_routing_strategy.experiments.legacy.replay_requests"):
            with self.subTest(module=module):
                self.help_command("-m", module)

    def test_legacy_cli_direct_script(self):
        if not (ROOT / "endpoint_routing_strategy/experiments/legacy/router_strategy.py").is_file():
            self.skipTest("Historical tools are intentionally excluded from the GitHub release")
        self.help_command(str(ROOT / "endpoint_routing_strategy/experiments/legacy/router_strategy.py"))

    def test_documents_have_clear_locations(self):
        for relative in ("README.md", "endpoint_routing_strategy/README.md",
                         "endpoint_routing_strategy/docs/README.md",
                         "endpoint_routing_strategy/docs/状态维护总结.md",
                         "endpoint_routing_strategy/docs/分步骤实验方案.md"):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())


if __name__ == "__main__":
    unittest.main()
