"""Exercise module invocation and exit codes as a user would."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        return subprocess.run(
            [sys.executable, "-m", "anf_deformer", *arguments],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def test_valid_metadata_succeeds_without_modifying_input(self) -> None:
        payload = json.dumps({
            "schema_version": 1,
            "controls": [{"name": "jaw", "minimum": 0, "maximum": 1, "neutral": 0}],
            "vertex_count": 3,
            "coordinate_space": "head_local",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rig with spaces.json"
            path.write_text(payload, encoding="utf-8")
            result = self.run_cli("validate-rig", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 controls, 3 vertices", result.stdout)
            self.assertEqual(path.read_text(encoding="utf-8"), payload)

    def test_invalid_metadata_fails_with_readable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"schema_version": 1}', encoding="utf-8")
            result = self.run_cli("validate-rig", str(path))
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("validate-rig", str(Path(directory) / "missing.json"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_help_and_usage_exit_codes(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("validate-rig", result.stdout)
        self.assertEqual(self.run_cli("validate-rig").returncode, 2)


if __name__ == "__main__":
    unittest.main()
