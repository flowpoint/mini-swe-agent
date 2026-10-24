"""Tests for minisweagent.__init__."""

import os
import subprocess
import sys


def test_package_import_is_quiet_with_non_utf8_stdout(tmp_path):
    """Imports leave stdout available for command output, regardless of locale."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "MSWEA_SILENT_STARTUP": "",
        "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path),
    }
    result = subprocess.run([sys.executable, "-c", "import minisweagent"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
