import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup_py_value(option: str) -> str:
    result = subprocess.run(
        [sys.executable, "setup.py", option],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_legacy_setup_py_exposes_project_metadata():
    assert setup_py_value("--name") == "acetele"
    assert setup_py_value("--version") == "0.1.0"
