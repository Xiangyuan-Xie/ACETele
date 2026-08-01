import subprocess
import sys
import zipfile
from pathlib import Path


def test_operator_ui_wheel_contains_only_shared_ui(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
        ),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("ace_operator_ui-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {
        "ace_operator_ui/__init__.py",
        "ace_operator_ui/model.py",
        "ace_operator_ui/window.py",
    }.issubset(names)
    assert all(not name.startswith(("acetele/", "ace_robot_zmq/", "ros2/")) for name in names)


def test_metadata_refresh_updates_both_camera_views():
    root = Path(__file__).resolve().parents[1]
    source = (root / "ace_operator_ui" / "window.py").read_text(encoding="utf-8")
    method = source.split("def update_metadata", 1)[1].split(
        "def update_metrics", 1
    )[0]

    assert "self.front_metadata_view" in method
    assert "self.wrist_metadata_view" in method
    recording = source.split("def _set_recording", 1)[1].split(
        "def _update_camera_status", 1
    )[0]
    assert "wrist_json" not in recording
