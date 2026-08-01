from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

package_root = Path(__file__).resolve().parents[1]


def test_zmq_wheel_contains_only_the_adapter_package(tmp_path):
    source_directory = tmp_path / "source"
    wheel_directory = tmp_path / "wheel"
    source_directory.mkdir()
    wheel_directory.mkdir()
    shutil.copy2(package_root / "pyproject.toml", source_directory / "pyproject.toml")
    shutil.copytree(
        package_root / "ace_robot_zmq",
        source_directory / "ace_robot_zmq",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheel_directory),
        ],
        cwd=source_directory,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(wheel_directory.glob("ace_robot_zmq-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        files = set(wheel.namelist())

    assert {
        "ace_robot_zmq/__init__.py",
        "ace_robot_zmq/__main__.py",
        "ace_robot_zmq/application.py",
        "ace_robot_zmq/cli.py",
        "ace_robot_zmq/options.py",
        "ace_robot_zmq/px4_xrce.py",
        "ace_robot_zmq/protocol.py",
        "ace_robot_zmq/sdk.py",
        "ace_robot_zmq/security.py",
        "ace_robot_zmq/transport.py",
        "ace_robot_zmq/ArmJointState.msg",
    }.issubset(files)
    assert all(
        not path.startswith(("acetele/", "ros2/", "third_party/", "xrce/"))
        for path in files
    )
