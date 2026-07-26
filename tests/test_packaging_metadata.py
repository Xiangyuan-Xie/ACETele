import shutil
import subprocess
import sys
import zipfile
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
    assert setup_py_value("--version") == "0.2.0"
    assert setup_py_value("--license") == "Apache-2.0"


def _copy_isolated_wheel_source(destination: Path) -> None:
    destination.mkdir()
    for filename in ("LICENSE", "README.md", "pyproject.toml", "setup.py"):
        shutil.copy2(ROOT / filename, destination / filename)

    source_package = ROOT / "acetele"
    destination_package = destination / "acetele"
    copied_suffixes = {".py", ".toml", ".urdf", ".xml"}
    for source_file in source_package.rglob("*"):
        if not source_file.is_file() or source_file.suffix not in copied_suffixes:
            continue
        relative_path = source_file.relative_to(source_package)
        if relative_path.parts[:2] in (
            ("deploy", "px4_msgs"),
            ("deploy", "realsense-ros"),
        ):
            continue
        destination_file = destination_package / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)

    for robot_type in ("ace_leader", "ace_follower"):
        mesh = (
            destination_package
            / "robot"
            / robot_type
            / "description"
            / "meshes"
            / "link_1.STL"
        )
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"isolated wheel test mesh")

    excluded_sentinels = (
        destination_package / "deploy" / "px4_msgs" / "sentinel.py",
        destination_package
        / "deploy"
        / "realsense-ros"
        / "realsense2_camera"
        / "sentinel.py",
        destination / "tests" / "sentinel.py",
        destination / "build" / "lib" / "tests" / "stale.py",
        destination
        / "build"
        / "lib"
        / "acetele"
        / "deploy"
        / "realsense-ros"
        / "stale.py",
    )
    for sentinel in excluded_sentinels:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("SHOULD_NOT_BE_PACKAGED = True\n")


def test_isolated_wheel_contains_runtime_files_without_excluded_sources(tmp_path):
    source_root = tmp_path / "source"
    wheel_directory = tmp_path / "wheel"
    _copy_isolated_wheel_source(source_root)
    wheel_directory.mkdir()

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
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_path = next(wheel_directory.glob("acetele-*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        files = set(wheel.namelist())

    assert {
        "acetele/config/ace_follower.toml",
        "acetele/config/robot_config.py",
        "acetele/equipment/dexterous_hands/o6.py",
        "acetele/equipment/feetech/arm.py",
        "acetele/equipment/feetech/state_estimator.py",
        "acetele/equipment/joint_device.py",
        "acetele/robot/ace_follower/description/ace_follower.urdf",
        "acetele/robot/ace_follower/description/meshes/link_1.STL",
        "acetele/robot/ace_leader/description/ace_leader.xml",
        "acetele/robot/joint_robot.py",
    }.issubset(files)
    assert all(not path.startswith(("build/", "tests/")) for path in files)
    assert all("/px4_msgs/" not in path for path in files)
    assert all("/realsense-ros/" not in path for path in files)
