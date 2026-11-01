import configparser
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from acetele import config

project_root = Path(__file__).resolve().parents[1]


def test_submodule_urls_inherit_the_parent_repository_transport():
    modules = configparser.ConfigParser()
    modules.read(project_root / ".gitmodules", encoding="utf-8")

    assert modules['submodule "third_party/px4_msgs"']["url"] == "../px4_msgs.git"
    assert (
        modules['submodule "third_party/realsense_ros"']["url"]
        == "../../realsenseai/realsense-ros.git"
    )
    assert (
        modules['submodule "third_party/micro_xrce_dds_client"']["url"]
        == "../../eProsima/Micro-XRCE-DDS-Client.git"
    )
    assert (
        modules['submodule "third_party/micro_xrce_dds_agent"']["url"]
        == "../../eProsima/Micro-XRCE-DDS-Agent.git"
    )


def setup_py_value(option: str) -> str:
    result = subprocess.run(
        [sys.executable, "setup.py", option],
        cwd=project_root,
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
    """Copies build sources and real EAA meshes into an isolated tree.

    Args:
        destination: New source directory used only for wheel validation.
    """
    destination.mkdir()
    for filename in ("LICENSE", "README.md", "pyproject.toml", "setup.py"):
        shutil.copy2(project_root / filename, destination / filename)

    source_package = project_root / "acetele"
    destination_package = destination / "acetele"
    copied_suffixes = {".py", ".toml", ".urdf", ".xml"}
    for source_file in source_package.rglob("*"):
        if not source_file.is_file() or source_file.suffix not in copied_suffixes:
            continue
        relative_path = source_file.relative_to(source_package)
        destination_file = destination_package / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)

    for robot_type in ("ace_leader", "ace_follower"):
        mesh = (
            destination_package
            / "model"
            / "robots"
            / robot_type
            / "description"
            / "meshes"
            / "link_1.STL"
        )
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"isolated wheel test mesh")

    for model_name in ("eaa_leader", "eaa_follower"):
        mesh_root = (
            source_package / "model/robots" / model_name / "description/meshes"
        )
        for source_mesh in mesh_root.rglob("*"):
            if not source_mesh.is_file():
                continue
            destination_mesh = (
                destination_package / source_mesh.relative_to(source_package)
            )
            destination_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mesh, destination_mesh)

    excluded_sentinels = (
        destination / "ros2" / "sentinel.py",
        destination / "third_party" / "px4_msgs" / "sentinel.py",
        destination / "tests" / "sentinel.py",
        destination / "build" / "lib" / "tests" / "stale.py",
        destination
        / "build"
        / "lib"
        / "acetele"
        / "stale.py",
    )
    for sentinel in excluded_sentinels:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("SHOULD_NOT_BE_PACKAGED = True\n")


def _assert_eaa_wheel_resources(
    installed_root: Path, wheel_files: set[str]
) -> None:
    """Validates EAA presets and every referenced mesh after wheel extraction.

    Args:
        installed_root: Directory containing the extracted wheel contents.
        wheel_files: Archive member names from the built wheel.
    """
    installed_root = installed_root.resolve()
    for model_name in ("eaa_leader", "eaa_follower"):
        preset = (
            installed_root
            / "acetele/config/presets"
            / model_name
            / "fashionstar_rs485.toml"
        )
        assert preset.relative_to(installed_root).as_posix() in wheel_files
        spec = config.load_robot_spec(preset)
        assert spec.model == model_name
        assert spec.urdf_path is not None

        model_root = installed_root / "acetele/model/robots" / model_name
        urdf_path = Path(spec.urdf_path).resolve()
        assert urdf_path == model_root / "description" / f"{model_name}.urdf"
        assert urdf_path.relative_to(installed_root).as_posix() in wheel_files
        assert urdf_path.is_file()
        mesh_names = {
            mesh.attrib["filename"]
            for mesh in ElementTree.parse(urdf_path).iter("mesh")
        }
        assert mesh_names, "Packaged EAA models must reference display meshes"
        for mesh_name in sorted(mesh_names):
            mesh_path = (urdf_path.parent / mesh_name).resolve()
            assert mesh_path.is_relative_to(model_root)
            assert mesh_path.relative_to(installed_root).as_posix() in wheel_files
            assert mesh_path.is_file(), f"Missing packaged mesh: {mesh_path}"
            assert mesh_path.stat().st_size > 0


def test_isolated_wheel_contains_runtime_files_without_excluded_sources(
    tmp_path: Path,
) -> None:
    """Checks installed resources while excluding deployment sources.

    Args:
        tmp_path: Isolated directory supplied by pytest.
    """
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
    installed_root = tmp_path / "installed"
    with zipfile.ZipFile(wheel_path) as wheel:
        files = set(wheel.namelist())
        wheel.extractall(installed_root)

    _assert_eaa_wheel_resources(installed_root, files)

    assert {
        "acetele/config/presets/ace_follower/feetech_sms_rs485.toml",
        "acetele/config/presets/ace_follower/fashionstar_rs485.toml",
        "acetele/config/presets/ace_follower/feetech_hls_ttl.toml",
        "acetele/config/presets/ace_leader/feetech_hls_ttl.toml",
        "acetele/config/catalog.py",
        "acetele/config/loader.py",
        "acetele/control/cartesian.py",
        "acetele/control/position.py",
        "acetele/core/contracts.py",
        "acetele/estimation/joint_state.py",
        "acetele/hardware/buses/actor.py",
        "acetele/hardware/buses/serial.py",
        "acetele/hardware/devices/adapter.py",
        "acetele/hardware/devices/hands/linker/adapter.py",
        "acetele/hardware/devices/servos/fashionstar/adapter.py",
        "acetele/hardware/devices/servos/feetech/adapter.py",
        "acetele/hardware/inputs/joystick.py",
        "acetele/model/urdf.py",
        "acetele/model/joint_angle.py",
        "acetele/model/robots/ace_follower/description/ace_follower.urdf",
        "acetele/model/robots/ace_follower/description/meshes/link_1.STL",
        "acetele/model/robots/ace_leader/description/ace_leader.xml",
        "acetele/runtime/preflight.py",
        "acetele/runtime/calibration.py",
        "acetele/runtime/robot.py",
        "acetele/runtime/teleop/follower.py",
        "acetele/runtime/teleop/leader.py",
        "acetele/specification/robot.py",
        "acetele/specification/backend.py",
        "acetele/tools/tui.py",
    }.issubset(files)
    assert all(not path.startswith(("build/", "tests/")) for path in files)
    assert all(not path.startswith(("ros2/", "third_party/")) for path in files)
    assert all(not path.startswith("zeromq/") for path in files)
    assert all(not path.startswith("acetele/equipment/") for path in files)
    assert all(not path.startswith("acetele/robot/") for path in files)
