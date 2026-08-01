from __future__ import annotations

import ast
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
package_root = repository_root / "acetele"
zmq_root = repository_root / "zeromq" / "ace_robot_zmq" / "ace_robot_zmq"


def _acetele_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(
                alias.name for alias in node.names if alias.name.startswith("acetele")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("acetele"):
                modules.append(node.module)
    return tuple(modules)


def _violations(root: Path, forbidden: tuple[str, ...]) -> list[str]:
    violations = []
    for path in root.rglob("*.py"):
        for module in _acetele_imports(path):
            if module.startswith(forbidden):
                violations.append(f"{path.relative_to(package_root)} -> {module}")
    return violations


def test_core_and_specification_have_only_inward_dependencies():
    core_forbidden = tuple(
        f"acetele.{name}"
        for name in (
            "config",
            "control",
            "estimation",
            "hardware",
            "model",
            "runtime",
            "specification",
            "tools",
        )
    )
    specification_forbidden = tuple(
        f"acetele.{name}"
        for name in (
            "config",
            "control",
            "estimation",
            "hardware",
            "model",
            "runtime",
            "tools",
        )
    )

    violations = _violations(package_root / "core", core_forbidden)
    violations += _violations(
        package_root / "specification",
        specification_forbidden,
    )

    assert not violations, "inward dependency violations: " + ", ".join(violations)


def test_algorithms_and_hardware_do_not_depend_on_runtime_or_ros():
    forbidden = ("acetele.runtime", "acetele.tools", "acetele.deploy")
    violations = []
    for directory in ("control", "estimation", "hardware", "model"):
        violations += _violations(package_root / directory, forbidden)
    for directory in ("control", "estimation", "hardware", "model"):
        for path in (package_root / directory).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "import rclpy" in source or "from rclpy" in source:
                violations.append(f"{path.relative_to(package_root)} -> rclpy")

    assert not violations, "lower-layer dependency violations: " + ", ".join(violations)


def test_runtime_depends_on_adapter_ports_not_concrete_vendors():
    forbidden = (
        "acetele.hardware.devices.servos",
        "acetele.hardware.devices.hands",
        "acetele.hardware.simulators",
    )
    violations = _violations(package_root / "runtime", forbidden)
    for path in (package_root / "runtime").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "import rclpy" in source or "from rclpy" in source:
            violations.append(f"{path.relative_to(package_root)} -> rclpy")

    assert not violations, "runtime adapter violations: " + ", ".join(violations)


def test_ros_and_third_party_are_outside_the_core_wheel_package():
    assert (repository_root / "ros2" / "ace_robot_ros2").is_dir()
    assert (repository_root / "third_party" / "px4_msgs").is_dir()
    assert not (package_root / "deploy").exists()
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "import rclpy" not in source
        assert "from rclpy" not in source


def test_zmq_adapter_is_parallel_to_ros_and_never_enters_the_core_package():
    assert (zmq_root / "application.py").is_file()
    assert (zmq_root / "protocol.py").is_file()
    native_xrce_root = zmq_root.parent / "xrce"
    assert (native_xrce_root / "CMakeLists.txt").is_file()
    assert (native_xrce_root / "src" / "publisher.c").is_file()
    assert not (native_xrce_root / "ace_px4_xrce").exists()
    assert not (repository_root / "xrce").exists()
    for path in package_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import ace_robot_zmq" not in source
        assert "from ace_robot_zmq" not in source


def test_domain_contracts_do_not_own_backend_or_angle_algorithms():
    core_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (package_root / "core").rglob("*.py")
    )
    assert "class Backend" not in core_sources
    assert "def wrap_to_pi" not in core_sources
    assert "def unwrap_near" not in core_sources
    assert (package_root / "specification" / "backend.py").is_file()
    assert (package_root / "model" / "joint_angle.py").is_file()


def test_joint_angle_transformations_have_one_implementation():
    definitions = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
                "wrap_to_pi",
                "unwrap_near",
                "_wrap_to_pi",
                "_unwrap_near",
            ):
                definitions.append(f"{path.relative_to(package_root)}:{node.name}")
    assert definitions == [
        "model/joint_angle.py:wrap_to_pi",
        "model/joint_angle.py:unwrap_near",
    ]


def test_removed_architecture_paths_do_not_reappear():
    removed = (
        "utils",
        "deploy",
        "equipment",
        "robot",
        "hardware/serial",
        "hardware/smart_servos",
        "hardware/dexterous_hands",
        "hardware/joystick",
        "hardware/mock",
        "runtime/robot_runtime.py",
        "runtime/leader_session.py",
        "runtime/follower_session.py",
        "config/spec_loader.py",
        "config/specs.py",
        "control/position_pipeline.py",
        "control/cartesian_teleop.py",
        "hardware/state_estimator.py",
    )

    existing = tuple(path for path in removed if (package_root / path).exists())

    assert not existing, "removed architecture paths still exist: " + ", ".join(existing)
