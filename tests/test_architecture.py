from __future__ import annotations

import ast
from pathlib import Path

project_root = Path(__file__).resolve().parents[1] / "acetele"


def _acetele_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names if alias.name.startswith("acetele"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("acetele"):
                modules.append(node.module)
    return tuple(modules)


def test_core_has_no_inward_dependencies():
    forbidden = (
        "acetele.config",
        "acetele.control",
        "acetele.deploy",
        "acetele.equipment",
        "acetele.hardware",
        "acetele.model",
        "acetele.robot",
        "acetele.runtime",
    )

    violations = []
    for path in (project_root / "core").glob("*.py"):
        for module in _acetele_imports(path):
            if module.startswith(forbidden):
                violations.append(f"{path.relative_to(project_root)} -> {module}")

    assert not violations, "core dependency violations: " + ", ".join(violations)


def test_new_hardware_layer_does_not_depend_on_legacy_equipment_or_ros():
    forbidden = ("acetele.equipment", "acetele.robot", "acetele.deploy")

    violations = []
    for path in (project_root / "hardware").rglob("*.py"):
        for module in _acetele_imports(path):
            if module.startswith(forbidden):
                violations.append(f"{path.relative_to(project_root)} -> {module}")

    assert not violations, "hardware dependency violations: " + ", ".join(violations)


def test_runtime_sessions_do_not_depend_on_ros():
    violations = []
    for path in (project_root / "runtime").glob("*_session.py"):
        for module in _acetele_imports(path):
            if module.startswith("acetele.deploy"):
                violations.append(f"{path.relative_to(project_root)} -> {module}")
        source = path.read_text(encoding="utf-8")
        if "import rclpy" in source or "from rclpy" in source:
            violations.append(f"{path.relative_to(project_root)} -> rclpy")

    assert not violations, "runtime session dependency violations: " + ", ".join(violations)


def test_composed_runtime_ros_nodes_do_not_import_legacy_devices():
    ros_package = project_root / "deploy" / "ace_robot_ros2" / "ace_robot_ros2"
    violations = []
    for path in ros_package.glob("runtime_*_node.py"):
        for module in _acetele_imports(path):
            if module.startswith(("acetele.equipment", "acetele.robot")):
                violations.append(f"{path.relative_to(project_root)} -> {module}")

    assert not violations, "runtime ROS dependency violations: " + ", ".join(violations)


def test_legacy_architecture_has_been_removed():
    legacy_paths = (
        project_root / "config" / "config_loader.py",
        project_root / "config" / "robot_config.py",
        project_root / "runtime" / "legacy_factory.py",
    )
    legacy_sources = tuple((project_root / "equipment").rglob("*.py")) + tuple(
        (project_root / "robot").rglob("*.py")
    )

    assert not legacy_sources
    assert not [str(path.relative_to(project_root)) for path in legacy_paths if path.exists()]
