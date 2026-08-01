"""Discovery of RobotSpec presets shipped as package data."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class RobotSpecResource:
    """Stable resource identifier and concrete path for one packaged preset."""

    resource_id: str
    path: Path


def packaged_robot_specs() -> tuple[RobotSpecResource, ...]:
    """Return packaged presets in deterministic model and filename order."""

    root = files("acetele.config").joinpath("presets")
    resources: list[RobotSpecResource] = []
    for model_directory in ("ace_leader", "ace_follower"):
        directory = root.joinpath(model_directory)
        for resource in sorted(directory.iterdir(), key=lambda entry: entry.name):
            if resource.is_file() and resource.name.endswith(".toml"):
                resources.append(
                    RobotSpecResource(
                        f"{model_directory}/{resource.name}",
                        Path(str(resource)).resolve(),
                    )
                )
    if not resources:
        raise RuntimeError("no packaged ACETele RobotSpec files were found")
    return tuple(resources)


def packaged_robot_spec(model: str, filename: str) -> Path:
    """Resolve one exact packaged preset without duplicating resource traversal."""

    resource_id = f"{model}/{filename}"
    for resource in packaged_robot_specs():
        if resource.resource_id == resource_id:
            return resource.path
    raise FileNotFoundError(f"packaged RobotSpec does not exist: {resource_id}")


__all__ = ["RobotSpecResource", "packaged_robot_spec", "packaged_robot_specs"]
