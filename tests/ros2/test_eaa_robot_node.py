"""Tests EAA node selection without creating ROS or hardware resources."""

import collections.abc
import dataclasses
from unittest import mock

import pytest

from acetele import config
from eaa_robot_ros2 import ace_robot_node
from eaa_robot_ros2 import runtime_follower_node
from eaa_robot_ros2 import runtime_leader_node


@pytest.fixture(name="node_constructors", autouse=True)
def node_constructors_fixture() -> collections.abc.Iterator[
    dict[str, mock.MagicMock]
]:
    """Replaces node constructors and blocks live ROS or serial startup.

    Yields:
        Mock constructors indexed by their supported EAA model names.
    """
    with (
        mock.patch.object(
            runtime_leader_node, "RuntimeLeaderNode"
        ) as leader_constructor,
        mock.patch.object(
            runtime_follower_node, "RuntimeFollowerNode"
        ) as follower_constructor,
        mock.patch(
            "serial.Serial",
            side_effect=AssertionError("Dispatch tests must not open serial"),
        ) as serial_constructor,
        mock.patch.object(
            ace_robot_node.rclpy,
            "init",
            side_effect=AssertionError("Dispatch tests must not start ROS"),
        ) as initialize_ros,
    ):
        yield {
            "eaa_leader": leader_constructor,
            "eaa_follower": follower_constructor,
        }
        serial_constructor.assert_not_called()
        initialize_ros.assert_not_called()


@pytest.mark.parametrize("model_name", ("eaa_leader", "eaa_follower"))
def test_factory_selects_configured_node(
    model_name: str,
    node_constructors: dict[str, mock.MagicMock],
) -> None:
    """Checks that each physical preset selects its corresponding node.

    Args:
        model_name: EAA model whose preset is passed to the factory.
        node_constructors: Mock leader and follower constructors.
    """
    preset = config.packaged_robot_spec(model_name, "fashionstar_rs485.toml")
    expected_spec = config.load_robot_spec(preset)

    result = ace_robot_node._make_robot_node(str(preset))

    selected_constructor = node_constructors[model_name]
    selected_constructor.assert_called_once_with(expected_spec)
    assert result is selected_constructor.return_value

    for candidate_name, constructor in node_constructors.items():
        if candidate_name != model_name:
            constructor.assert_not_called()


def test_factory_rejects_empty_path() -> None:
    """Checks that an empty path is rejected before loading configuration."""
    with mock.patch.object(
        ace_robot_node, "load_robot_spec"
    ) as load_spec:
        with pytest.raises(ValueError, match="config_path"):
            ace_robot_node._make_robot_node("")

    load_spec.assert_not_called()


def test_factory_rejects_unsupported_model(
    node_constructors: dict[str, mock.MagicMock],
) -> None:
    """Checks that an unsupported model cannot create either robot node.

    Args:
        node_constructors: Mock leader and follower constructors.
    """
    preset = config.packaged_robot_spec(
        "eaa_leader", "fashionstar_rs485.toml"
    )
    unsupported_spec = dataclasses.replace(
        config.load_robot_spec(preset),
        model="unsupported_robot",
    )

    with mock.patch.object(
        ace_robot_node,
        "load_robot_spec",
        return_value=unsupported_spec,
    ):
        with pytest.raises(ValueError, match="does not support model"):
            ace_robot_node._make_robot_node(str(preset))

    for constructor in node_constructors.values():
        constructor.assert_not_called()
