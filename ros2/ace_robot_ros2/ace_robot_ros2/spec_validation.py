"""Hardware-free constraints imposed by the current ROS 2 wire interfaces."""

from __future__ import annotations

from acetele.specification import RobotSpec


def validate_ros2_robot_spec(
    spec: RobotSpec,
    *,
    expected_model: str,
    arm_capacity: int | None = None,
    minimum_arm_joints: int = 1,
) -> None:
    """Reject a spec that cannot be represented by the selected ROS 2 adapter."""

    if not isinstance(spec, RobotSpec) or spec.model != expected_model:
        raise ValueError(
            f"runtime ROS 2 adapter requires an {expected_model} RobotSpec"
        )
    end_effector_count = sum(
        arm.end_effector is not None for arm in spec.arms
    )
    if end_effector_count > 1:
        raise RuntimeError("the current ROS 2 adapter supports at most one end effector")
    if arm_capacity is not None:
        if (
            type(minimum_arm_joints) is not int
            or minimum_arm_joints < 1
            or minimum_arm_joints > arm_capacity
        ):
            raise ValueError(
                "minimum_arm_joints must be between 1 and arm_capacity"
            )
        joint_count = sum(len(arm.joints) for arm in spec.arms)
        if not minimum_arm_joints <= joint_count <= arm_capacity:
            raise RuntimeError(
                "the ROS 2 arm state supports between "
                f"{minimum_arm_joints} and {arm_capacity} joints; got {joint_count}"
            )


__all__ = ["validate_ros2_robot_spec"]
