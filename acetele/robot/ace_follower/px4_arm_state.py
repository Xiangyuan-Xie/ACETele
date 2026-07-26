from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from acetele.equipment.joint_device import JointDeviceState

PX4_ARM_JOINT_CAPACITY = 14
PX4_POLICY_ARM_JOINT_COUNT = 4


@dataclass(frozen=True)
class EncodedPX4ArmJointState:
    joint_count: int
    positions: tuple[float, ...]
    velocities: tuple[float, ...]


def encode_px4_arm_joint_states(
    arm_states: Sequence[JointDeviceState],
) -> EncodedPX4ArmJointState:
    """Flatten arm-only states in assembly order and pad the PX4 wire arrays."""
    states = tuple(arm_states)
    if not states:
        raise ValueError("at least one arm state is required")

    position_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    for index, state in enumerate(states):
        positions = np.asarray(state.public_positions, dtype=float)
        velocities = np.asarray(state.velocities, dtype=float)
        if positions.ndim != 1 or velocities.shape != positions.shape:
            raise ValueError(
                f"arm state {index} positions and velocities must be one-dimensional and equal length"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError(f"arm state {index} positions and velocities must be finite")
        position_parts.append(positions)
        velocity_parts.append(velocities)

    joint_count = sum(len(part) for part in position_parts)
    if not 1 <= joint_count <= PX4_ARM_JOINT_CAPACITY:
        raise ValueError(
            f"arm joint count must be between 1 and {PX4_ARM_JOINT_CAPACITY}, got {joint_count}"
        )

    positions = np.zeros(PX4_ARM_JOINT_CAPACITY, dtype=float)
    velocities = np.zeros(PX4_ARM_JOINT_CAPACITY, dtype=float)
    positions[:joint_count] = np.concatenate(position_parts)
    velocities[:joint_count] = np.concatenate(velocity_parts)
    return EncodedPX4ArmJointState(
        joint_count=joint_count,
        positions=tuple(positions.tolist()),
        velocities=tuple(velocities.tolist()),
    )


__all__ = [
    "EncodedPX4ArmJointState",
    "PX4_ARM_JOINT_CAPACITY",
    "PX4_POLICY_ARM_JOINT_COUNT",
    "encode_px4_arm_joint_states",
]
