from __future__ import annotations

import math

from acetele.config.robot_config import MockJointConfig, O6DexterousHandConfig
from acetele.equipment.joint_device import JointDevice, MockJointDevice


def _mock_joint_specs(config: O6DexterousHandConfig) -> tuple[MockJointConfig, ...]:
    if config.mock_joints:
        return config.mock_joints
    return tuple(
        MockJointConfig(
            name=name,
            joint_id=joint_id,
            initial_position=0.0,
            lower_limit=-math.pi,
            upper_limit=math.pi,
            max_velocity=2.0,
        )
        for name, joint_id in zip(config.joint_names, config.joint_ids)
    )


def create_o6_dexterous_hand(
    config: O6DexterousHandConfig,
    *,
    backend: str,
) -> JointDevice:
    if backend == "mock":
        return MockJointDevice(_mock_joint_specs(config))
    raise RuntimeError("O6 dexterous hand physical backend is not implemented")


__all__ = ["create_o6_dexterous_hand"]
