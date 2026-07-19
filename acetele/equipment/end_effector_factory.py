from __future__ import annotations

import math
from functools import singledispatch
from typing import Any, Callable

from acetele.config.robot_config import (
    FeeTechGripperConfig,
    MockJointConfig,
    O6DexterousHandConfig,
)
from acetele.equipment.dexterous_hands import create_o6_dexterous_hand
from acetele.equipment.feetech.gripper import FeeTechGripper
from acetele.equipment.joint_device import JointDevice, MockJointDevice


@singledispatch
def _dispatch_end_effector(
    config: object,
    *,
    backend: str,
    driver: Any = None,
) -> JointDevice:
    raise RuntimeError(f"no {backend} end effector is registered for {type(config).__name__}")


class EndEffectorFactory:
    def __call__(
        self,
        config: object,
        *,
        backend: str,
        driver: Any = None,
    ) -> JointDevice:
        if backend not in ("mock", "physical"):
            raise ValueError("end effector backend must be 'mock' or 'physical'")
        return _dispatch_end_effector(config, backend=backend, driver=driver)

    def register(
        self,
        implementation: Callable[..., JointDevice],
    ) -> Callable[..., JointDevice]:
        return _dispatch_end_effector.register(implementation)


create_end_effector = EndEffectorFactory()


@create_end_effector.register
def _create_gripper(
    config: FeeTechGripperConfig,
    *,
    backend: str,
    driver: Any = None,
) -> JointDevice:
    if backend == "mock":
        joint = MockJointConfig(
            name=config.joint_name,
            joint_id=config.joint_id,
            initial_position=config.home_pose,
            lower_limit=0.0,
            upper_limit=1.0,
            max_velocity=1.0,
        )
        raw_position_scale = config.travel_range_counts * math.pi / 2048.0
        return MockJointDevice(
            (joint,),
            raw_position_scales=(raw_position_scale,),
        )
    if driver is None:
        raise RuntimeError("physical FEETECH gripper requires a driver")
    return FeeTechGripper(config, driver=driver)


@create_end_effector.register
def _create_o6_dexterous_hand(
    config: O6DexterousHandConfig,
    *,
    backend: str,
    driver: Any = None,
) -> JointDevice:
    del driver
    return create_o6_dexterous_hand(config, backend=backend)


__all__ = ["create_end_effector"]
