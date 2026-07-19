from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from acetele.config.robot_config import MockJointConfig
from acetele.utils.angle import unwrap_near, wrap_to_pi
from acetele.utils.joint import normalize_joint_ids


class TorqueEnable(Enum):
    Disable = 0
    Enable = 1


def _validate_torque_enable(enable: Any) -> TorqueEnable:
    if not isinstance(enable, TorqueEnable):
        raise ValueError("enable must be a TorqueEnable value")
    return enable


def _normalize_profile_value(name: str, value: Any, count: int) -> Any:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        result: Any = float(array)
        finite = np.isfinite(result)
        nonnegative = result >= 0.0
    elif array.ndim == 1 and len(array) == count:
        result = array.copy()
        finite = np.all(np.isfinite(result))
        nonnegative = np.all(result >= 0.0)
    else:
        raise ValueError(f"{name} must be scalar or match ids length")
    if not finite:
        raise ValueError(f"{name} must contain only finite values")
    if name in ("velocities", "accelerations") and not nonnegative:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class JointDeviceState:
    public_positions: np.ndarray
    raw_positions: np.ndarray
    velocities: np.ndarray
    motor_torque_magnitude: np.ndarray
    motor_torque_signed: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=float).copy()
            for name in self.__dataclass_fields__
        }
        lengths = {len(array) for array in arrays.values() if array.ndim == 1}
        if any(array.ndim != 1 for array in arrays.values()) or len(lengths) != 1:
            raise ValueError("joint device state fields must be one-dimensional arrays of equal length")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    def act(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.public_positions.copy(),
            self.velocities.copy(),
            self.motor_torque_magnitude.copy(),
        )


@runtime_checkable
class JointDevice(Protocol):
    @property
    def joint_names(self) -> tuple[str, ...]: ...

    @property
    def ids(self) -> np.ndarray: ...

    def get_state(self) -> JointDeviceState: ...

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None: ...

    def validate_position_command(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None: ...

    def set_torque_enable(
        self,
        enable: TorqueEnable,
        ids: Optional[Sequence[int]] = None,
    ) -> None: ...

    def close(self) -> None: ...


class MockJointDevice:
    """In-memory joint device used by mock robot backends."""

    def __init__(
        self,
        joints: Sequence[MockJointConfig],
        *,
        wrap_public_positions: bool = False,
        raw_position_scales: Optional[Sequence[float]] = None,
    ):
        self._joints = tuple(joints)
        if not self._joints:
            raise ValueError("mock joint device requires at least one joint")
        if type(wrap_public_positions) is not bool:
            raise ValueError("wrap_public_positions must be a boolean")
        self._wrap_public_positions = wrap_public_positions
        self._joint_names = tuple(joint.name for joint in self._joints)
        self._ids = np.asarray(
            normalize_joint_ids(
                [joint.joint_id for joint in self._joints],
                field_name="mock device joint ids",
            ),
            dtype=int,
        )
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("mock joint names must be unique")
        if len(set(self._ids.tolist())) != len(self._ids):
            raise ValueError("mock joint ids must be unique")
        if raw_position_scales is None:
            self._raw_position_scales = np.ones(len(self._joints), dtype=float)
        else:
            self._raw_position_scales = np.asarray(
                raw_position_scales,
                dtype=float,
            )
            if self._raw_position_scales.shape != (len(self._joints),) or not np.all(
                np.isfinite(self._raw_position_scales)
                & (self._raw_position_scales > 0.0)
            ):
                raise ValueError(
                    "raw_position_scales must contain one finite positive value per joint"
                )
            self._raw_position_scales = self._raw_position_scales.copy()
        self._lower_limits = np.asarray([joint.lower_limit for joint in self._joints], dtype=float)
        self._upper_limits = np.asarray([joint.upper_limit for joint in self._joints], dtype=float)
        self._max_velocities = np.asarray([joint.max_velocity for joint in self._joints], dtype=float)
        self._positions = np.asarray([joint.initial_position for joint in self._joints], dtype=float)
        self._velocities = np.zeros(len(self._joints), dtype=float)
        self._efforts = np.zeros(len(self._joints), dtype=float)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def ids(self) -> np.ndarray:
        return self._ids.copy()

    @property
    def max_velocities(self) -> np.ndarray:
        return self._max_velocities.copy()

    def get_state(self) -> JointDeviceState:
        public_positions = (
            wrap_to_pi(self._positions)
            if self._wrap_public_positions
            else self._positions
        )
        return JointDeviceState(
            public_positions=public_positions,
            raw_positions=self._positions * self._raw_position_scales,
            velocities=self._velocities * self._raw_position_scales,
            motor_torque_magnitude=np.abs(self._efforts),
            motor_torque_signed=self._efforts,
        )

    def act(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_state().act()

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        self.validate_position_command(
            positions,
            ids=ids,
            velocities=velocities,
            accelerations=accelerations,
            torque=torque,
        )
        positions_array = np.asarray(positions, dtype=float)
        indices = self._resolve_indices(ids, len(positions_array))
        if self._wrap_public_positions:
            limit_midpoints = (
                self._lower_limits[indices] + self._upper_limits[indices]
            ) / 2.0
            positions_array = unwrap_near(positions_array, limit_midpoints)
        next_positions = self._positions.copy()
        next_positions[indices] = np.clip(
            positions_array,
            self._lower_limits[indices],
            self._upper_limits[indices],
        )
        self._positions = next_positions

    def validate_position_command(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.ndim != 1 or not np.all(np.isfinite(positions_array)):
            raise ValueError("joint positions must be a finite one-dimensional sequence")
        self._resolve_indices(ids, len(positions_array))
        _normalize_profile_value("velocities", velocities, len(positions_array))
        _normalize_profile_value("accelerations", accelerations, len(positions_array))
        _normalize_profile_value("torque", torque, len(positions_array))

    def _resolve_indices(self, ids: Optional[Sequence[int]], value_count: int) -> np.ndarray:
        if ids is None:
            if value_count != len(self._positions):
                raise ValueError("joint positions must match the configured joint count")
            return np.arange(len(self._positions), dtype=int)
        ids_array = np.asarray(
            normalize_joint_ids(ids, field_name="joint ids"),
            dtype=int,
        )
        if len(ids_array) != value_count or len(set(ids_array.tolist())) != len(ids_array):
            raise ValueError("joint ids must be unique and match positions")
        index_by_id = {int(joint_id): index for index, joint_id in enumerate(self._ids)}
        try:
            return np.asarray([index_by_id[int(joint_id)] for joint_id in ids_array], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown joint id {int(exc.args[0])}") from exc

    def set_torque_enable(
        self,
        enable: TorqueEnable,
        ids: Optional[Sequence[int]] = None,
    ) -> None:
        _validate_torque_enable(enable)
        if ids is not None:
            self._resolve_indices(ids, len(ids))

    def close(self) -> None:
        pass


class CompositeJointDevice:
    """Expose several disjoint joint devices through one ordered interface."""

    def __init__(self, devices: Sequence[JointDevice]):
        self._devices = tuple(devices)
        if not self._devices:
            raise ValueError("composite joint device requires at least one device")
        self._joint_names = tuple(name for device in self._devices for name in device.joint_names)
        self._device_ids = tuple(
            np.asarray(
                normalize_joint_ids(device.ids, field_name="joint device ids"),
                dtype=int,
            )
            for device in self._devices
        )
        self._ids = np.concatenate(self._device_ids)
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("joint device names must be globally unique")
        if len(set(self._ids.tolist())) != len(self._ids):
            raise ValueError("joint device ids must be globally unique")

    @property
    def devices(self) -> tuple[JointDevice, ...]:
        return self._devices

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_names

    @property
    def ids(self) -> np.ndarray:
        return self._ids.copy()

    def get_state(self) -> JointDeviceState:
        states = tuple(device.get_state() for device in self._devices)
        values = {
            name: np.concatenate([getattr(state, name) for state in states])
            for name in JointDeviceState.__dataclass_fields__
        }
        return JointDeviceState(**values)

    def act(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.get_state().act()

    def set_position(
        self,
        positions: Sequence[float],
        ids: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[float] | float] = None,
        accelerations: Optional[Sequence[float] | float] = None,
        torque: Optional[Sequence[float] | float] = None,
    ) -> None:
        ids_array, positions_array = self._validate_command(ids, positions)
        profiles = {
            "velocities": self._validate_profile("velocities", velocities, len(ids_array)),
            "accelerations": self._validate_profile("accelerations", accelerations, len(ids_array)),
            "torque": self._validate_profile("torque", torque, len(ids_array)),
        }
        routes = []
        for device, device_ids in zip(self._devices, self._device_ids):
            mask = np.isin(ids_array, device_ids)
            if not np.any(mask):
                continue
            kwargs = {
                name: value if value is None or np.asarray(value).ndim == 0 else np.asarray(value)[mask]
                for name, value in profiles.items()
            }
            routes.append((device, positions_array[mask], ids_array[mask], kwargs))

        for device, device_positions, device_ids, kwargs in routes:
            device.validate_position_command(
                device_positions,
                ids=device_ids,
                **kwargs,
            )
        for device, device_positions, device_ids, kwargs in routes:
            device.set_position(
                device_positions,
                ids=device_ids,
                **kwargs,
            )

    def _validate_command(
        self,
        ids: Optional[Sequence[int]],
        positions: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        ids_array = (
            self._ids.copy()
            if ids is None
            else np.asarray(normalize_joint_ids(ids, field_name="joint ids"), dtype=int)
        )
        positions_array = np.asarray(positions, dtype=float)
        if positions_array.ndim != 1 or len(positions_array) != len(ids_array):
            raise ValueError("positions must be one-dimensional and match ids length")
        if not np.all(np.isfinite(positions_array)):
            raise ValueError("positions must contain only finite values")
        if len(set(ids_array.tolist())) != len(ids_array):
            raise ValueError("joint ids must be unique")
        unknown = ids_array[~np.isin(ids_array, self._ids)]
        if len(unknown):
            raise ValueError(f"unknown joint id {int(unknown[0])}")
        return ids_array, positions_array.copy()

    @staticmethod
    def _validate_profile(name: str, value: Any, count: int) -> Any:
        return _normalize_profile_value(name, value, count)

    def set_torque_enable(
        self,
        enable: TorqueEnable,
        ids: Optional[Sequence[int]] = None,
    ) -> None:
        _validate_torque_enable(enable)
        ids_array = (
            self._ids.copy()
            if ids is None
            else np.asarray(normalize_joint_ids(ids, field_name="joint ids"), dtype=int)
        )
        if len(set(ids_array.tolist())) != len(ids_array):
            raise ValueError("joint ids must be unique")
        unknown = ids_array[~np.isin(ids_array, self._ids)]
        if len(unknown):
            raise ValueError(f"unknown joint id {int(unknown[0])}")
        for device, device_ids in zip(self._devices, self._device_ids):
            selected = ids_array[np.isin(ids_array, device_ids)]
            if len(selected):
                device.set_torque_enable(enable, ids=selected)

    def close(self) -> None:
        closed: set[int] = set()
        first_error: Optional[BaseException] = None
        for device in self._devices:
            if id(device) not in closed:
                try:
                    device.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                closed.add(id(device))
        if first_error is not None:
            raise first_error


__all__ = [
    "CompositeJointDevice",
    "JointDevice",
    "JointDeviceState",
    "MockJointDevice",
    "TorqueEnable",
]
