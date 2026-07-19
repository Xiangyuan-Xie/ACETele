from __future__ import annotations

import operator
from typing import Any, Sequence, Tuple

import numpy as np


def normalize_joint_id(value: Any, *, field_name: str = "joint id") -> int:
    if isinstance(value, (bool, np.bool_, np.ndarray)):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def normalize_joint_ids(
    values: Sequence[Any],
    *,
    field_name: str = "joint ids",
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a one-dimensional sequence of integers")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional sequence of integers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a one-dimensional sequence of integers") from exc
    return tuple(
        normalize_joint_id(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(items)
    )


def normalize_joint_sign(value: Any, *, field_name: str = "joint sign") -> int:
    sign = normalize_joint_id(value, field_name=field_name)
    if sign not in (-1, 1):
        raise ValueError(f"{field_name} must be -1 or 1")
    return sign


def normalize_joint_signs(
    values: Sequence[Any],
    *,
    field_name: str = "joint signs",
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a one-dimensional sequence of integers")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional sequence of integers")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a one-dimensional sequence of integers") from exc
    return tuple(
        normalize_joint_sign(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(items)
    )


__all__ = [
    "normalize_joint_id",
    "normalize_joint_ids",
    "normalize_joint_sign",
    "normalize_joint_signs",
]
