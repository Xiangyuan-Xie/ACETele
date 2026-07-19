from __future__ import annotations

from typing import Sequence

import numpy as np

KT_MAPPING = {
    "HL3960": 1.0 / 14.84,
    "HL3950": 1.0 / 20.8,
    "HL3930": 1.0 / 12.5,
    "HL3915": 1.0 / 9.3,
}

NO_LOAD_CURRENT = {
    "HL3960": 300,
    "HL3950": 330,
    "HL3930": 150,
    "HL3915": 260,
}

HLS_PROFILE_DEFAULTS_BY_SERVO = {
    "HL3960": {"acceleration": 0, "current": 1000, "velocity": 110},
    "HL3950": {"acceleration": 0, "current": 1000, "velocity": 110},
    "HL3930": {"acceleration": 250, "current": 1000, "velocity": 100},
    "HL3915": {"acceleration": 0, "current": 500, "velocity": 250},
}


def validate_feetech_servo_models(
    servo_models: Sequence[str],
    *,
    context: str,
) -> None:
    supported_models = (
        set(HLS_PROFILE_DEFAULTS_BY_SERVO)
        & set(KT_MAPPING)
        & set(NO_LOAD_CURRENT)
    )
    unknown_models = sorted(set(servo_models) - supported_models)
    if unknown_models:
        raise ValueError(
            f"{context} has unsupported servo models: {unknown_models}; "
            "each model must define profile, KT, and no-load-current data"
        )


PROFILE_VELOCITY_UNIT_RAD_PER_SEC = 0.732 * np.pi / 30.0
PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2 = 8.7 * np.pi / 180.0
