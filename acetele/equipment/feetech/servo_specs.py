from __future__ import annotations

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

PROFILE_VELOCITY_UNIT_RAD_PER_SEC = 0.732 * np.pi / 30.0
PROFILE_ACCELERATION_UNIT_RAD_PER_SEC2 = 8.7 * np.pi / 180.0
