from __future__ import annotations

import numpy as np


def wrap_to_pi(values):
    return (np.asarray(values, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def unwrap_near(values, references):
    values_array = np.asarray(values, dtype=float)
    references_array = np.asarray(references, dtype=float)
    return values_array + np.round((references_array - values_array) / (2 * np.pi)) * (2 * np.pi)
