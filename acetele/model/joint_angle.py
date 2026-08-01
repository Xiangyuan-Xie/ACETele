"""Joint-angle transformations shared by control and runtime routing."""

from __future__ import annotations

import numpy as np


def wrap_to_pi(values):
    """Wrap angles to the half-open interval ``[-pi, pi)``."""

    return (np.asarray(values, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_near(values, references):
    """Choose each angle's equivalent turn nearest the supplied reference."""

    values_array = np.asarray(values, dtype=float)
    references_array = np.asarray(references, dtype=float)
    return values_array + np.round(
        (references_array - values_array) / (2.0 * np.pi)
    ) * (2.0 * np.pi)


__all__ = ["unwrap_near", "wrap_to_pi"]
