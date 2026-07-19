import numpy as np
import pytest

from acetele.utils.joint import normalize_joint_id, normalize_joint_ids


def test_joint_id_normalization_accepts_python_and_numpy_integers():
    assert normalize_joint_id(4) == 4
    assert normalize_joint_id(np.int64(5)) == 5
    assert normalize_joint_ids([0, np.int32(1)]) == (0, 1)


@pytest.mark.parametrize("value", [4.0, "4", True, np.bool_(False)])
def test_joint_id_normalization_rejects_non_integer_types(value):
    with pytest.raises(ValueError, match="integer"):
        normalize_joint_id(value)


@pytest.mark.parametrize(
    "values",
    ["01", np.array([[0, 1]], dtype=int), [[0], [1]]],
)
def test_joint_id_normalization_rejects_non_vector_sequences(values):
    with pytest.raises(ValueError, match="one-dimensional|integer"):
        normalize_joint_ids(values)
