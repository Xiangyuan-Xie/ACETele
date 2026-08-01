import numpy as np
import pytest
from ace_operator_ui import JointView, OperatorSnapshot


def test_operator_snapshot_detaches_images_and_joint_values():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    snapshot = OperatorSnapshot(
        images={"front_color": image},
        joints=JointView(1, ("joint_1",), (0.1,), (0.2,), (0.3,)),
    )

    image[:] = 5
    assert not snapshot.images["front_color"].any()
    with pytest.raises(ValueError):
        snapshot.images["front_color"][0, 0, 0] = 1


def test_joint_view_rejects_nonfinite_state():
    with pytest.raises(ValueError, match="finite"):
        JointView(1, ("joint_1",), (float("nan"),))
