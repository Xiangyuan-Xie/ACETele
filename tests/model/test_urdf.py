from __future__ import annotations

from pathlib import Path

import pytest

from acetele.model import load_urdf_model


def _urdf(tmp_path: Path) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(
        """<robot name="test">
  <link name="base"/><link name="one"/><link name="two"/><link name="tool"/>
  <joint name="joint_1" type="revolute">
    <parent link="base"/><child link="one"/>
    <limit lower="-1" upper="1" effort="2" velocity="3"/>
  </joint>
  <joint name="fixed" type="fixed"><parent link="one"/><child link="two"/></joint>
  <joint name="joint_2" type="continuous">
    <parent link="two"/><child link="tool"/><limit effort="4" velocity="5"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    return path


def test_urdf_model_preserves_kinematic_order_and_limits(tmp_path):
    model = load_urdf_model(_urdf(tmp_path))
    metadata = model.arm_metadata(("joint_1",), require_limits=True)

    assert model.root_link == "base"
    assert model.movable_joint_order == ("joint_1", "joint_2")
    assert metadata.lower_limits == (-1.0,)
    assert metadata.upper_limits == (1.0,)
    assert metadata.velocity_limits == (3.0,)


def test_urdf_model_rejects_wrong_order_and_missing_continuous_limits(tmp_path):
    model = load_urdf_model(_urdf(tmp_path))

    with pytest.raises(ValueError, match="kinematic order"):
        model.arm_metadata(("joint_2", "joint_1"), require_limits=False)
    with pytest.raises(ValueError, match="finite lower and upper"):
        model.arm_metadata(("joint_2",), require_limits=True)
