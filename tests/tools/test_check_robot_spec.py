from __future__ import annotations

from pathlib import Path

from acetele.tools.check_robot_spec import check_robot_spec

project_root = Path(__file__).resolve().parents[2]


def test_packaged_examples_pass_static_preflight_without_hardware():
    sm_lines = check_robot_spec(
        project_root / "acetele/config/ace_follower/feetech_sms_rs485.toml"
    )
    fashionstar_lines = check_robot_spec(
        project_root / "acetele/config/ace_follower/fashionstar_rs485.toml"
    )
    leader_hls_lines = check_robot_spec(
        project_root / "acetele/config/ace_leader/feetech_hls_ttl.toml"
    )
    follower_hls_lines = check_robot_spec(
        project_root / "acetele/config/ace_follower/feetech_hls_ttl.toml"
    )

    assert "type=feetech_packet" in sm_lines[1]
    assert "rate=100Hz" in sm_lines[1]
    assert "type=fashionstar_rs485" in fashionstar_lines[1]
    assert "rate=25Hz" in fashionstar_lines[1]
    assert "verified_identity=false" in fashionstar_lines[1]
    assert "type=feetech_packet" in leader_hls_lines[1]
    assert "rate=100Hz" in leader_hls_lines[1]
    assert "verified_identity=false" in leader_hls_lines[1]
    assert "type=feetech_packet" in follower_hls_lines[1]
    assert "rate=100Hz" in follower_hls_lines[1]
