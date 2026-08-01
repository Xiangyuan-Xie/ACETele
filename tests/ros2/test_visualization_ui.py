from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
window_source = (
    project_root
    / "ros2/visualization_ros2/visualization_ros2/visualization_window.py"
)
node_source = (
    project_root
    / "ros2/visualization_ros2/visualization_ros2/visualization_node.py"
)
parameter_file = (
    project_root / "ros2/visualization_ros2/config/visualization_params.yaml"
)


def test_operator_layout_prioritizes_front_camera_and_structured_telemetry():
    source = window_source.read_text(encoding="utf-8")

    assert "grid.addWidget(self.front_rgb_panel, 0, 0, 1, 3)" in source
    assert "QSplitter(Qt.Horizontal" in source
    assert "QSplitter(Qt.Vertical" in source
    assert '["Stream", "State", "Latency"]' in source
    assert '["Joint", "Position", "Velocity", "Effort"]' in source
    assert "QPlainTextEdit" in source
    assert "def update_fonts" not in source
    assert "arm_state_view" not in source


def test_visualization_reports_expected_streams_before_first_message():
    source = node_source.read_text(encoding="utf-8")

    for stream in (
        "front_color",
        "front_depth",
        "wrist_color",
        "wrist_depth",
        "front_metadata",
        "wrist_metadata",
        "arm_state",
    ):
        assert f'"{stream}": None' in source


def test_visualization_parameters_target_the_runtime_node_name():
    parameters = parameter_file.read_text(encoding="utf-8")

    assert parameters.startswith("visualization:\n")
