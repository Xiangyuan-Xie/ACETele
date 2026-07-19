# ACETele Agent Guide

This file is for AI agents and maintainers working in this repository. It summarizes the project shape, safety constraints, and checks that matter most.

## Project Snapshot

ACETele is a Python robotics teleoperation package with ROS 2 deployment packages. It controls real hardware through FEETECH HLS servos, joystick inputs, leader/follower robot abstractions, gripper control, PX4-facing messages, RealSense camera integration, and data conversion tools.

The project is experimental. Treat hardware-facing code as safety-critical: serial ports, calibration, PID tuning, gravity compensation, and diagnostics can move actuators or write servo parameters.

## Repository Map

```text
acetele/config/       TOML config loader and default robot configs.
acetele/core/         make_robot factory and FEETECH calibration entry point.
acetele/equipment/    Joint-device contract, FEETECH arm/gripper, dexterous hands, joystick driver.
acetele/robot/        Shared joint-robot composition plus leader/follower topology and ROS 2 adapters.
acetele/deploy/       ROS 2 packages plus PX4 and RealSense deployment dependencies.
acetele/tools/        rosbag/HDF5 tooling and hardware diagnostics.
acetele/utils/        Teleop sync enums plus angle and joint-ID helpers.
tests/                Unit tests and ROS 2 behavior tests.
```

Avoid editing third-party or vendored areas unless the user explicitly asks:

```text
acetele/deploy/realsense-ros/
acetele/deploy/px4_msgs/
acetele/equipment/feetech/feetech_sdk/
```

## Core Contracts

`ConfigLoader` reads `acetele/config/default.toml`, optionally follows `basic.config_file`, and returns
a typed `RobotConfig`. `make_robot()` owns the single `(robot_type, runtime)` entry-point map.

Supported robot types:

```text
ace_leader
ace_follower
ace_follower_dual
```

Device backends and runtimes are separate:

```text
backend = physical | mock
runtime = standalone | ros2
```

Each `[arms.<name>]` table contains one arm and its optional nested
`[arms.<name>.end_effector]`. Do not restore the removed `[linker.*]`, `[gripper.*]`, `variant`, or
`gripper_type` schema. Gripper travel is an explicit physical value in `travel_range_rad`.

`JointDevice` is the shared arm/end-effector protocol, and every device returns `JointDeviceState`.
`CompositeJointDevice` implements command routing and state aggregation. `JointRobot` builds named
`ArmAssembly` instances and owns shared serial drivers. Direct robot APIs use combined joint order:
all arm joints first, then configured end-effector joints in arm order.

`BaseRobot.act()` returns `(positions, velocities, efforts)`.
`BaseRobot.name` uses `<robot_type>_<backend>_<runtime>`. Hardware `joint_ids` are independent from
URDF/Pinocchio/ROS 2 names. Every arm TOML table must explicitly define `joint_names`, and every
FEETECH gripper must define `joint_name`; never derive kinematic names from servo bus IDs.

ROS 2 topics intentionally split arm and gripper traffic:

```text
/ace_leader/arm/command
/ace_follower/arm/state
/ace_leader/gripper/command
/ace_follower/gripper/state
/ace_leader/arm/sync_mode
/ace_follower/arm/sync_status
```

The PX4 bridge still publishes `/fmu/in/arm_joint_state` as the 5D `[arm..., gripper]` vector expected by the adapter.

`FeeTechArm` positions are radians. `FeeTechGripper` public positions are normalized to `[0.0, 1.0]`.
`set_position()` APIs accept `velocities`, `accelerations`, and `torque`; do not reintroduce older
singular/profile/current keyword arguments unless the tests and callers are intentionally changed.

## Hardware Safety

Ask before running commands that may touch hardware, serial ports, ROS 2 live nodes, or servo nonvolatile state. This includes:

```bash
python -m acetele.core.calibrate --config /path/to/physical_robot.toml
python -m acetele.tools.backlash_diagnostics ...
python -m acetele.tools.gravity_compensation_diagnostics ...
python -m acetele.tools.feetech_pid_autotune ...
ros2 launch ace_robot_ros2 ace_robot.launch.py ...
ros2 launch acetele_bringup follower_system.launch.py
```

Before hardware runs, confirm the intended config file, serial port, robot side, power state, mechanical clearance, and emergency stop plan.

Calibration rejects `backend="mock"` before creating a driver. Do not add an implicit backend
override or bypass this check.

Prefer `backend="mock"` or test doubles for local validation. Never change calibration offsets, PID values, torque enable behavior, current limits, or gravity compensation constants as a casual refactor.

## Development Workflow

Install the Python package in editable mode:

```bash
python -m pip install -e .
```

Install local development tools:

```bash
python -m pip install pytest pre-commit
pre-commit install
```

Run focused tests while iterating:

```bash
python -m pytest tests/config/test_config_loader.py -q
python -m pytest tests/equipment/feetech -q
python -m pytest tests/robot -q
```

Run the full Python test suite before broad changes:

```bash
python -m pytest
```

Run pre-commit before handing off larger code changes:

```bash
pre-commit run --all-files
```

The pytest config excludes `acetele/deploy/realsense-ros` because it is a large third-party submodule.

## Packaging Notes

Package metadata is duplicated in `pyproject.toml` and `setup.py`; keep them aligned. The declared Python requirement is `>=3.9`.

Package data includes TOML configs and robot description assets:

```text
acetele.config/*.toml
acetele.robot.ace_follower/description/*.urdf
acetele.robot.ace_follower/description/meshes/*.STL
acetele.robot.ace_leader/description/*.urdf
acetele.robot.ace_leader/description/*.xml
acetele.robot.ace_leader/description/meshes/*.STL
```

If a change affects installed resources, update both packaging files and `tests/test_packaging_metadata.py` when appropriate.

## ROS 2 Notes

`ace_robot_ros2/ace_robot_node.py` creates a `ConfigLoader` with `runtime_override="ros2"`; the
configured `physical` or `mock` device backend remains unchanged. Passing `config_path` selects the
robot TOML file at launch time.

Useful commands:

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py config_path:=/path/to/ace_follower.toml
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
ros2 run joystick_ros2 manual_control
```

`acetele_bringup/follower_system.launch.py` expects external packages such as `realsense2_camera` and `ros2_px4_odometry`. If those dependencies are absent, test lower-level packages directly.

## Common Change Patterns

When adding a robot topology or runtime:

1. Add the implementation under `acetele/robot/...`.
2. Update `_ROBOT_ENTRYPOINTS` in `acetele/core/make_robot.py`.
3. Add or update TOML config files under `acetele/config/`.
4. Add tests for config loading and factory behavior.
5. Update README usage/configuration notes.

When adding an end effector:

1. Add a descriptive typed config and a `JointDevice` implementation.
2. Put dexterous-hand models under `acetele/equipment/dexterous_hands/`.
3. Register creation in `end_effector_factory.py`.
4. Keep model codes such as O6 contextualized by the directory and class/config names.

When changing arm or gripper command semantics:

1. Update both `ace_leader` and `ace_follower` paths.
2. Preserve direct API ordering: arm joints, then gripper.
3. Preserve ROS 2 arm/gripper topic separation unless the caller contract is intentionally changed.
4. Update tests under `tests/equipment/feetech/` and `tests/robot/`.

When changing ROS 2 topics or sync behavior:

1. Update leader and follower node classes together.
2. Update launch/config YAML if parameters change.
3. Update data collector topic regexes or bag conversion mappings if recorded data changes.
4. Add tests under `tests/deploy/` when possible.

## Current Workspace Caution

This repository may contain user-created or generated untracked files. Always check:

```bash
git status --short
```

Do not delete, overwrite, or reformat unrelated untracked files. Work with user changes if they touch the files you need; otherwise leave them alone.

## Documentation Style

The root `README.md` is the Simplified Chinese default. It follows the ACESim-style presentation: top anchor, centered badges/title/slogan/language switch, Chinese table of contents, project overview, status, tech stack, quick start, usage, project layout, roadmap, contributing, license, contact, and acknowledgments. Keep every section heading and back-to-top link in Chinese.

Keep `README.en.md` as the English counterpart when changing public documentation.

The root project is licensed under Apache-2.0. Keep `LICENSE`, README license sections, `pyproject.toml`, `setup.py`, and ROS 2 `package.xml` license metadata aligned when changing licensing-related files.
