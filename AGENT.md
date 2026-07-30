# ACETele Agent Guide

This guide defines the architecture, safety boundaries, and verification expectations for work in
this repository.

## Project Snapshot

ACETele is a Python robotics teleoperation package with ROS 2 deployment packages. It supports
FEETECH packet and Modbus buses, FashionStar UART/RS485 servos, Linker Hand RS485 devices, joystick
input, leader/follower synchronization, PX4-facing messages, RealSense integration, and data tools.

Hardware-facing code is safety-critical. Serial traffic, torque state, calibration, control
compensation, and watchdog behavior can move actuators or alter nonvolatile device state.

## Repository Map

```text
acetele/core/         Immutable vendor-neutral state, command, capability, and protocol contracts.
acetele/config/       Strict TOML loader and immutable RobotSpec types.
acetele/model/        Packaged URDF assets, metadata validation, and optional Pinocchio reduction.
acetele/hardware/     Serial Actor, vendor protocols/profiles, state estimation, joystick, and mock.
acetele/control/      Thread-free command conditioning and compensation pipeline.
acetele/runtime/      Robot assembly, lifecycle, safety state machine, and teleop sessions.
acetele/deploy/       ROS 2 packages plus PX4 and RealSense deployment dependencies.
acetele/tools/        Static preflight and hardware calibration tools.
acetele/utils/        Teleop synchronization enums and small numeric helpers.
tests/                Unit, architecture, packaging, runtime, and ROS adapter tests.
```

Do not edit vendored areas unless the user explicitly asks:

```text
acetele/deploy/realsense-ros/
acetele/deploy/px4_msgs/
```

## Architecture

The dependency direction is fixed:

```text
core <- config/model/hardware/control <- runtime <- ROS 2 adapters
```

- `core` must not import configuration, hardware, runtime, or ROS packages.
- `hardware` must not import ROS or deployment packages.
- ROS nodes compose a `RobotRuntime`; they do not inherit from robot or hardware classes.
- Every physical serial port has exactly one `SerialBusActor` owner.
- Public control uses canonical joint names and SI units. Servo IDs, counts, registers, and packet
  formats remain inside hardware profiles, protocols, and diagnostics.
- Constructors perform static validation only. `RobotRuntime.connect()` is the hardware lifecycle
  boundary that may open serial ports and start threads.

The old `equipment`, `robot`, `ConfigLoader`, `RobotConfig`, `make_robot()`, and `act()` architecture
has been removed. Do not restore compatibility wrappers or schema fallbacks. Migrate callers to
`RobotSpec`, `RobotRuntime.read()`, and `RobotRuntime.write()`.

## Core Contracts

Use the immutable contracts in `acetele.core`:

```text
JointState
JointCommand
SensorState
RobotState
RobotCommand
DeviceCapabilities
```

Arrays owned by these contracts are copied and read-only. A command includes a monotonic submission
time, deadline, generation, explicit unit, and optional per-joint limits. Validate a complete command
before publishing any part of it to hardware.

Configuration uses one strict schema:

```toml
[basic]
model = "ace_follower"
backend = "physical"

[buses.arm]
type = "feetech_packet"
port = "/dev/ttyUSB0"
baudrate = 1000000
cycle_hz = 100
physical_layer = "rs485"
family = "sms"
external_estop = true

[arms.single]
bus = "arm"

[[arms.single.joints]]
name = "joint_1"
servo_id = 1
servo_model = "SM8512BL"
direction = 1
home_position_rad = 0.0
```

Unknown fields are errors. Joint names are the URDF, Pinocchio, and ROS 2 identity; `servo_id` is
only a bus address. Do not derive one from the other. A model must have an exact profile backed by
official protocol information; never substitute the nearest model. Any physical bus whose protocol
cannot verify torque disable per device must declare `external_estop = true` and have the independent
hardware stop installed; the configuration flag alone is not a safety mechanism.

Public HLS documentation does not provide a trustworthy model-register mapping for every supported
model. HLS profiles therefore record the observed register without comparing it to a guessed value.
When `expected_model_number` is explicitly configured from a trustworthy source, connection must
enforce it. SMS profiles with documented model numbers always enforce identity. A physical bus whose
protocol cannot expose model identity must explicitly set `allow_unverified_identity = true`; static
preflight must report that identity remains unverified. FashionStar firmware is readable and must be
compared with `firmware_version` even though the generic protocol does not expose a product model ID.

## Bus And State Rules

Each Actor owns:

- a strict FIFO for safety and lifecycle transactions;
- a bounded latest-value mailbox for streaming motion commands;
- periodic fast-state reads and budgeted slow telemetry;
- generation and deadline checks;
- an atomic immutable state snapshot.

Streaming commands may replace commands that have not started. Safety transactions may not be
dropped or reordered. A motion write must not starve the following state read. Use blocking serial
I/O with monotonic deadlines and condition waits; do not add busy polling, one thread per device, or
an unbounded command queue.

Physical joint positions and velocities pass through `RobustJointStateEstimator`. Keep filtering
low-latency and preserve NIS/physical innovation gates, covariance checks, acceleration limits, and
diagnostics. Policies and ROS adapters consume the same estimated state.

Known HLS profiles may estimate torque only from their documented model-specific KT and no-load
current. Do not apply those constants to RS485 models without official parameters.

## Hardware Safety

Ask before running anything that may open hardware, launch live ROS nodes, enable torque, move a
joint, or write nonvolatile state. Confirm the intended config, robot side, serial port, power state,
mechanical clearance, and emergency-stop plan.

Safe static preflight does not open hardware:

```bash
python -m acetele.tools.check_robot_spec /path/to/robot.toml
```

Preflight must validate URDF order and limits, exact profiles, firmware capabilities, unique port
ownership, and worst-case bus utilization. Utilization above 70% is a configuration error.

The safety state machine is:

```text
DISCONNECTED -> SAFE_DISABLED -> READY -> ACTIVE -> HOLD/FAULT
```

- Commands older than their deadline or from an old generation are discarded.
- Lost follower heartbeats enter `HOLD`, clear pending motion, and require synchronization again.
- Emergency stop clears motion, increments generation, executes the strongest supported hardware
  action, and remains latched until explicit reset.
- Stale state, repeated protocol failures, device reset, or a missing arm joint enters `FAULT`.
- Devices without verifiable torque disable require an independent physical emergency stop. Every
  affected physical bus spec must declare that external stop.

FEETECH packet home calibration is exposed through
`python -m acetele.tools.calibrate_feetech_home <config> --yes`. Keep calibration profile-specific,
require `SAFE_DISABLED`, validate the complete plan before connecting hardware, and preserve queue
barriers and bounded cleanup. Other protocols must not reuse the FEETECH procedure without their own
documented safety transaction.

## ROS 2

`ace_robot_ros2` accepts only new-schema specs and dispatches only to composed `RuntimeLeaderNode`
or `RuntimeFollowerNode` adapters. The generic launch defaults to the packaged physical HLS TTL
Leader spec; system-level launches default to their matching Leader/Follower specs. Production runs
should pass `config_path` explicitly after preflight:

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py config_path:=/path/to/robot.toml
```

High-rate arm/end-effector commands and state use `BEST_EFFORT + KEEP_LAST(1) + VOLATILE`.
Synchronization mode/status use `RELIABLE + KEEP_LAST(1)`. Keep callbacks non-blocking and use
monotonic time for deadlines and heartbeat logic; use the ROS clock only for message headers.

The PX4 bridge publishes `/fmu/in/arm_joint_state` as an arm-only fixed-capacity 14-joint message.
`joint_count` marks the dense valid prefix. Grippers and dexterous hands are excluded. Parallel
grippers use the existing gripper topics; dexterous hands use the separate end-effector topics.

## Development Workflow

Install and run focused tests:

```bash
python -m pip install -e .
python -m pip install pytest pre-commit
python -m pytest tests/config tests/hardware tests/runtime tests/control -q
```

Before handing off broad changes, run:

```bash
python -m pytest
python -m compileall acetele
git diff --check
pre-commit run --all-files
```

The pytest configuration excludes `acetele/deploy/realsense-ros` because it is a large third-party
submodule.

## Packaging

Keep metadata in `pyproject.toml` and `setup.py` aligned. Python support is `>=3.10`. Package data
includes:

```text
acetele.config/*.toml
acetele.model.robots.ace_follower/description/*.urdf
acetele.model.robots.ace_follower/description/meshes/*.STL
acetele.model.robots.ace_leader/description/*.urdf
acetele.model.robots.ace_leader/description/*.xml
acetele.model.robots.ace_leader/description/meshes/*.STL
```

When installed resources change, update both packaging files and
`tests/test_packaging_metadata.py`. Never package `acetele/equipment` or `acetele/robot`; those
names belong to the removed architecture.

## Change Checklist

When adding a servo or hand profile:

1. Record the official source URL, document version, and checksum.
2. Add codec golden-frame tests before integrating the protocol.
3. Keep serial ownership, deadlines, retries, and scheduling in the Actor path.
4. Add bandwidth and capability preflight checks.
5. Complete protocol-level tests before requesting hardware validation.

When changing command, safety, or ROS behavior, update both leader and follower paths, preserve
joint ordering, and add focused regression tests. Always inspect `git status --short`; do not revert
or overwrite unrelated user changes.

`README.md` is the Simplified Chinese default and `README.en.md` is its English counterpart. Keep
their public behavior and command examples aligned. The project is Apache-2.0 licensed; keep
`LICENSE`, package metadata, and ROS package metadata consistent.
