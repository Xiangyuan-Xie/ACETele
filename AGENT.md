# ACETele Agent Guide

This guide defines the architecture, safety boundaries, and verification expectations for work in
this repository.

## Project Snapshot

ACETele is a Python robotics teleoperation package with parallel ROS 2 and ZeroMQ adapters. It supports
FEETECH packet and Modbus buses, FashionStar UART/RS485 servos, Linker Hand RS485 devices, joystick
input, leader/follower synchronization, PX4-facing messages, and RealSense image transport.

Hardware-facing code is safety-critical. Serial traffic, torque state, calibration, control
compensation, and watchdog behavior can move actuators or alter nonvolatile device state.

## Repository Map

```text
acetele/core/         Immutable vendor-neutral state, command, pose, and unit contracts.
acetele/specification/ Immutable bus, control, joint, and robot specifications.
acetele/config/       Strict TOML loader, packaged presets, and resource catalog.
acetele/model/        Packaged URDF assets, metadata validation, and Pinocchio kinematics/dynamics.
acetele/control/      Thread-free position, Cartesian, gravity, and null-space control algorithms.
acetele/estimation/   Robust low-latency joint state estimation.
acetele/hardware/     Bus infrastructure, device adapters, operator inputs, and simulators.
acetele/runtime/      Preflight, lifecycle, safety state machine, and teleop sessions.
acetele/tools/        Static preflight, hardware calibration, and terminal UI tools.
ros2/                 First-party ROS 2 packages; never included in the core wheel.
zeromq/               Independent ZeroMQ adapter and its native XRCE companion component.
apps/ace_operator_ui/ Shared transport-neutral Qt operator window.
third_party/          PX4 messages and RealSense ROS submodules.
tests/                Unit, architecture, packaging, runtime, and ROS adapter tests.
```

Do not edit vendored areas unless the user explicitly asks:

```text
third_party/realsense_ros/
third_party/px4_msgs/
```

## Architecture

The dependency direction is fixed:

```text
core <- specification <- config/model/control/hardware <- runtime <- ROS 2 and ZeroMQ adapters
```

- `core` must not import configuration, hardware, runtime, or ROS packages.
- `Backend` belongs to `specification`; joint-angle transforms belong to `model.joint_angle`.
- `hardware` must not import ROS or deployment packages.
- ROS nodes compose a `RobotRuntime`; they do not inherit from robot or hardware classes.
- Every physical port has exactly one `BusActor` owner.
- Runtime imports only `BusAdapter` and immutable `AdapterPlan`; concrete FEETECH, FashionStar,
  Linker, and mock modules stay below the adapter registry.
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

Safe static preflight through the unified TUI does not open hardware:

```bash
python -m acetele.tools.tui
```

Preflight must validate URDF order and limits, exact profiles, firmware capabilities, unique port
ownership, and worst-case bus utilization. Utilization above 70% is a configuration error.

The safety state machine is:

```text
DISCONNECTED -> SAFE_DISABLED -> READY -> ACTIVE -> HOLD/FAULT
```

- Commands older than their deadline or from an old generation are discarded.
- Follower applications always seed the measured pose, enable torque, and enter `HOLD` after
  connection; transport adapters must not expose a runtime bypass for this startup policy.
- Lost follower heartbeats enter `HOLD`, clear pending motion, and require synchronization again.
- A Leader with a parallel gripper automatically powers arm alignment after healthy Follower
  feedback arrives; one release-to-close gripper gesture then enters tracking. A triggerless Leader
  still requires explicit alignment and tracking commands.
- End-effector-only commands never establish or refresh the arm heartbeat. ROS command streams use
  finite lifespan and latest-value delivery.
- Each bus actor independently enforces the admitted-command heartbeat. Repeated motion-write
  failures or sustained fast-state loss clear pending motion, attempt to hold the latest successful
  command, and latch an actor fault.
- Actor P95/P99 motion latency is measured through successful protocol-write completion; mailbox
  admission latency must not be reported as hardware latency.
- Emergency stop clears motion, increments generation, executes the strongest supported hardware
  action, and remains latched until explicit reset.
- Stale state, repeated protocol failures, device reset, or a missing arm joint enters `FAULT`.
- `FAULT` carries an explicit containment action: communication/state-trust failures hold the last
  successful target, device hardware alarms request protocol-level disable, and unsupported disable
  requires the external emergency stop. Explicit emergency stop always requests the strongest action.
- Devices without verifiable torque disable require an independent physical emergency stop. Every
  affected physical bus spec must declare that external stop.
- An in-process actor cannot protect against process termination, host power loss, or kernel failure.

FEETECH packet home calibration is exposed only through the TUI and backed by
`acetele.runtime.calibrate_feetech_home(spec)`. Keep calibration profile-specific, require
`SAFE_DISABLED`, validate the complete plan before connecting hardware, and preserve queue barriers
and bounded cleanup. Other protocols must not reuse the FEETECH procedure without their own
documented safety transaction.

Interactive operators should use `python -m acetele.tools.tui`. After explicit Enter confirmation,
its launch workflows must leave curses and execute ROS 2 or ZeroMQ with an argv sequence, never
through a shell. Its calibration workflow must display the complete immutable write plan, require
explicit Enter confirmation, leave hardware configuration read-only, and restore the terminal
before connecting a bus. Keep curses rendering separate from discovery, preflight, persistence,
and hardware execution so safety behavior remains testable without a terminal.

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

## ZeroMQ

`zeromq/ace_robot_zmq` is a separately packaged adapter and may depend on `pyzmq` and `msgpack`; the
core ACETele package may not. Keep its direct two-port PUB/SUB protocol versioned, bounded to one
MessagePack frame, latest-value only, and strict about names, units, finite values, session IDs, and
monotonically increasing sequence numbers. Remote wall-clock timestamps are diagnostic only.

The Follower's local receipt time drives a 500 ms remote-session lease. Its periodic local state loop
reissues the latest valid target and owns the shorter bus watchdog, so ordinary network jitter cannot
cause HOLD/re-enable cycles. New peer sessions call `reset_peer()`, invalidate old generations, enter
HOLD, and require synchronization again. CURVE
mode must authenticate the exact configured peer key; plaintext mode is only for trusted wired
networks. The ZMQ Follower does not use ROS 2, but it must publish measured arm-only state to PX4 via
the pinned native Agent/sidecar in `zeromq/ace_robot_zmq/xrce/`; that process lifecycle must complete before hardware
connection and any publication failure must force HOLD. `PoseLeaderClient` is the supported VR
integration surface and must reuse the same follower session and Cartesian safety path.

Keep ZMQ control (`5555/5556`) and image transport (`5562`) separate. Camera capture and the Qt
window run outside the control process and must never refresh the control heartbeat. Their failure is
degraded operation, while the arm-state XRCE Publisher remains safety-critical. Remote color and depth
previews are compressed and are not persisted. When CURVE is configured, the image endpoint must use
the same exact-peer credentials without plaintext fallback.

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
python -m compileall acetele zeromq/ace_robot_zmq
git diff --check
pre-commit run --all-files
```

The pytest configuration excludes `third_party/realsense_ros` because it is a large third-party
submodule.

## Packaging

Keep metadata in `pyproject.toml` and `setup.py` aligned. Python support is `>=3.10`. Package data
includes:

```text
acetele.config/presets/ace_leader/*.toml
acetele.config/presets/ace_follower/*.toml
acetele.model.robots.ace_follower/description/*.urdf
acetele.model.robots.ace_follower/description/meshes/*.STL
acetele.model.robots.ace_leader/description/*.urdf
acetele.model.robots.ace_leader/description/*.xml
acetele.model.robots.ace_leader/description/meshes/*.STL
```

When installed resources change, update both packaging files and
`tests/test_packaging_metadata.py`. Never package `ros2/`, `zeromq/`, or `third_party/` in the core
wheel. The ZeroMQ wheel must contain only `ace_robot_zmq` and its metadata. Do not restore
`acetele/equipment`, `acetele/robot`, `acetele/deploy`, or `acetele/utils`; those paths belong to the
removed architecture.

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
