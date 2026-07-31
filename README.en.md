<a id="readme-top"></a>

<div align="center">

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-%3E%3D3.10-3776AB.svg" alt="Python >= 3.10" /></a>
  <a href="https://docs.ros.org/en/humble/"><img src="https://img.shields.io/badge/ROS%202-Humble-22314E.svg" alt="ROS 2 Humble" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="Apache-2.0" /></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg" alt="pre-commit enabled" /></a>
</p>

<h1 align="center">ACETele</h1>

<p align="center">
  From teleoperation to autonomous robots.
  <br />
  <a href="README.md">简体中文</a>
</p>

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-introduction">Project Introduction</a>
      <ul>
        <li><a href="#tech-stack">Tech Stack</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#sanity-check">Sanity Check</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#python-api">Python API</a></li>
        <li><a href="#configuration-system">Configuration System</a></li>
        <li><a href="#ros-2-deployment">ROS 2 Deployment</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Project Introduction

ACETele is a Python/ROS2 engineering framework for robot teleoperation and data collection. It aims
to provide a unified development workflow from local validation to real-hardware deployment.

```text
ACETele/
├── acetele/
│   ├── config/       Robot configurations
│   ├── control/      Thread-free control and compensation pipelines
│   ├── core/         Vendor-neutral state and command contracts
│   ├── deploy/       ROS2 deployment packages
│   ├── hardware/     Serial Actors, protocols, profiles, and mocks
│   ├── model/        URDF assets and Pinocchio models
│   ├── runtime/      Robot assembly, lifecycle, and safety state machine
│   ├── tools/        Common tools
│   └── utils/
├── tests/
├── pyproject.toml
├── setup.py
├── LICENSE
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Tech Stack

- [Python 3](https://docs.python.org/3/)
- [ROS2 Humble](https://docs.ros.org/en/humble/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- [Python 3.10 or newer](https://www.python.org/downloads/)
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) (optional)

### Installation

The commands below use Ubuntu/Linux as the example environment. Windows and macOS users can reuse
the Python virtual environment steps, but ROS2, serial-port permissions, and hardware drivers need
platform-specific setup.

1. Install base tools:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
python3 --version
```

Confirm that Python is 3.10 or newer.

2. Clone the repository and initialize submodules:

```bash
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele
git submodule update --init --recursive
```

If the repository has already been cloned, run this from the project root to fill in or update
submodules:

```bash
git submodule update --init --recursive
```

3. Create an isolated Python environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

After opening a new terminal inside the project, run `source .venv/bin/activate` first.

4. Install development tools:

```bash
python -m pip install pytest pre-commit
pre-commit install
```

5. If real servos will be connected, confirm that the current user has serial-port permissions:

```bash
sudo usermod -aG dialout "$USER"
```

This usually requires logging out and back in. Before connecting hardware, re-check the serial port,
servo IDs, mechanical limits, power supply, and emergency stop.

6. To build ROS2 packages, install and source ROS2 Humble first, then prepare a workspace:

```bash
source /opt/ros/humble/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# If rosdep has not been initialized on this machine, run sudo rosdep init first.
# If it reports that rosdep already exists, skip that step.
rosdep update

ACETELE_ROOT="$(pwd)"
mkdir -p ~/ws_acetele_ros2/src
cp -r "${ACETELE_ROOT}/acetele/deploy/"* ~/ws_acetele_ros2/src/
cd ~/ws_acetele_ros2
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Hardware Check

The project ships physical HLS TTL configurations for both Leader and Follower, and the generic ROS 2
launch defaults to the Leader spec. Before first use, verify ports, servo IDs, models, and mechanical
state, then run the static preflight, which does not open the serial port:

```bash
python -m acetele.tools.check_robot_spec acetele/config/ace_leader/feetech_hls_ttl.toml
python -m acetele.tools.check_robot_spec acetele/config/ace_follower/feetech_hls_ttl.toml
python -m acetele.tools.check_robot_spec acetele/config/ace_follower/feetech_sms_rs485.toml
```

Passing preflight confirms configuration, URDF, capability, and bus-budget consistency only; it is
not a physical safety qualification. Without hardware, set `basic.backend` to `mock` in your own
configuration or run the test suite.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Python API

`RobotRuntime` is the only Python robot entry point. Construction performs static preflight only;
`connect()` creates serial ports and Actor threads, and `disconnect()` performs bounded cleanup.

```python
from acetele.config.spec_loader import load_robot_spec
from acetele.runtime import RobotRuntime

spec = load_robot_spec("acetele/config/ace_follower/feetech_sms_rs485.toml")
runtime = RobotRuntime(spec)
runtime.connect()
try:
    state = runtime.read()
    print(state.joints["single"].positions)
finally:
    runtime.disconnect()
```

`RobotState` stores immutable `JointState` values by assembly name. Physical servo positions and
velocities pass through the low-latency estimator with NIS and physical innovation gates.
`runtime.diagnostics()` exposes bus cycles, state age, command replacement, and estimator data
without exposing vendor registers or bus IDs to control code.

#### Multi-vendor RS485 Runtime

The new hardware runtime uses one Actor per physical port. It keeps the strict safety FIFO, latest
motion mailbox, periodic state reads, and slow telemetry under one serial owner. Protocol-level
adapters currently cover:

- FEETECH HLS packet TTL;
- FEETECH SMS/SM packet RS485;
- FEETECH Modbus-RTU RS485;
- FashionStar UART/RS485 packet;
- Generic Linker Hand RS485, with current profiles for O6, L6, L7, and L10 rather than one model-specific adapter.

New configurations explicitly declare buses, joints, and model profiles. Validate URDF mappings,
profiles, firmware capabilities, and bus utilization without opening a serial port:

```bash
python -m acetele.tools.check_robot_spec acetele/config/ace_follower/feetech_sms_rs485.toml
python -m acetele.tools.check_robot_spec acetele/config/ace_follower/fashionstar_rs485.toml
```

Each `port` may define only one bus. Arms and end effectors on the same serial port must share that
bus so the port always has exactly one Actor owner. Unknown TOML fields fail validation instead of
silently falling back to defaults.

The ROS 2 entry point accepts only the `buses + joints` schema and uses composed Leader/Follower
nodes that own a `RobotRuntime`. Without `config_path`, it starts the HLS TTL Leader spec:

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/ace_follower/feetech_sms_rs485.toml"
```

The HLS TTL configurations are `ace_leader/feetech_hls_ttl.toml` and
`ace_follower/feetech_hls_ttl.toml`. The generic launch defaults to the Leader configuration;
Follower and other hardware assemblies must be selected explicitly through `config_path`.

High-rate arm/gripper command and state topics use `BEST_EFFORT + KEEP_LAST(1)`, while sync topics
use `RELIABLE + KEEP_LAST(1)`. Valid follower commands enter the latest-value mailbox directly in
the subscription callback without waiting for another control timer. Parallel grippers retain
`/ace_leader/gripper/command` and `/ace_follower/gripper/state`; dexterous hands use the separate
`/ace_leader/end_effector/command` and `/ace_follower/end_effector/state` topics and are not treated
as the gripper synchronization trigger.

FashionStar and Linker Hand currently have official-frame and test-double verification. Production
use still requires model-specific hardware identity, disconnect HOLD, emergency-stop, and sustained
load tests. FEETECH packet, FashionStar, and Linker Hand paths cannot verify torque disable per
device, so their physical configurations must set `external_estop = true` and actually provide an
independent hardware emergency stop. This field declares a safety prerequisite; it does not replace
the hardware circuit.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Configuration System

The multi-vendor RS485 runtime uses the `buses + joints` schema. `joint.name` is the URDF/ROS 2
kinematic name, while `servo_id` is only a bus address; neither is derived from the other. Example:

```toml
[basic]
model = "ace_follower"
backend = "physical"
urdf_path = "../model/robots/ace_follower/description/ace_follower.urdf"

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
tool_frame = "link_5"

[[arms.single.joints]]
name = "joint_1"
servo_id = 1
servo_model = "SM8512BL"
direction = 1
home_position_rad = 0.0
```

Before startup, the runtime estimates line speed, frame length, responses, and turnaround intervals
for the worst case. It refuses utilization above 70%. HLS profiles do not have trustworthy public
model-register mappings, so configurations must select an exact model profile. The reported register
is read and recorded at connection time, but it is not compared with a guessed value. Add the
optional `expected_model_number` when a trustworthy value is available to enable strict identity
verification. A physical bus whose protocol cannot expose model identity must also set
`allow_unverified_identity = true`; preflight continues to report `verified_identity=false`. This is
an explicit acknowledgement of the protocol limitation, not software verification of the model.
FashionStar firmware is read separately and strictly compared with `firmware_version`.
Known HLS profiles estimate output torque using model-specific KT and no-load current values.
RS485 models without official parameters do not inherit constants from a similar model.

Key fields:

| Field | Purpose |
| --- | --- |
| `basic.model` | Robot model and packaged URDF name |
| `basic.backend` | `physical` or `mock` |
| `basic.urdf_path` | Optional explicit URDF path; packaged model fallback when omitted |
| `buses.<name>.type` | Vendor protocol and physical bus type |
| `buses.<name>.port` | Serial port owned by one Actor |
| `buses.<name>.cycle_hz` | Target bus cycle rate |
| `buses.<name>.external_estop` | Declares a real independent hardware stop; required for physical buses without verified disable |
| `buses.<name>.allow_unverified_identity` | Explicitly accepts a protocol without readable product identity after manual hardware verification |
| `arms.<name>.bus` | Bus used by the arm |
| `arms.<name>.tool_frame` | URDF TCP link used by Cartesian control; required by `ee_pose` mode |
| `arms.<name>.joints` | Joints declared in URDF order |
| `joint.name` | URDF/ROS 2 kinematic name |
| `joint.servo_id` | Vendor bus address |
| `joint.servo_model` | Model requiring an official profile |
| `joint.direction` | Joint direction, restricted to `-1` or `1` |
| `joint.home_position_rad` | Joint angle written when the servo is at its mechanical home pose |
| `joint.expected_model_number` | Optional uint16 model register value for strict connection-time verification |
| `joint.firmware_version` | Expected FashionStar firmware version, read and checked at connection time |

`mock` and `physical` use the same typed schema. Unknown fields, duplicate ports, unsupported models,
incorrect joint order, invalid limits, or bus utilization above 70% fail before hardware is opened.
After placing every FEETECH packet joint at its mechanical home pose and confirming that the robot
can remain still safely, run:

```bash
python -m acetele.tools.calibrate_feetech_home \
  acetele/config/ace_follower/feetech_hls_ttl.toml --yes
```

The command completes all static checks before connecting and writes nonvolatile offsets only while
the runtime is `SAFE_DISABLED`. It does not apply this procedure to FashionStar, FEETECH Modbus, or
Linker Hand devices.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### ROS 2 Deployment

ROS 2 packages are located in `acetele/deploy`:

| Package | Purpose |
| --- | --- |
| `ace_robot_ros2` | Starts a leader or follower robot node according to `config_path` |
| `data_collector_ros2` | Triggers rosbag data recording according to remote-control channel status |
| `visualization_ros2` | Displays RGB-D images, joint states, and topic runtime status |
| `px4_msgs` | PX4 message definition submodule |
| `realsense-ros` | RealSense camera ROS 2 driver submodule |

Build example:

```bash
ACETELE_ROOT="$(pwd)"
mkdir -p ~/ws_acetele_ros2/src
cd ~/ws_acetele_ros2/src
cp -r "${ACETELE_ROOT}/acetele/deploy/"* .
cd ..
colcon build
source install/setup.bash
```

Common launch commands:

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/ace_follower/feetech_hls_ttl.toml"
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
```

The default `teleop_mode:=joint` preserves joint-space teleoperation. Start both Leader and Follower
with the same mode to use end-effector pose teleoperation:

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/ace_follower/feetech_hls_ttl.toml" \
  teleop_mode:=ee_pose translation_scale:=2.0 rotation_scale:=1.0
```

Synchronization remains joint-based. Once `TRACKING` begins, the first `PoseStamped` captures the
Leader/Follower relative tool anchors without causing a command jump. Subsequent Leader translation
is mapped by `2.0`, while rotation is mapped by `1.0`. A four-DOF arm cannot reproduce an arbitrary
six-dimensional SE(3) pose, so inverse kinematics tracks reachable translation first and reduces
orientation error in the remaining nullspace. URDF limits, command deadlines, synchronization
heartbeats, and the bus safety state machine remain active.

The generic `/ace_teleop/arm/ee_pose/command` input uses `geometry_msgs/PoseStamped` with
`BEST_EFFORT + KEEP_LAST(1)`. The Follower publishes its measured TCP pose on
`/ace_follower/arm/ee_pose/state`. A future VR source only needs to become the sole active publisher
of the command topic and provide a stable, non-empty reference frame; Follower mapping and IK do not
change.

Common topics:

- `/ace_leader/arm/command`
- `/ace_teleop/arm/ee_pose/command`
- `/ace_follower/arm/state`
- `/ace_follower/arm/ee_pose/state`
- `/ace_leader/gripper/command`
- `/ace_follower/gripper/state`
- `/ace_leader/arm/sync_mode`
- `/ace_follower/arm/sync_status`
- `/fmu/in/arm_joint_state`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Issues, feature suggestions, and pull requests are welcome. To make maintenance and code review
easier, each contribution should focus on one clearly bounded change, such as hardware drivers, robot
wrappers, ROS 2 nodes, data collection workflows, configuration files, test fixes, or documentation
updates. Please avoid mixing multiple unrelated changes in the same branch or PR.

### Branches And Commits

Create a feature branch from the latest main branch:

```bash
git checkout main
git pull
git checkout -b feat/your-feature-name
```

Use concise and clear commit messages, for example:

```text
feat(robot): add follower ros2 backend
fix(gripper): correct home pose calibration
docs: update teleoperation guide
test: add mock robot tests
```

### Local Checks

Before submitting a PR, run at least the following checks:

```bash
python -m pytest
pre-commit run --all-files
```

If the change touches ROS 2 packages, launch files, or message interfaces, also build and test the
workspace:

```bash
colcon build --symlink-install
colcon test
```

If the change depends on real hardware, such as FEETECH servos, grippers, joysticks, RealSense, PX4,
or external odometry, please provide the corresponding verification notes. If it cannot be reproduced
without hardware, describe which mocks, logs, or minimal examples were used for validation.

### Submodule Changes

If you need to modify `px4_msgs/`, `realsense-ros/`, or another submodule, make and commit the change
inside the corresponding submodule first:

```bash
cd px4_msgs
git checkout -b feat/your-change
git add .
git commit -m "feat: your change"
```

Then return to the ACETele parent repository and update and commit the corresponding gitlink:

```bash
cd ..
git add px4_msgs
git commit -m "chore: update px4_msgs submodule"
```

Do not commit only the uncommitted working-tree state of a submodule from the parent repository;
otherwise other users cannot reproduce the change.

### Pull Request

When submitting a PR, briefly describe:

* The purpose and background of the change;
* The main files, packages, or modules touched;
* The test, build, or verification commands that were run;
* Whether it depends on real hardware, external ROS 2 packages, or specific configuration;
* Whether it changes submodules, topic interfaces, or data formats.

This helps maintainers understand, reproduce, and merge your contribution more quickly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

This project uses the Apache 2.0 open-source license. See [LICENCE](LICENSE) for details. Submodules
and third-party components used by the project follow the licenses declared in their own repositories.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Project maintainer: Xiangyuan Xie

Project link: <https://github.com/Xiangyuan-Xie/ACETele>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [ROS2 Humble](https://docs.ros.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
