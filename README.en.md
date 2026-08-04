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
  Let intent cross distance and become motion.
  <br />
  <a href="README.md">简体中文</a>
</p>

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#recommended-entry-point">Recommended Entry Point</a></li>
    <li><a href="#hardware-setup-and-calibration">Hardware Setup And Calibration</a></li>
    <li><a href="#teleoperation">Teleoperation</a></li>
    <li><a href="#zeromq-and-px4">ZeroMQ And PX4</a></li>
    <li><a href="#custom-configuration">Custom Configuration</a></li>
    <li><a href="#python-api">Python API</a></li>
    <li><a href="#development-checks">Development Checks</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Overview

ACETele brings robot specifications, kinematics, control, safety state, and hardware buses under one `RobotRuntime`. Supported devices include FEETECH HLS TTL, FEETECH SMS/SM RS485, FEETECH Modbus-RTU, FashionStar RS485, and Linker Hand RS485.

> [!WARNING]
> Real hardware requires an independent emergency stop. Software timeouts, torque
> disable commands, and bus diagnostics do not replace a power-disconnect circuit.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

Python 3.10 or newer is required. ROS 2 Humble is required only for the ROS 2 adapter.

```bash
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For SSH, replace the clone URL with:

```text
git@github.com:Xiangyuan-Xie/ACETele.git
```

Submodules use relative URLs and follow the HTTPS or SSH transport of the parent clone.

### Optional ZeroMQ Components

```bash
python -m pip install -e apps/ace_operator_ui
python -m pip install -e zeromq/ace_robot_zmq

# Dual-RealSense image transport on the Follower
python -m pip install -e "zeromq/ace_robot_zmq[camera]"

# Image viewer on the Leader
python -m pip install -e "zeromq/ace_robot_zmq[visualization]"
```

### ROS 2 Workspace

Run `sudo rosdep init` once before using `rosdep` for the first time.

```bash
source /opt/ros/humble/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep
rosdep update
rosdep install --from-paths ros2 third_party/px4_msgs \
  third_party/realsense_ros --ignore-src -r -y

colcon build --symlink-install \
  --base-paths ros2 third_party/px4_msgs third_party/realsense_ros
source install/setup.bash
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Recommended Entry Point

```bash
python -m acetele.tools.tui
```

The TUI runs hardware-free preflight first, then executes the confirmed action:

| Workflow | Purpose |
| --- | --- |
| `Launch ROS 2 Robot` | Start a ROS 2 Leader or Follower |
| `Launch ZMQ Robot` | Start direct peer-to-peer ZeroMQ teleoperation |
| `Calibrate FEETECH Home` | Write the complete FEETECH Home offsets |

Packaged specifications are under `acetele/config/presets/`:

| Specification | Hardware |
| --- | --- |
| `ace_leader/feetech_hls_ttl.toml` | HLS TTL Leader |
| `ace_follower/feetech_hls_ttl.toml` | HLS TTL Follower |
| `ace_follower/feetech_sms_rs485.toml` | SMS/SM RS485 Follower |
| `ace_follower/fashionstar_rs485.toml` | FashionStar RS485 Follower |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Hardware Setup And Calibration

1. Verify power, emergency stop, serial port, servo IDs, models, directions, and
   mechanical limits.
2. Add the current user to the serial-port group, then log in again:

   ```bash
   sudo usermod -aG dialout "$USER"
   ```

3. Move every joint to the mechanical Home pose declared by the RobotSpec.
4. Open the TUI, choose `Calibrate FEETECH Home`, inspect the complete write plan,
   and confirm it.

Calibration writes nonvolatile offsets for every FEETECH arm and end-effector joint.
It does not apply to FashionStar, FEETECH Modbus, or Linker Hand devices.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Teleoperation

Open the TUI on both Leader and Follower hosts. Select the corresponding RobotSpec and
the same teleoperation mode on both sides.

Startup sequence:

1. The Follower reads its current position, powers the arm, and enters hold.
2. The Leader automatically aligns after receiving healthy Follower state.
3. After alignment, release the Leader gripper below `0.25`, then close it above `0.75`.
4. The session enters `TRACKING`, and the Leader arm switches to local effort assistance.

Brief network jitter replaces old targets with the latest one. A sustained disconnect
stops remote commands and enters `HOLD`; reconnection requires synchronization again.
Follower holding, Leader gravity assistance, and communication timeouts do not replace
the hardware emergency stop.

### Teleoperation Modes

| Mode | Behavior |
| --- | --- |
| `joint` | Default direct joint-position mapping |
| `ee_pose` | Leader FK, relative pose mapping, and Follower IK |

`ee_pose` uses a default translation scale of `2.0` and rotation scale of `1.0`.
The current 4-DOF arm can only follow the reachable projection of a full SE(3) target;
position takes priority over unreachable orientation.

### Manual ROS 2 Launch

The TUI is preferred. For scripted deployment, invoke the launch file directly:

```bash
# Follower
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/presets/ace_follower/feetech_hls_ttl.toml" \
  teleop_mode:=joint

# Leader
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/presets/ace_leader/feetech_hls_ttl.toml" \
  teleop_mode:=joint
```

For pose control, add the same arguments on both hosts:

```text
teleop_mode:=ee_pose translation_scale:=2.0 rotation_scale:=1.0
```

Explicit emergency-stop services:

```text
/ace_leader/emergency_stop
/ace_follower/emergency_stop
```

The Follower publishes 4 to 14 filtered measured arm positions and velocities to
`/fmu/in/arm_joint_state`. Grippers and dexterous hands are excluded from this message.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ZeroMQ And PX4

ZeroMQ uses Leader command port `5555` and Follower state port `5556`. The Follower
also starts the pinned XRCE Agent and native sidecar, which publish `ArmJointState` to
PX4 over UDP port `8888`.

Build the XRCE component before first use:

```bash
git submodule update --init --recursive
cmake -S zeromq/ace_robot_zmq/xrce -B build/ace_robot_zmq-xrce \
  -DACETELE_XRCE_PREFIX="$HOME/.local/lib/acetele/xrce-2.4.2"
cmake --build build/ace_robot_zmq-xrce --parallel
```

PX4 parameters:

```text
UXRCE_DDS_CFG=Ethernet
UXRCE_DDS_AG_IP=<Follower wired-network IP>
UXRCE_DDS_PRT=8888
UXRCE_DDS_DOM_ID=0
```

Plaintext ZeroMQ is for trusted wired networks only. On untrusted networks, generate
CURVE certificates with `ace-robot-zmq keygen` and configure each host with its local
secret key and the peer public key in the TUI.

### Optional Image Transport

Image transport is isolated from control. A camera or UI failure does not refresh the
robot heartbeat.

```bash
# Follower
python -m ace_robot_zmq cameras
python -m ace_robot_zmq camera \
  --front-serial FRONT_SERIAL --wrist-serial WRIST_SERIAL

# Leader
python -m ace_robot_zmq visualize --follower-host FOLLOWER_IP
```

Image transport uses TCP `5562` for JPEG color, Zstd depth, and camera calibration.
It does not record data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Custom Configuration

RobotSpec TOML files describe buses, arms, and joints. Joint names belong to URDF and
control; `servo_id` is only a bus address and is never used to infer kinematics.

```toml
[basic]
model = "ace_follower"
backend = "physical"

[buses.arm]
type = "feetech_packet"
port = "/dev/ttyUSB0"
baudrate = 1000000
cycle_hz = 100
physical_layer = "ttl"
family = "hls"
external_estop = true
allow_unverified_identity = true # Only after manually verifying the servo model

[arms.single]
bus = "arm"
tool_frame = "link_5"

[[arms.single.joints]]
name = "joint_1"
servo_id = 0
servo_model = "HL3915"
direction = 1
home_position_rad = 0.0
```

The example expands one joint only. A real specification must list the complete arm in
URDF order. Start by copying the closest packaged preset instead of an empty file.

Before opening a serial port, preflight validates TOML fields, URDF mappings, model
profiles, joint limits, and bus utilization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Python API

```python
from acetele.config import load_robot_spec
from acetele.runtime import RobotRuntime

runtime = RobotRuntime(load_robot_spec("robot.toml"))
runtime.connect()
try:
    state = runtime.read()
    print(state.joints)
finally:
    runtime.disconnect()
```

Constructing `RobotRuntime` performs static preflight only. `connect()` opens serial
ports and starts bus actors.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Development Checks

```bash
python -m pytest
python -m compileall acetele
pre-commit run --all-files
```

For ROS 2 changes, also run `colcon build` and `colcon test`. Real-hardware changes
should report the hardware model, configuration, test duration, and emergency-stop test.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

ACETele is licensed under the [Apache License 2.0](LICENSE). Third-party submodules
retain their own licenses.

## Contact

- Maintainer: Xiangyuan Xie
- Repository: <https://github.com/Xiangyuan-Xie/ACETele>

## Acknowledgments

- [ROS 2](https://docs.ros.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
