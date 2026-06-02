<a id="readme-top"></a>

<div align="center">

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-%3E%3D3.9-blue.svg" alt="Python" /></a>
  <a href="https://docs.ros.org/en/humble/"><img src="https://img.shields.io/badge/ROS%202-Humble-22314E.svg" alt="ROS 2 Humble" /></a>
  <a href="#project-status"><img src="https://img.shields.io/badge/status-experimental-orange.svg" alt="Status: experimental" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0" /></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit" /></a>
</p>

<h1 align="center">ACETele</h1>

---

<p align="center">
  A real-time teleoperation system for robotic platforms.
</p>

<p align="center">
  <a href="README.md">简体中文</a>
  ·
  <strong>English</strong>
</p>

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#project-status">Project Status</a></li>
    <li><a href="#tech-stack">Tech Stack</a></li>
    <li>
      <a href="#quick-start">Quick Start</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#python-entry">Python Entry</a></li>
        <li><a href="#configuration">Configuration</a></li>
        <li><a href="#ros-2-deployment">ROS 2 Deployment</a></li>
        <li><a href="#data-and-diagnostic-tools">Data and Diagnostic Tools</a></li>
      </ul>
    </li>
    <li><a href="#project-layout">Project Layout</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Project Overview

ACETele is a real-time teleoperation system for robotic platforms. It brings leader robots, follower robots, grippers, joysticks, ROS 2 nodes, data collection, and hardware diagnostics into one project. The goal is to make reusable, extensible, hardware-ready teleoperation pipelines easy to build in the lab.

Core capabilities:

- Supports two robot roles: `ace_leader` and `ace_follower`.
- Supports three backends: `mock`, `default`, and `ros2`, making it easy to switch among no-hardware tests, direct hardware control, and ROS 2 deployment.
- Wraps the FEETECH HLS servo driver, arm `Linker`, normalized `Gripper`, and joystick input.
- Supports synchronized leader-follower teleoperation, keeps arm and gripper on separate ROS 2 topics, and preserves the 5D joint state required by the PX4 adapter.
- Provides rosbag-to-HDF5 conversion, HDF5 visualization, backlash diagnostics, gravity compensation diagnostics, and FEETECH PID autotuning tools.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Status

ACETele is currently experimental. It mainly serves research, course experiments, and robotics system prototyping. APIs, configuration fields, ROS 2 topics, and hardware tuning workflows may still change as experiments evolve.

Before running on real hardware, confirm serial-port configuration, servo IDs, mechanical limits, power, emergency stop, and personnel safety. Commands related to calibration, PID tuning, gravity compensation, and diagnostics may move servos or write control parameters.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Tech Stack

Main runtime and integration dependencies:

- [Python](https://www.python.org/) 3.9+
- [NumPy](https://numpy.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/)
- [PX4](https://px4.io/) / `px4_msgs`
- [Intel RealSense ROS](https://github.com/IntelRealSense/realsense-ros)
- [pygame](https://www.pygame.org/), [pyserial](https://pyserial.readthedocs.io/), [h5py](https://www.h5py.org/)

Base Python dependencies are declared in `pyproject.toml` and `requirements.txt`; ROS 2, camera, PX4, and GUI dependencies should be prepared as needed in the ROS 2 workspace.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Quick Start

### Prerequisites

- Python 3.9 or newer.
- Git and pip.
- Ubuntu + ROS 2 Humble are recommended for ROS 2 deployment.
- Real hardware requires FEETECH HLS servos, serial-port permissions, and a complete safety-check workflow.
- Optional devices include a JDK FPV or pygame-compatible joystick, RealSense cameras, and PX4-related ROS 2 messages.

### Installation

Clone the repository:

```bash
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele
```

If the repository has already been cloned, initialize submodules with:

```bash
git submodule update --init --recursive
```

Install the Python package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install development tools and run tests:

```bash
python -m pip install pytest pre-commit
pre-commit install
python -m pytest
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Python Entry

The minimal entry point is `make_robot()`. By default, `acetele/config/default.toml` points to `ace_leader.toml`, where `ace_leader` currently uses the `mock` backend for quick no-hardware checks.

```python
from acetele.core.make_robot import make_robot

robot = make_robot()
try:
    joint_pos, joint_vel, joint_tau = robot.act()
    print(joint_pos, joint_vel, joint_tau)
finally:
    robot.close()
```

`BaseRobot.act()` returns position, velocity, and force/torque estimates. In the direct Robot API with a gripper, combined joint order is arm joints first, then gripper joints.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Configuration

The configuration entry point is `acetele/config/default.toml`:

```toml
[basic]
config_file = "ace_leader.toml"
```

Robot configuration files:

- `acetele/config/ace_leader.toml`
- `acetele/config/ace_follower.toml`

Key fields:

- `basic.robot_type`: robot role, such as `ace_leader` or `ace_follower`.
- `basic.backend`: runtime backend, such as `mock`, `default`, or `ros2`.
- `linker.single.port`: serial port for arm servos.
- `linker.single.joint_ids`: arm servo IDs.
- `linker.single.joint_signs`: joint direction convention.
- `linker.single.home_poses`: calibrated home positions.
- `linker.single.servo_types`: servo models, currently `HL3950`, `HL3930`, and `HL3915`.
- `gripper.single`: optional gripper configuration, including gripper servo ID, port, direction, initial position, and gripper type.

`ConfigLoader` creates concrete robot classes from the internal mapping by `(robot_type, backend)`. When adding a robot role or backend, update configuration, mapping, and tests together.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### ROS 2 Deployment

ROS 2 packages live under `acetele/deploy`:

- `ace_robot_ros2`: unified robot node that creates a leader or follower from `config_path`.
- `acetele_bringup`: system-level launch files.
- `data_collector_ros2`: records rosbags triggered by the remote-control channel.
- `joystick_ros2`: converts joystick input to PX4 manual control input.
- `visualization_ros2`: RGB-D image, joint-state, and topic-status visualization GUI.
- `px4_msgs`, `realsense-ros`: deployment message packages and camera submodule.

Build the core ROS 2 packages:

```bash
ACETELE_ROOT="$(pwd)"
mkdir -p ~/ws_acetele_ros2/src
cd ~/ws_acetele_ros2/src
cp -r "${ACETELE_ROOT}/acetele/deploy/"* .
cd ..
colcon build --packages-up-to \
  ace_robot_ros2 \
  data_collector_ros2 \
  joystick_ros2 \
  visualization_ros2
source install/setup.bash
```

`visualization_ros2` depends on `realsense2_camera_msgs`, `cv_bridge`, OpenCV, and PySide6. Before using `acetele_bringup`, make sure external dependencies such as `realsense2_camera` and `ros2_px4_odometry` are available in the workspace.

Common commands:

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="${ACETELE_ROOT}/acetele/config/ace_follower.toml"

ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
ros2 run joystick_ros2 manual_control
```

System-level bringup:

```bash
ros2 launch acetele_bringup leader_system.launch.py
ros2 launch acetele_bringup follower_system.launch.py
```

Main topics:

- `/ace_leader/arm/command`
- `/ace_follower/arm/state`
- `/ace_leader/gripper/command`
- `/ace_follower/gripper/state`
- `/ace_leader/arm/sync_mode`
- `/ace_follower/arm/sync_status`
- `/ace_follower/arm/external_joint_torque`
- `/ace_follower/arm/external_wrench`
- `/fmu/in/arm_joint_state`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Data and Diagnostic Tools

Convert rosbag to HDF5:

```bash
python -m acetele.tools.bag2hdf5 /path/to/rosbag /tmp/session.hdf5 \
  --sync-topic /ace_leader/arm/command
```

Visualize HDF5 data:

```bash
python -m acetele.tools.hdf5_viewer /tmp/session.hdf5 --stride 2
```

Hardware calibration and diagnostics:

```bash
python -m acetele.core.calibrate
python -m acetele.tools.backlash_diagnostics hold-error --target 0,0,0,0
python -m acetele.tools.gravity_compensation_diagnostics auto-calibrate
python -m acetele.tools.feetech_pid_autotune --ids 0,1,2,3
```

These hardware commands interact with real servos or control parameters. Make sure the robot is in a safe state before running them.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Layout

```text
ACETele/
├── acetele/
│   ├── config/          TOML configuration entry point and robot configs
│   ├── core/            Robot creation and calibration entry points
│   ├── deploy/          ROS 2 deployment packages and third-party message/camera submodules
│   ├── equipment/       Hardware abstractions, FEETECH driver, gripper, and joystick
│   ├── robot/           ace_leader / ace_follower robot implementations
│   ├── tools/           Data conversion, visualization, and hardware diagnostics
│   └── utils/           Teleoperation sync, gripper conversion, and external-force helpers
├── tests/               Unit tests and ROS 2 behavior tests
├── pyproject.toml       Build configuration and package metadata
├── setup.py             Legacy setuptools entry point
├── LICENSE              Apache-2.0 license
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Python package robot creation entry point and TOML configuration system.
- [x] FEETECH arm, gripper, gravity compensation, and external-force estimation.
- [x] Leader-follower synchronized teleoperation state machine.
- [x] ROS 2 robot node, data-collection node, joystick node, and visualization node.
- [x] Split arm topics and gripper topics while preserving PX4 5D state adaptation.
- [ ] Improve real-hardware deployment docs, including serial-port permissions, calibration, safety checks, and fault recovery.
- [ ] Add examples for dataset format, diagnostic tools, and ROS 2 bringup.
- [ ] Establish a stable release strategy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Issues and pull requests are welcome, especially for:

- Fixes to hardware drivers, ROS 2 nodes, or configuration.
- New robot, gripper, servo, or joystick adapters.
- Real-hardware deployment notes, data-collection workflows, and diagnostic examples.
- Better test coverage, documentation, and tooling.

Recommended flow:

```bash
git checkout -b feature/my-feature
python -m pytest
pre-commit run --all-files
git commit -m "feat: add my feature"
```

When opening a PR, include the motivation, scope, test results, and whether real hardware or external ROS 2 dependencies are required.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

This project is licensed under the [Apache License 2.0](LICENSE). Apache-2.0 is a good fit because it matches the license declared by the main ROS 2 subpackages in this repository, is permissive, and includes an explicit patent grant suitable for robotics software/hardware integration projects.

Third-party submodules and external dependencies keep their own licenses. For example, `acetele/deploy/px4_msgs` uses BSD 3-Clause, and `acetele/deploy/realsense-ros` keeps its upstream license. Follow the corresponding third-party licenses when reusing or redistributing the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Maintainer: Xiangyuan Xie

- Email: <dragonboat_xxy@163.com>
- Project Link: <https://github.com/Xiangyuan-Xie/ACETele>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [ROS 2](https://docs.ros.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
- [PX4](https://px4.io/)
- [Intel RealSense ROS](https://github.com/IntelRealSense/realsense-ros)
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- FEETECH servo SDK and the open-source robotics community

<p align="right">(<a href="#readme-top">back to top</a>)</p>
