<a id="readme-top"></a>

<div align="center">

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-%3E%3D3.9-3776AB.svg" alt="Python >= 3.9" /></a>
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
│   ├── core/         Local Python entry points
│   ├── deploy/       ROS2 deployment packages for users
│   ├── equipment/    Hardware drivers
│   ├── robot/        Robot classes
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

- [Python 3.9 or newer](https://www.python.org/downloads/)
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

Confirm that Python is 3.9 or newer.

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

### Sanity Check

The default configuration `acetele/config/default.toml` points to `ace_leader.toml`, and `ace_leader`
uses the `mock` backend by default. After installation, run a no-hardware check first:

```bash
python -m acetele.core.make_robot
```

Press `Ctrl+C` once it starts printing mock states.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Python API

`make_robot()` is the unified Python creation entry point. It reads `ConfigLoader` configuration and
selects and instantiates the matching robot class from `(robot_type, backend)`.

```python
from acetele.core.make_robot import make_robot

robot = make_robot()
try:
    joint_pos, joint_vel, joint_tau = robot.act()
    print(joint_pos, joint_vel, joint_tau)
finally:
    robot.close()
```

To load a configuration explicitly:

```python
from pathlib import Path

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot

config = ConfigLoader(Path("acetele/config/ace_follower.toml"))
robot = make_robot(config)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Configuration System

Configuration entry file:

```toml
# acetele/config/default.toml
[basic]
config_file = "ace_leader.toml"
```

Robot configuration files:

- `acetele/config/ace_leader.toml`
- `acetele/config/ace_follower.toml`

Key fields:

| Field | Purpose |
| --- | --- |
| `basic.robot_type` | Robot configuration, currently supporting `ace_leader` and `ace_follower` |
| `basic.backend` | Runtime backend configuration, currently including `mock`, `default`, and `ros2` |
| `linker.single.port` | Serial port for arm servos |
| `linker.single.joint_ids` | Servo IDs for each arm joint |
| `linker.single.joint_signs` | Arm joint direction convention |
| `linker.single.home_poses` | Calibrated arm joint home positions |
| `linker.single.servo_types` | Servo model configuration, such as `HL3960`, `HL3950`, `HL3930`, and `HL3915` |
| `gripper.single` | Gripper configuration, including servo ID, port, direction, home pose, and gripper type |

Here, the `mock` backend is used for local API checks and no-hardware debugging; the `default`
backend directly accesses real FEETECH devices; the `ros2` backend is mainly used for robot interface
wrapping inside ROS2 nodes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### ROS 2 Deployment

ROS 2 packages are located in `acetele/deploy`:

| Package | Purpose |
| --- | --- |
| `ace_robot_ros2` | Starts a leader or follower robot node according to `config_path` |
| `acetele_bringup` | Provides leader/follower system-level startup and combined launch files |
| `data_collector_ros2` | Triggers rosbag data recording according to remote-control channel status |
| `joystick_ros2` | Converts joystick input to PX4 manual control input |
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
ros2 launch ace_robot_ros2 ace_robot.launch.py
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
```

System-level startup:

```bash
ros2 launch acetele_bringup leader_system.launch.py
ros2 launch acetele_bringup follower_system.launch.py
```

Common topics:

- `/ace_leader/arm/command`
- `/ace_follower/arm/state`
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
