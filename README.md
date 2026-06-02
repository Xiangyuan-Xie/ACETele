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
  从遥操作开始，走向自主机器人。
  <br />
  <a href="README.en.md">English</a>
</p>

</div>

<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#项目简介">项目简介</a>
      <ul>
        <li><a href="技术栈">技术栈</a></li>
      </ul>
    </li>
    <li>
      <a href="#快速开始">快速开始</a>
      <ul>
        <li><a href="#环境要求">环境要求</a></li>
        <li><a href="#安装">安装</a></li>
        <li><a href="#快速自检">快速自检</a></li>
      </ul>
    </li>
    <li>
      <a href="#使用">使用</a>
      <ul>
        <li><a href="#python-api">Python API</a></li>
        <li><a href="#配置系统">配置系统</a></li>
        <li><a href="#ros-2-部署">ROS 2 部署</a></li>
      </ul>
    </li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>

## 项目简介

ACETele 是一套面向机器人遥操作与数据采集的 Python/ROS2 工程框架，旨在提供从本地验证到真实硬件部署的统一开发流程。

```text
ACETele/
├── acetele/
│   ├── config/       机器人配置
│   ├── core/         本地调用接口
│   ├── deploy/       用户ROS2部署的功能包
│   ├── equipment/    硬件驱动
│   ├── robot/        机器人类
│   ├── tools/        常用工具
│   └── utils/
├── tests/
├── pyproject.toml
├── setup.py
├── LICENSE
└── README.md
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### 技术栈

- [Python 3](https://docs.python.org/3/)
- [ROS2 Humble](https://docs.ros.org/en/humble/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 快速开始

### 环境要求

- [Python 3.9 及以上](https://www.python.org/downloads/)
- [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)（可选）

### 安装

以下命令以 Ubuntu/Linux 环境为例。Windows 或 macOS 用户可以复用 Python 虚拟环境步骤，
但 ROS2、串口权限和硬件驱动需要按对应平台单独准备。

1. 安装基础工具：

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
python3 --version
```

确认 Python 版本为 3.9 或更高。

2. 克隆仓库并初始化子模块：

```bash
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele
git submodule update --init --recursive
```

如果已经克隆过仓库，请在项目根目录补齐或更新子模块：

```bash
git submodule update --init --recursive
```

3. 创建隔离的 Python 环境并安装项目：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

之后每次打开新的终端进入项目时，先运行 `source .venv/bin/activate`。

4. 安装开发工具：

```bash
python -m pip install pytest pre-commit
pre-commit install
```

5. 如需连接真实舵机，确认当前用户有串口权限：

```bash
sudo usermod -aG dialout "$USER"
```

该命令通常需要注销并重新登录后生效。连接硬件前，请再次确认串口路径、舵机 ID、
机械限位、供电和急停状态。

6. 如需构建 ROS2 功能包，先安装并激活 ROS2 Humble，再准备工作空间：

```bash
source /opt/ros/humble/setup.bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# 如果本机尚未初始化 rosdep，先运行 sudo rosdep init；如果提示已存在可跳过。
rosdep update

ACETELE_ROOT="$(pwd)"
mkdir -p ~/ws_acetele_ros2/src
cp -r "${ACETELE_ROOT}/acetele/deploy/"* ~/ws_acetele_ros2/src/
cd ~/ws_acetele_ros2
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 快速自检

默认配置 `acetele/config/default.toml` 指向 `ace_leader.toml`，而 `ace_leader` 默认使用 `mock` 后端。
安装后可以先在无硬件环境中确认入口可用：

```bash
python -m acetele.core.make_robot
```

看到 mock 状态持续输出后，按 `Ctrl+C` 退出。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用

### Python API

`make_robot()` 是统一的 Python 创建入口，负责读取 `ConfigLoader` 配置，并根据 `(robot_type, backend)` 选择并实例化对应的机器人类。

```python
from acetele.core.make_robot import make_robot

robot = make_robot()
try:
    joint_pos, joint_vel, joint_tau = robot.act()
    print(joint_pos, joint_vel, joint_tau)
finally:
    robot.close()
```

如果需要显式加载配置：

```python
from pathlib import Path

from acetele.config.config_loader import ConfigLoader
from acetele.core.make_robot import make_robot

config = ConfigLoader(Path("acetele/config/ace_follower.toml"))
robot = make_robot(config)
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### 配置系统

配置入口文件：

```toml
# acetele/config/default.toml
[basic]
config_file = "ace_leader.toml"
```

机器人配置文件：

- `acetele/config/ace_leader.toml`
- `acetele/config/ace_follower.toml`

关键字段：

| 字段 | 作用 |
| --- | --- |
| `basic.robot_type` | 机器人配置，目前支持 `ace_leader` 和 `ace_follower` |
| `basic.backend` | 运行后端配置，目前包括 `mock`、`default`、`ros2` |
| `linker.single.port` | 机械臂舵机串口 |
| `linker.single.joint_ids` | 机械臂各关节对应的舵机ID |
| `linker.single.joint_signs` | 机械臂关节方向约定 |
| `linker.single.home_poses` | 标定后的机械臂关节Home位 |
| `linker.single.servo_types` | 舵机型号配置，例如 `HL3950`、`HL3930`、`HL3915` |
| `gripper.single` | 夹爪相关配置，包括舵机ID、串口、方向、Home位、夹爪类型和力控开关 |

其中，`mock` 后端用于本地 API 自检与无硬件调试；`default` 后端会直接访问真实 FEETECH 设备；`ros2` 后端主要用于 ROS2 节点内部的机器人接口封装。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### ROS 2 部署

ROS 2 包位于 `acetele/deploy`：

| 包 | 作用 |
| --- | --- |
| `ace_robot_ros2` | 根据 `config_path` 启动 leader 或 follower 机器人节点 |
| `acetele_bringup` | 提供 leader/follower 系统级启动与组合 launch 文件 |
| `data_collector_ros2` | data_collector_ros2	根据遥控通道状态触发 rosbag 数据录制 |
| `joystick_ros2` | 将手柄输入转换为 PX4 manual control 输入 |
| `visualization_ros2` | 用于显示 RGB-D 图像、关节状态和 topic 运行状态 |
| `px4_msgs` | PX4 消息定义子模块 |
| `realsense-ros` | RealSense 相机 ROS 2 驱动子模块 |

构建示例：

```bash
ACETELE_ROOT="$(pwd)"
mkdir -p ~/ws_acetele_ros2/src
cd ~/ws_acetele_ros2/src
cp -r "${ACETELE_ROOT}/acetele/deploy/"* .
cd ..
colcon build
source install/setup.bash
```

常用启动命令：

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
```

系统级启动：

```bash
ros2 launch acetele_bringup leader_system.launch.py
ros2 launch acetele_bringup follower_system.launch.py
```

常见 topic：

- `/ace_leader/arm/command`
- `/ace_follower/arm/state`
- `/ace_leader/gripper/command`
- `/ace_follower/gripper/state`
- `/ace_leader/arm/sync_mode`
- `/ace_follower/arm/sync_status`
- `/ace_follower/arm/external_joint_torque`
- `/ace_follower/arm/external_wrench`
- `/fmu/in/arm_joint_state`

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 贡献

欢迎提交 Issue、功能建议和 Pull Request。为便于维护和代码审查，建议每次贡献围绕一个边界清晰的改动展开，例如硬件驱动、机器人封装、ROS 2 节点、数据采集流程、配置文件、测试修复或文档更新等。请避免在同一个分支或 PR 中混合多个无关改动。

### 分支与提交

建议从最新的主分支创建功能分支：

```bash
git checkout main
git pull
git checkout -b feat/your-feature-name
```

提交信息建议使用简洁明确的格式，例如：

```text
feat(robot): add follower ros2 backend
fix(gripper): correct home pose calibration
docs: update teleoperation guide
test: add mock robot tests
```

### 本地检查

提交 PR 前，建议至少运行以下检查：

```bash
python -m pytest
pre-commit run --all-files
```

如果改动涉及 ROS 2 包、launch 文件或消息接口，建议同时进行工作空间构建和测试：

```bash
colcon build --symlink-install
colcon test
```

如果改动依赖真实硬件，例如 FEETECH 舵机、夹爪、手柄、RealSense、PX4 或外部 odometry，请尽量提供对应的验证说明；若无法在无硬件环境中复现，请说明已通过哪些 mock、日志或最小示例完成验证。

### 子模块修改

如果需要修改 `px4_msgs/`、`realsense-ros/` 或其他子模块，请先在对应子模块内完成修改并提交：

```bash
cd px4_msgs
git checkout -b feat/your-change
git add .
git commit -m "feat: your change"
```

随后回到 ACETele 父仓库，更新并提交对应的 gitlink：

```bash
cd ..
git add px4_msgs
git commit -m "chore: update px4_msgs submodule"
```

请不要只在父仓库中提交子模块目录的未提交工作区状态，否则其他用户无法复现该修改。

### Pull Request

提交 PR 时，请简要说明：

* 本次改动的目的和背景；
* 修改涉及的主要文件、包或模块；
* 已运行的测试、构建或验证命令；
* 是否依赖真实硬件、外部 ROS 2 包或特定配置；
* 是否涉及子模块、topic 接口或数据格式变化。

这样可以帮助维护者更快地理解、复现和合并你的贡献。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 许可证

本项目采用 Apache 2.0 开源许可证，详情请参见 [LICENCE](LICENSE)。项目中引用的子模块及第三方组件遵循其各自仓库声明的许可证。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 联系

项目维护者：Xiangyuan Xie

项目链接: <https://github.com/Xiangyuan-Xie/ACETele>

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 致谢

- [ROS2 Humble](https://docs.ros.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>
