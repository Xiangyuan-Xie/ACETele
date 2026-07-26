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

### 硬件自检

默认配置 `acetele/config/default.toml` 指向 `ace_leader.toml`。Leader 默认使用
`physical` 设备后端和 `standalone` 运行方式，会访问配置中的 `/dev/ttyUSB0`。
连接机械臂并确认串口名称和访问权限后运行：

```bash
python -m acetele.core.make_robot
```

看到真实关节状态持续输出后，按 `Ctrl+C` 退出。无硬件环境下请运行测试，或显式通过
`ConfigLoader(..., backend_override="mock")` 创建 Mock 后端，不要依赖项目默认值。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用

### Python API

`make_robot()` 是统一的 Python 创建入口，负责读取 `ConfigLoader` 配置，并根据
`(robot_type, runtime)` 选择机器人入口。`backend` 只决定使用真实设备还是 mock 设备。

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

`robot.name` 同时包含拓扑、设备后端和运行方式，例如
`ace_follower_physical_ros2` 或 `ace_leader_physical_standalone`。

物理 FEETECH 设备会保留驱动层的原始寄存器快照，并在设备层使用带异常观测门控的
低延迟常速度卡尔曼估计器。`robot.act()`、ROS2 和 FMU 接口使用滤波后的位置与速度；
`JointDeviceState.raw_positions` 仍保留未经滤波的编码器角度。机械臂和夹爪的
`get_state_estimator_diagnostics()` 可用于查看创新、NIS、观测门限、限速状态和累计拒绝次数。

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
| `basic.robot_type` | 机器人拓扑，支持 `ace_leader`、`ace_follower` 和 `ace_follower_dual` |
| `basic.backend` | 设备后端：`mock` 或 `physical` |
| `basic.runtime` | 运行入口：`standalone` 或 `ros2` |
| `arms.<name>.port` | 该机械臂的舵机串口 |
| `arms.<name>.joint_ids` | 机械臂各关节对应的舵机 ID |
| `arms.<name>.joint_names` | 必填；按顺序对应 URDF、Pinocchio 和 ROS2 的机械臂关节名 |
| `arms.<name>.joint_signs` | 机械臂关节方向约定 |
| `arms.<name>.home_poses` | 标定后的机械臂关节 Home 位 |
| `arms.<name>.servo_models` | 舵机型号，例如 `HL3960`、`HL3950`、`HL3930` 和 `HL3915` |
| `arms.<name>.end_effector` | 与该机械臂绑定的可选夹爪或灵巧手配置 |
| `arms.<name>.end_effector.joint_name` | FEETECH 夹爪必填；夹爪在运动学和 ROS2 接口中的关节名 |
| `travel_range_rad` | 归一化夹爪从 `0` 到 `1` 对应的实际舵机角行程；换算后须为 `1` 至 `2047` 个 FEETECH 位置计数 |

`mock` 用于本地 API 自检与无硬件调试，`physical` 会访问真实设备；`runtime = "ros2"`
选择 ROS2 节点封装，但不会隐式改变设备后端。机械臂和末端执行器在同一个装配表中绑定，
单臂使用 `arms.single`，双臂可使用 `arms.left` 和 `arms.right`。

TOML 不配置 Mock 专用的初始位置、限位或速度。Mock 机械臂使用 `home_poses` 作为初始状态；
存在 URDF 时从中读取关节限位，否则使用设备内部默认值。Mock 夹爪和灵巧手的模拟约束也由
对应设备实现提供。这样同一份机器人配置可以在 `mock` 与 `physical` 后端之间切换，而不会
混入仅对模拟器有效的参数。

`joint_ids` 仅表示舵机总线地址；每条机械臂都必须显式配置 `joint_names`，FEETECH 夹爪必须
显式配置 `joint_name`。这些名称用于 URDF、Pinocchio 和 ROS2 接口，重新分配舵机 ID 时无需
修改运动学关节名称。标定只接受 `backend = "physical"` 的配置，并应显式指定配置文件：

```bash
python -m acetele.core.calibrate --config acetele/config/ace_follower.toml
```

使用 `mock` 配置执行标定会在打开串口前直接失败。

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
