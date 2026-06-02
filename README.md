<a id="readme-top"></a>

<div align="center">

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-%3E%3D3.9-blue.svg" alt="Python" /></a>
  <a href="https://docs.ros.org/en/humble/"><img src="https://img.shields.io/badge/ROS%202-Humble-22314E.svg" alt="ROS 2 Humble" /></a>
  <a href="#项目状态"><img src="https://img.shields.io/badge/status-experimental-orange.svg" alt="Status: experimental" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0" /></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit" /></a>
</p>

<h1 align="center">ACETele</h1>

---

<p align="center">
  面向机器人平台的实时遥操作系统。
</p>

<p align="center">
  <strong>简体中文</strong>
  ·
  <a href="README.en.md">English</a>
</p>

</div>

<details>
  <summary>目录</summary>
  <ol>
    <li><a href="#项目简介">项目简介</a></li>
    <li><a href="#项目状态">项目状态</a></li>
    <li><a href="#技术栈">技术栈</a></li>
    <li>
      <a href="#快速开始">快速开始</a>
      <ul>
        <li><a href="#环境要求">环境要求</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li>
      <a href="#使用">使用</a>
      <ul>
        <li><a href="#python-入口">Python 入口</a></li>
        <li><a href="#配置">配置</a></li>
        <li><a href="#ros-2-部署">ROS 2 部署</a></li>
        <li><a href="#数据与诊断工具">数据与诊断工具</a></li>
      </ul>
    </li>
    <li><a href="#目录结构">目录结构</a></li>
    <li><a href="#路线图">路线图</a></li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>

## 项目简介

ACETele 是一套面向机器人平台的实时遥操作系统。它把主端机器人、从端机器人、夹爪、手柄、ROS 2 节点、数据采集和硬件诊断放在统一工程中，目标是在实验室快速搭建可复用、可扩展、可接入真实硬件的遥操作链路。

核心能力：

- 支持 `ace_leader` 与 `ace_follower` 两类机器人角色。
- 支持 `mock`、`default`、`ros2` 三类后端，便于在无硬件测试、真实设备控制和 ROS 2 部署之间切换。
- 封装 FEETECH HLS 舵机驱动、机械臂 `Linker`、归一化夹爪 `Gripper` 和手柄输入。
- 支持主从同步遥操作，机械臂与夹爪使用独立 ROS 2 topic，并保留 PX4 适配所需的 5D 关节状态。
- 提供 rosbag 转 HDF5、HDF5 可视化、回差诊断、重力补偿诊断和 FEETECH PID 自动调参等工具。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 项目状态

ACETele 目前处于实验阶段，主要服务于科研、课程实验和机器人系统原型验证。接口、配置字段、ROS 2 topic 和硬件调参流程仍可能随实验需求变化。

在真实硬件上运行前，请确认串口配置、舵机 ID、机械限位、供电、急停和人员安全。涉及标定、PID 调参、重力补偿和诊断工具的命令都可能移动舵机或写入控制参数。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 技术栈

主要运行时与集成依赖：

- [Python](https://www.python.org/) 3.9+
- [NumPy](https://numpy.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/)
- [PX4](https://px4.io/) / `px4_msgs`
- [Intel RealSense ROS](https://github.com/IntelRealSense/realsense-ros)
- [pygame](https://www.pygame.org/)、[pyserial](https://pyserial.readthedocs.io/)、[h5py](https://www.h5py.org/)

Python 包的基础依赖由 `pyproject.toml` 和 `requirements.txt` 声明；ROS 2、相机、PX4 和 GUI 相关依赖需要在 ROS 2 工作空间中按需准备。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 快速开始

### 环境要求

- Python 3.9 或更新版本。
- Git 与 pip。
- 如需 ROS 2 部署，推荐 Ubuntu + ROS 2 Humble。
- 如需真实硬件，准备 FEETECH HLS 舵机、串口权限和完整安全检查流程。
- 可选外设包括 JDK FPV/pygame 兼容手柄、RealSense 相机和 PX4 相关 ROS 2 消息。

### 安装

克隆仓库：

```bash
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele
```

如果已经克隆过仓库，可补拉子模块：

```bash
git submodule update --init --recursive
```

安装 Python 包：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

安装开发工具并运行测试：

```bash
python -m pip install pytest pre-commit
pre-commit install
python -m pytest
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用

### Python 入口

最小调用入口是 `make_robot()`。默认 `acetele/config/default.toml` 指向 `ace_leader.toml`，其中 `ace_leader` 当前使用 `mock` 后端，可用于无硬件快速验证。

```python
from acetele.core.make_robot import make_robot

robot = make_robot()
try:
    joint_pos, joint_vel, joint_tau = robot.act()
    print(joint_pos, joint_vel, joint_tau)
finally:
    robot.close()
```

`BaseRobot.act()` 返回位置、速度和力/力矩估计。带夹爪的 direct Robot API 中，组合关节顺序为机械臂关节在前，夹爪关节在后。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### 配置

配置入口位于 `acetele/config/default.toml`：

```toml
[basic]
config_file = "ace_leader.toml"
```

机器人配置文件位于：

- `acetele/config/ace_leader.toml`
- `acetele/config/ace_follower.toml`

核心字段：

- `basic.robot_type`：机器人角色，例如 `ace_leader`、`ace_follower`。
- `basic.backend`：运行后端，例如 `mock`、`default`、`ros2`。
- `linker.single.port`：机械臂舵机串口。
- `linker.single.joint_ids`：机械臂舵机 ID。
- `linker.single.joint_signs`：关节方向约定。
- `linker.single.home_poses`：标定零位。
- `linker.single.servo_types`：舵机型号，目前支持 `HL3950`、`HL3930`、`HL3915`。
- `gripper.single`：可选夹爪配置，包含夹爪舵机 ID、串口、方向、初始位置和夹爪类型。

`ConfigLoader` 会根据 `(robot_type, backend)` 从内部映射表中创建对应机器人类。新增机器人角色或后端时，需要同步更新配置、映射和测试。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### ROS 2 部署

ROS 2 包位于 `acetele/deploy`：

- `ace_robot_ros2`：统一机器人节点，通过 `config_path` 创建 leader 或 follower。
- `acetele_bringup`：系统级 launch 文件。
- `data_collector_ros2`：由遥控通道触发 rosbag 录制。
- `joystick_ros2`：将手柄输入转换为 PX4 manual control 输入。
- `visualization_ros2`：RGB-D 图像、关节状态和 topic 状态可视化 GUI。
- `px4_msgs`、`realsense-ros`：部署相关消息包和相机子模块。

构建核心 ROS 2 包：

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

`visualization_ros2` 依赖 `realsense2_camera_msgs`、`cv_bridge`、OpenCV 和 PySide6。使用 `acetele_bringup` 前，请确认 `realsense2_camera`、`ros2_px4_odometry` 等外部依赖已经在工作空间中可用。

常用命令：

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="${ACETELE_ROOT}/acetele/config/ace_follower.toml"

ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
ros2 run joystick_ros2 manual_control
```

系统级启动：

```bash
ros2 launch acetele_bringup leader_system.launch.py
ros2 launch acetele_bringup follower_system.launch.py
```

主要 topic：

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

### 数据与诊断工具

rosbag 转 HDF5：

```bash
python -m acetele.tools.bag2hdf5 /path/to/rosbag /tmp/session.hdf5 \
  --sync-topic /ace_leader/arm/command
```

HDF5 可视化：

```bash
python -m acetele.tools.hdf5_viewer /tmp/session.hdf5 --stride 2
```

硬件标定与诊断：

```bash
python -m acetele.core.calibrate
python -m acetele.tools.backlash_diagnostics hold-error --target 0,0,0,0
python -m acetele.tools.gravity_compensation_diagnostics auto-calibrate
python -m acetele.tools.feetech_pid_autotune --ids 0,1,2,3
```

以上硬件命令会接触真实舵机或控制参数，运行前请确认机器人处于安全状态。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 目录结构

```text
ACETele/
├── acetele/
│   ├── config/          TOML 配置入口与机器人配置
│   ├── core/            机器人创建与标定入口
│   ├── deploy/          ROS 2 部署包与第三方消息/相机子模块
│   ├── equipment/       硬件抽象、FEETECH 驱动、夹爪和手柄
│   ├── robot/           ace_leader / ace_follower 机器人实现
│   ├── tools/           数据转换、可视化和硬件诊断工具
│   └── utils/           遥操作同步、夹爪转换和外力估计辅助逻辑
├── tests/               单元测试与 ROS 2 行为测试
├── pyproject.toml       构建配置与包元数据
├── setup.py             legacy setuptools 入口
├── LICENSE              Apache-2.0 许可证
└── README.md
```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 路线图

- [x] Python 包形式的机器人创建入口与 TOML 配置系统。
- [x] FEETECH 机械臂、夹爪、重力补偿和外力估计。
- [x] 主从同步遥操作状态机。
- [x] ROS 2 机器人节点、数据采集节点、手柄节点和可视化节点。
- [x] 机械臂 topic 与夹爪 topic 分离，同时保留 PX4 5D 状态适配。
- [ ] 完善真实硬件部署文档，包括串口权限、标定、安全检查和故障恢复。
- [ ] 补充数据集格式、诊断工具和 ROS 2 bringup 示例。
- [ ] 建立稳定版本发布策略。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 贡献

欢迎提交 issue 和 pull request，尤其是：

- 修复硬件驱动、ROS 2 节点或配置问题。
- 增加新的机器人、夹爪、舵机或手柄适配。
- 补充真实硬件部署经验、数据采集流程和诊断示例。
- 改进测试覆盖、文档和工具链。

推荐流程：

```bash
git checkout -b feature/my-feature
python -m pytest
pre-commit run --all-files
git commit -m "feat: add my feature"
```

提交 PR 时请说明改动背景、影响范围、测试结果，以及是否需要真实硬件或 ROS 2 外部依赖。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。选择 Apache-2.0 是因为它与仓库内主要 ROS 2 子包的 `package.xml` 保持一致，属于宽松许可证，并包含明确的专利授权条款，适合机器人软硬件集成项目。

第三方子模块与外部依赖保留各自许可证，例如 `acetele/deploy/px4_msgs` 使用 BSD 3-Clause，`acetele/deploy/realsense-ros` 使用其上游许可证。复用或再分发时请同时遵守相关第三方许可证。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 联系

维护者：Xiangyuan Xie

- Email: <dragonboat_xxy@163.com>
- Project Link: <https://github.com/Xiangyuan-Xie/ACETele>

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 致谢

- [ROS 2](https://docs.ros.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)
- [PX4](https://px4.io/)
- [Intel RealSense ROS](https://github.com/IntelRealSense/realsense-ros)
- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- FEETECH 舵机 SDK 与开源机器人社区

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>
