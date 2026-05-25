# ACETele

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) [![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#项目状态) [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

ACETele 是一套面向机器人平台的实时遥操作系统，致力于构建统一且稳定的控制与数据传输链路。

---

- [ACETele](#acetele)
  - [项目简介](#项目简介)
  - [功能特性](#功能特性)
  - [项目状态](#项目状态)
  - [目录结构](#目录结构)
  - [快速开始](#快速开始)
    - [运行环境](#运行环境)
    - [安装](#安装)
    - [获取遥操作数据](#获取遥操作数据)
  - [配置说明](#配置说明)
  - [ROS 2 集成与部署](#ros-2-集成与部署)
  - [与 ACESim 协作](#与-acesim-协作)
  - [开发与测试](#开发与测试)
  - [贡献指南](#贡献指南)
  - [开源协议](#开源协议)
  - [致谢](#致谢)

---

## 项目简介

ACETele 是一套用于机器人遥操作系统的软件框架。该系统通常包含本地操作端、远程执行端与真实或仿真环境三部分。我们希望通过清晰的架构与可扩展的设计，实现以下目标：

- 在迁移至真实硬件环境时最大限度复用同一套逻辑；
- 以统一接口接入多种硬件设备（舵机、电机、手柄等），降低集成成本；
- 与 ROS 2 等主流机器人软件生态保持良好兼容性，便于系统集成与扩展。

## 功能特性

- 遥操作
  - 支持单臂/多臂的同构遥操作。

- ROS 2 集成
  - `ace_robot_ros2`：机器人端 ROS 2 节点。
  - `data_collector_ros2`：数据采集与记录节点。
  - `joystick_ros2`：手柄输入与手动控制节点。
  - `visualization_ros2`：可视化与监控 GUI。
- 工具与数据处理
  - 训练数据采集与格式转换。

## 项目状态

ACETele 目前处于实验性（experimental）阶段，主要面向科研研究和教学演示场景。接口与内部实现仍可能根据需求调整，尚不承诺长期向后兼容。若用于工程化或产品级场景，建议在充分评估与测试基础上引入版本管理策略。

## 目录结构

```text
ACETele/
├─ acetele/
│  ├─ config/          配置文件
│  ├─ core/            核心接口
│  ├─ deploy/          ROS 2 部署包
│  ├─ equipment/       硬件抽象与驱动
│  ├─ robot/           主从遥操作机器人端逻辑
│  ├─ tools/           工具脚本
├─ README.md           本说明文档
├─ requirements.txt    运行时依赖
├─ pyproject.toml      项目元数据与构建配置
├─ setup.py            setuptools 安装脚本
└─ .pre-commit-config.yaml  开发阶段代码质量检查配置
```

## 快速开始

### 运行环境

- 操作系统：Ubuntu（推荐）/Windows。
- Python：3.10 及以上。
- ROS 2（可选）：Humble。

### 安装

```bash
git clone https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele

pip install -e .
```

### 获取遥操作数据

`BaseRobot` 及其子类负责将外部输入（如主臂当前关节位置）转换为目标动作指令。

配置入口见 [`acetele/config/default.toml`](acetele/config/default.toml)：

```toml
[basic]
config_file = "ace_leader.toml"
```

示例配置见 [`acetele/config/ace_leader.toml`](acetele/config/ace_leader.toml)：

```toml
[basic]
robot_type = "ace_leader"
backend = "mock"

[linker.single]
port = "/dev/ttyUSB0"
joint_ids = [0, 1, 2, 3]
joint_signs = [1, -1, -1, -1]
home_poses = [0.0, 0.0, 0.0, 0.0]
enable_gravity_compensation = true
enable_estimate_external_torque = false
servo_types = ["HL3915", "HL3915", "HL3915", "HL3915"]

[gripper.single]
port = "/dev/ttyUSB0"
joint_id = 4
joint_sign = 1
home_pose = 0.0
servo_type = "HL3915"
gripper_type = "ace_leader"
enable_fragile_force_control = false
```

可通过修改 `default.toml` 的 `config_file` 切换不同配置，例如切换为 `ace_follower.toml` 以使用 ace_follower 机器人的配置文件。

调用方式：

```python
from acetele.core.make_robot import make_robot

robot = make_robot()
joint_pos, joint_vel, joint_tau = robot.act()
robot.close()
```

## 配置说明

配置系统由 [`ConfigLoader`](acetele/config/config_loader.py) 统一管理，核心概念包括：

- `basic.robot_type`：机器人类型，例如 `"ace_leader"` 或 `"ace_follower"`；
- `basic.backend`：后端类型，例如 `"default"`、`"ros2"`、`"mock"`；
- `linker.single` / `linker.dual`：机械臂硬件连线配置，包含串口号、关节 ID、符号方向以及初始姿态等信息；
- `gripper.single` / `gripper.dual`：可选夹爪配置，使用 `joint_id` 和 `port` 描述夹爪舵机。direct Robot API 的组合关节顺序固定为 `linker.single.joint_ids` 后追加 `gripper.single.joint_id`。
- ROS 2 topic 已按设备分域：`/ace_follower/arm/state` 和 `/ace_leader/arm/command` 只包含机械臂轴，夹爪使用 `/ace_follower/gripper/state` 和 `/ace_leader/gripper/command` 单独发布；PX4 `/fmu/in/arm_joint_state` 仍是适配层要求的 5D `[arm..., gripper]`。

`ConfigLoader` 使用内部 `_ROBOT_MAP` 将 `(robot_type, backend)` 映射为具体模块与类名，并由 `make_robot` 工厂函数创建机器人实例。

## ROS 2 集成与部署

`acetele/deploy` 目录包含若干 ROS 2 包：

- `ace_robot_ros2/`：将 ACETele 机器人端封装为 ROS 2 节点。
- `data_collector_ros2/`：数据采集与记录节点。
- `joystick_ros2/`：手柄输入与手动控制节点。
- `visualization_ros2/`：可视化 GUI 与节点。

构建示例：

```bash
mkdir -p ~/ws_acetele_ros2/src
cd ~/ws_acetele_ros2/src
cp -r /your/path/to/ACETele/acetele/deploy/ .
cd ..
colcon build --packages-select ace_robot_ros2 data_collector_ros2 visualization_ros2 joystick_ros2
source install/setup.bash
```

启动示例：

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
ros2 launch joystick_ros2 joystick.launch.py
```

参数说明请参考各包的 `config/` 与 `launch/` 目录。

## 与 ACESim 协作

本仓库支持与 [ACESim](https://github.com/Xiangyuan-Xie/ACESim) 协作，借助 MuJoCo/Gazebo 等先进仿真环境进行高效数据采集与算法验证。

## 开发与测试

本仓库通过 [pre-commit](https://pre-commit.com/) 统一代码风格与基础质量检查。

```bash
pip install pre-commit
pre-commit install
```

提交前可执行：

```bash
pre-commit run --all-files
```

主要检查项：

- 代码格式化：`black`、`isort`。
- 代码质量：`flake8`。
- 类型检查：`mypy`。
- 其他基础检查：YAML 合法性、大文件检测、尾空格等。

## 贡献指南

欢迎参与改进，包括修复缺陷、补充文档、扩展硬件设备驱动或优化工具链。推荐流程如下：

1. Fork 本仓库并创建功能分支：

   ```bash
   git checkout -b feature/my-awesome-feature
   ```

2. 在本地实现与自测：
   - 确认与 ROS 2 等外部依赖能正常协同；
   - 运行 `pre-commit run --all-files` 确保检查通过。
3. 提交并推送：

   ```bash
   git commit -am "Add awesome feature"
   git push origin feature/my-awesome-feature
   ```

4. 发起 Pull Request，并说明背景、改动内容与依赖变化。

## 开源协议

当前仓库根目录尚未提供 LICENSE 文件，因此本 README 不构成法律意义上的授权声明。请在使用前与维护者确认授权条款，并在正式对外发布前补充 LICENSE 文件。

## 致谢

感谢以下开源项目与社区支持：

- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)：机器人动力学建模与算法库。
- [ROS 2](https://www.ros.org/) 及相关生态工具。

也感谢所有对本项目提出建议、报告问题和贡献代码的开发者。
