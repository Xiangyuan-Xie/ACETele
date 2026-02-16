# ACETele

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![Status](https://img.shields.io/badge/status-experimental-orange.svg)](#项目状态) [![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

ACETele 是一个面向机器人平台的实时遥操作系统，重点支持“飞行器 + 机械臂”的组合平台，旨在在仿真环境与真实硬件之间提供统一、可靠的控制与数据通路。

---

- [ACETele](#acetele)
  - [项目简介](#项目简介)
  - [功能特性](#功能特性)
  - [项目状态](#项目状态)
  - [目录结构](#目录结构)
  - [快速开始](#快速开始)
    - [环境依赖](#环境依赖)
    - [安装](#安装)
    - [运行 PX4 + MuJoCo 仿真](#运行-px4--mujoco-仿真)
    - [使用遥操作站端](#使用遥操作站端)
  - [配置说明](#配置说明)
  - [ROS 2 集成与部署](#ros-2-集成与部署)
  - [开发与测试](#开发与测试)
    - [开发环境建议](#开发环境建议)
  - [贡献指南](#贡献指南)
  - [开源协议](#开源协议)
  - [致谢](#致谢)

---

## 项目简介

现代机器人遥操作系统通常由“本地操作端（Leader Station）”“远程执行端（Follower Station）”以及“真实/仿真环境”三部分构成。ACETele 致力于提供一套结构清晰、可扩展的软件框架，以支持如下目标：

- 在 MuJoCo + PX4 SITL 仿真环境中评估控制策略与操作性能；
- 在迁移至真实硬件环境时最大限度复用同一套逻辑；
- 以统一接口接入多种硬件设备（舵机、电机、手柄等），降低集成成本；
- 与 ROS 2 等主流机器人软件生态保持良好兼容性，便于系统集成与扩展。

本仓库提供了遥操作实现、PX4 硬件在环仿真（HIL）适配层、硬件驱动组件、ROS 2 节点以及若干辅助工具脚本。

## 功能特性

- PX4 HIL 仿真
  - 通过 MAVLink 与 PX4 SITL 建立 TCP 链路。
  - 使用 MuJoCo 对带机械臂的四旋翼（x500_arm2x）进行高保真动力学仿真。
  - 仿真 IMU、磁力计、气压计与 GPS 传感器，并注入噪声以更贴近真实情况。
- 主从遥操作
  - 通过 `ConfigLoader` 动态选择站端类型与后端（默认、ROS 2、Mock 等）。
  - 支持单手柄/双手柄、单串口/多串口等不同连线方案。
  - 提供基于 Pinocchio 的机械臂模型加载接口。
- 硬件抽象层
  - Feetech 舵机驱动（基于 feetech SDK）。
  - 通用设备基类，便于扩展新的硬件设备。
  - 手柄（Joystick）驱动，用于人机交互输入。
- ROS 2 集成
  - `ace_station_ros2`：站端 ROS 2 节点。
  - `data_collector_ros2`：数据采集与记录节点。
  - `visualization_ros2`：可视化与监控 GUI。
- 工具与数据处理
  - 模型转换：URDF ↔ MJCF。
  - 动力学参数与推力系数标定。
  - 训练数据采集与格式转换。

## 项目状态

ACETele 目前处于实验性（experimental）阶段，主要面向科研研究和教学演示等场景使用。
项目的接口设计与内部实现仍可能根据需求进行调整，尚不承诺长期的向后兼容性。
如计划在工程化或产品级场景中长期使用，建议在充分评估和测试的基础上，引入必要的封装与版本管理策略。

## 目录结构

仓库根目录的整体结构如下所示：

```text
ACETele/
├─ acetele/
│  ├─ config/          配置文件
│  ├─ core/            核心接口
│  ├─ deploy/          ROS 2 部署包
│  ├─ equipment/       硬件抽象与驱动
│  ├─ simulation/      MuJoCo + PX4 仿真相关脚本与模型
│  ├─ station/         主从遥操作站端逻辑
│  ├─ tools/           实用工具脚本
│  └─ utils/           通用工具函数
├─ README.md           本说明文档
├─ requirements.txt    运行时依赖（与 pyproject.toml 中一致）
├─ pyproject.toml      项目元数据与构建配置
├─ setup.py            setuptools 安装脚本
└─ .pre-commit-config.yaml  开发阶段代码质量检查配置
```

其中，`acetele/simulation` 下的关键文件：

- `fly.py`：MuJoCo + PX4 HIL 仿真入口。
- `px4_interface.py`：与 PX4 的 MAVLink 通信与传感器/控制量封装。
- `description/x500_arm2x/`：带机械臂四旋翼的 URDF/MJCF 模型与网格文件。

`acetele/station` 下的关键模块：

- `base_station.py`：站端抽象基类。
- `ace_leader/`：领导站端实现（含 ROS 2 / mock 等后端）。
- `ace_follower/`：跟随站端实现（含 ROS 2 / mock 等后端）。

`acetele/core` 下的关键模块：

- `make_station.py`：站端工厂函数，对外提供统一的创建接口。

`acetele/deploy` 下包含若干 ROS 2 包，用于在 ROS 2 系统中直接集成 ACETele 的能力。

## 快速开始

### 环境依赖

- 操作系统：推荐使用 Linux；其他平台可根据实际环境进行适配。
- Python：3.9 及以上。
- MuJoCo：通过运行动力学仿真。
- PX4-Autopilot：用于运行 PX4 SITL（外部仿真模式 `none`）。
- ROS 2（可选）：用于运行 `deploy` 目录下的 ROS 2 包。

### 安装

推荐使用 [conda](https://docs.conda.io/) 管理 Python 环境，以便在不同项目之间隔离依赖。以下示例以 `conda` 为例说明安装过程：

```bash
git clone https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele

conda create -n acetele python=3.10
conda activate acetele

pip install -e .
```

### 运行 PX4 + MuJoCo 仿真

1. 在 PX4-Autopilot 仓库中启动 SITL（none 目标）：

   ```bash
   cd /path/to/PX4-Autopilot
   export PX4_SIM_MODEL=none_iris
   make px4_sitl none
   ```

   PX4 启动后将等待来自外部仿真器（本项目中为 MuJoCo）的连接。

2. 在 ACETele 仓库中启动 MuJoCo 仿真：

   ```bash
   cd /path/to/ACETele
   python -m acetele.simulation.fly
   ```

   该脚本会：

   - 加载 `x500_arm2x` 模型；
   - 通过 `PX4Interface` 与 PX4 建立 TCP 连接；
   - 周期性发送 HIL_SENSOR 与 HIL_GPS 消息；
   - 接收 PX4 输出的电机控制量并作用到 MuJoCo 模型。

3. 按需调整 `acetele/config` 下的配置，以选择合适的站端类型与后端。

### 使用遥操作站端

`BaseStation` 及其子类负责将人机输入（如手柄信号）转换为机械臂或平台的目标动作指令。

默认配置入口为 [`acetele/config/default.toml`](acetele/config/default.toml)：

```toml
[basic]
config_file = "ace_leader.toml"
```

例如，在 [`ace_leader.toml`](acetele/config/ace_leader.toml) 中可以看到：

```toml
[basic]
station_type = "ace_leader"
backend = "mock"

[linker.single]
port = "/dev/ttyUSB0"
joint_ids = [0, 1, 2, 3, 4]
joint_signs = [1, -1, -1, -1, 1]
home_poses = [0.0, 0.0, 0.0, 0.0, 0.0]
gripper_id = 4
gripper_type = "ace_leader"
enable_gravity_compensation = true
enable_estimate_external_torque = false
servo_types = ["HL3915", "HL3915", "HL3915", "HL3915", "HL3915"]
```

可以通过修改 `default.toml` 中的 `config_file` 字段快速切换不同的站端配置，例如切换为 `ace_follower.toml` 以使用跟随站与 ROS 2 后端。

在用户自定义脚本中创建站端实例的典型方式如下：

```python
from acetele.core.make_station import make_station

station = make_station()  # 使用默认配置
joint_pos, joint_vel, joint_tau = station.act()
station.close()
```

## 配置说明

配置系统由 [`ConfigLoader`](acetele/config/config_loader.py) 统一管理，核心概念包括：

- `basic.station_type`：站端类型，例如 `"ace_leader"` 或 `"ace_follower"`；
- `basic.backend`：后端实现类型，例如 `"default"`、`"ros2"`、`"mock"`；
- `linker.single` / `linker.dual`：硬件连线配置，包含串口号、关节 ID、符号方向以及初始姿态等信息。

`ConfigLoader` 通过内部的 `_STATION_MAP` 将 `(station_type, backend)` 映射为具体的 Python 模块路径和类名，并由 `make_station` 工厂函数创建站端实例，从而实现“通过配置选择站端实现”的机制。

## ROS 2 集成与部署

`acetele/deploy` 目录下包含若干 ROS 2 包：

- `ace_station_ros2/`：将 ACETele 站端封装为 ROS 2 节点。
- `data_collector_ros2/`：数据采集与记录节点。
- `visualization_ros2/`：基于 C++ 的可视化 GUI 与节点。

在已有的 ROS 2 工作空间中，可将本仓库作为一个子目录进行构建（示意流程如下）：

```bash
cd ~/ros2_ws/src
git clone https://github.com/your-name/ACETele.git
cd ..
colcon build --packages-select ace_station_ros2 data_collector_ros2 visualization_ros2
source install/setup.bash
```

随后即可使用各自的 `launch` 文件启动相应节点，例如：

```bash
ros2 launch ace_station_ros2 ace_station.launch.py
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
```

具体参数请参考各包下的 `config/` 与 `launch/` 目录。

## 开发与测试

本仓库以 Python 为主要开发语言，并通过 [pre-commit](https://pre-commit.com/) 对代码风格与类型检查进行统一约束。

### 开发环境建议

```bash
pip install -e .
pip install pre-commit
pre-commit install
```

在提交代码前，可以手动运行全部检查：

```bash
pre-commit run --all-files
```

主要检查包括：

- 代码格式化：`black`、`isort`。
- 代码质量：`flake8`。
- 类型检查：`mypy`。
- 其他基础检查：YAML 合法性、大文件检测、尾空格等。

如有需要，也可以直接运行：

```bash
mypy acetele
```

以进行类型检查。

## 贡献指南

欢迎以多种形式参与本项目的建设，包括但不限于：

- 修复缺陷或补充文档；
- 扩展新的硬件设备驱动；
- 引入新的仿真模型或控制策略；
- 优化开发流程与工具链配置。

以下为推荐的贡献流程：

1. 将本仓库 Fork 至个人账号。
2. 创建新的功能分支：

   ```bash
   git checkout -b feature/my-awesome-feature
   ```

3. 在本地实现并测试相关修改：
   - 确认与 PX4 / MuJoCo / ROS 2 等外部依赖能正常协同；
   - 运行 `pre-commit run --all-files` 确保代码风格与类型检查通过。
4. 提交更改并推送到你的远程仓库：

   ```bash
   git commit -am "Add awesome feature"
   git push origin feature/my-awesome-feature
   ```

5. 在 Git 平台上发起 Pull Request，并在描述中说明：
   - 修改的动机与背景；
   - 主要改动内容；
   - 相关的外部依赖或配置变更。

在提交 Pull Request 之前，建议在仿真环境以及（如适用）真实硬件环境中完成必要的功能验证。

## 开源协议

当前仓库尚未在根目录提供明确的 LICENSE 文件，因此本 README 不构成任何法律意义上的授权声明。

- 在将本项目用于科研或产品之前，请与项目作者或维护者确认具体授权条款；
- 常见的开源协议包括 MIT、Apache-2.0、BSD-3-Clause 等，可根据实际需求进行选择；
- 建议在正式对外发布之前补充相应的 LICENSE 文件，并在本章节中进行说明。

一旦仓库中添加了 LICENSE 文件，以该文件所载明的条款为准。

## 致谢

本项目的实现离不开以下开源项目与社区的支持（按字母顺序）：

- [MuJoCo](https://mujoco.org/)：机器人动力学仿真引擎。
- [PX4 Autopilot](https://px4.io/)：开源飞行控制软件。
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)：机器人动力学建模与算法库。
- [pymavlink](https://github.com/ArduPilot/pymavlink)：MAVLink Python 实现。
- [ROS 2](https://www.ros.org/) 及相关生态工具。

也感谢所有对本项目提出建议、报告问题和贡献代码的开发者。
