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
  让意图跨越距离，成为动作。
  <br />
  <a href="README.en.md">English</a>
</p>

</div>

<details>
  <summary>目录</summary>
  <ol>
    <li><a href="#项目简介">项目简介</a></li>
    <li><a href="#安装">安装</a></li>
    <li><a href="#推荐入口">推荐入口</a></li>
    <li><a href="#真机准备与标定">真机准备与标定</a></li>
    <li><a href="#遥操作">遥操作</a></li>
    <li><a href="#zeromq-与-px4">ZeroMQ 与 PX4</a></li>
    <li><a href="#自定义配置">自定义配置</a></li>
    <li><a href="#python-api">Python API</a></li>
    <li><a href="#开发检查">开发检查</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>

## 项目简介

ACETele 将机器人规格、运动学、控制、安全状态机和硬件总线收敛到同一套 `RobotRuntime`。目前支持 FEETECH HLS TTL、FEETECH SMS/SM RS485、FEETECH Modbus-RTU、FashionStar RS485 和 Linker Hand RS485。

> [!WARNING]
> 真机运行必须配备独立硬件急停。软件超时、扭矩禁用和总线诊断不能替代断电回路。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 安装

要求 Python 3.10 或更高版本。ROS 2 Humble 仅在使用 ROS 2 适配器时需要。

```bash
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git
cd ACETele

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

使用 SSH 时将 clone 地址替换为：

```text
git@github.com:Xiangyuan-Xie/ACETele.git
```

子模块使用相对 URL，会跟随主仓库的 HTTPS 或 SSH 克隆方式。

### ZeroMQ 可选组件

```bash
python -m pip install -e apps/ace_operator_ui
python -m pip install -e zeromq/ace_robot_zmq

# Follower 双 RealSense 图传
python -m pip install -e "zeromq/ace_robot_zmq[camera]"

# Leader 图传界面
python -m pip install -e "zeromq/ace_robot_zmq[visualization]"
```

### ROS 2 工作空间

首次使用 `rosdep` 时，先运行一次 `sudo rosdep init`。

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

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 推荐入口

```bash
python -m acetele.tools.tui
```

TUI 会先执行不打开串口的静态预检，再直接执行确认后的操作：

| 工作流 | 用途 |
| --- | --- |
| `Launch ROS 2 Robot` | 启动 ROS 2 Leader 或 Follower |
| `Launch ZMQ Robot` | 启动点对点 ZeroMQ 遥操作 |
| `Calibrate FEETECH Home` | 写入整机 FEETECH Home 偏置 |

内置配置位于 `acetele/config/presets/`：

| 配置 | 硬件 |
| --- | --- |
| `ace_leader/feetech_hls_ttl.toml` | HLS TTL Leader |
| `ace_follower/feetech_hls_ttl.toml` | HLS TTL Follower |
| `ace_follower/feetech_sms_rs485.toml` | SMS/SM RS485 Follower |
| `ace_follower/fashionstar_rs485.toml` | FashionStar RS485 Follower |

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 真机准备与标定

1. 核对电源、急停、串口、舵机 ID、型号、方向和机械限位。
2. 将当前用户加入串口组，然后重新登录：

   ```bash
   sudo usermod -aG dialout "$USER"
   ```

3. 将所有关节手动放到 RobotSpec 声明的机械 Home 姿态。
4. 运行 TUI，选择 `Calibrate FEETECH Home`，检查完整写入清单后确认。

标定会写入非易失偏置，并覆盖机械臂与末端执行器的全部 FEETECH 关节。该流程不适用于
FashionStar、FEETECH Modbus 或 Linker Hand。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 遥操作

在 Leader 和 Follower 主机分别启动 TUI，选择对应 RobotSpec 和相同的遥操作模式。

启动流程：

1. Follower 读取当前位置，上电并进入保持状态。
2. Leader 收到 Follower 状态后自动对齐。
3. 对齐完成后，先将 Leader 夹爪释放到 `0.25` 以下，再夹到 `0.75` 以上。
4. 系统进入 `TRACKING`，Leader 机械臂切换到本地力矩辅助。

短时网络抖动只保留最新目标；持续失联会停止远端命令并进入 `HOLD`。重新连接后必须重新
同步。Follower 的位置保持、Leader 的重力辅助和通信超时均不能代替硬件急停。

### 遥操作模式

| 模式 | 行为 |
| --- | --- |
| `joint` | 默认模式，直接映射关节位置 |
| `ee_pose` | Leader 正运动学、相对位姿映射、Follower 逆运动学 |

`ee_pose` 默认平移比例为 `2.0`，旋转比例为 `1.0`。当前 4-DOF 机械臂只能跟踪完整
SE(3) 目标的可达投影，位置任务优先于不可达姿态。

### ROS 2 手动启动

推荐使用 TUI。需要脚本化运行时可直接调用：

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

位姿模式在两侧统一增加：

```text
teleop_mode:=ee_pose translation_scale:=2.0 rotation_scale:=1.0
```

显式急停服务：

```text
/ace_leader/emergency_stop
/ace_follower/emergency_stop
```

Follower 会将 4 至 14 个机械臂关节的滤波后实测位置和速度发布到
`/fmu/in/arm_joint_state`；夹爪和灵巧手不进入该消息。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## ZeroMQ 与 PX4

ZeroMQ 使用 Leader 命令端口 `5555` 和 Follower 状态端口 `5556`。Follower 还会启动固定
版本的 XRCE Agent 与原生 sidecar，经 UDP `8888` 向 PX4 发布 `ArmJointState`。

首次使用前构建 XRCE 组件：

```bash
git submodule update --init --recursive
cmake -S zeromq/ace_robot_zmq/xrce -B build/ace_robot_zmq-xrce \
  -DACETELE_XRCE_PREFIX="$HOME/.local/lib/acetele/xrce-2.4.2"
cmake --build build/ace_robot_zmq-xrce --parallel
```

PX4 参数：

```text
UXRCE_DDS_CFG=Ethernet
UXRCE_DDS_AG_IP=<Follower 有线网 IP>
UXRCE_DDS_PRT=8888
UXRCE_DDS_DOM_ID=0
```

明文 ZeroMQ 仅适用于可信有线局域网。非可信网络应使用 `ace-robot-zmq keygen` 生成 CURVE
证书，并在 TUI 中为双方配置本机私钥和对端公钥。

### 可选图传

图传独立于控制链路，相机或界面故障不会刷新机械臂心跳。

```bash
# Follower
python -m ace_robot_zmq cameras
python -m ace_robot_zmq camera \
  --front-serial FRONT_SERIAL --wrist-serial WRIST_SERIAL

# Leader
python -m ace_robot_zmq visualize --follower-host FOLLOWER_IP
```

图传使用 TCP `5562`，传输 JPEG 彩色图、Zstd 深度图和相机标定信息，不进行数据录制。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 自定义配置

RobotSpec 使用 TOML 描述总线、机械臂和关节。关节名用于 URDF 与控制，`servo_id` 只表示
总线地址，两者不能互相推导。

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
allow_unverified_identity = true # 仅在人工核对舵机型号后使用

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

示例只展开了一个关节；实际配置必须按 URDF 顺序列出整条机械臂。建议复制最接近的内置
配置后修改，而不是从空文件开始。

启动前会检查 TOML 字段、URDF 映射、型号 Profile、关节限位和总线预算。任何静态错误都会
在打开串口前失败。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

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

构造 `RobotRuntime` 只执行静态预检；`connect()` 才创建串口和总线 Actor。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 开发检查

```bash
python -m pytest
python -m compileall acetele
pre-commit run --all-files
```

涉及 ROS 2 时额外运行 `colcon build` 和 `colcon test`。涉及真机时，请记录硬件型号、配置、
测试时长和急停验证结果。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 许可证

ACETele 使用 [Apache License 2.0](LICENSE)。第三方子模块遵循各自声明的许可证。

## 联系

- 项目维护者：Xiangyuan Xie
- 项目地址：<https://github.com/Xiangyuan-Xie/ACETele>

## 致谢

- [ROS 2](https://docs.ros.org/)
- [Pinocchio](https://stack-of-tasks.github.io/pinocchio/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>
