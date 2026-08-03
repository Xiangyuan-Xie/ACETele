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
        <li><a href="#技术栈">技术栈</a></li>
      </ul>
    </li>
    <li>
      <a href="#快速开始">快速开始</a>
      <ul>
        <li><a href="#环境要求">环境要求</a></li>
        <li><a href="#安装">安装</a></li>
        <li><a href="#硬件自检">硬件自检</a></li>
      </ul>
    </li>
    <li>
      <a href="#使用">使用</a>
      <ul>
        <li><a href="#python-api">Python API</a></li>
        <li><a href="#配置系统">配置系统</a></li>
        <li><a href="#ros-2-部署">ROS 2 部署</a></li>
        <li><a href="#zeromq-部署">ZeroMQ 部署</a></li>
      </ul>
    </li>
    <li><a href="#贡献">贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>

## 项目简介

ACETele 是一套面向机器人遥操作与数据采集的 Python 工程框架，提供并列的 ROS 2 与
ZeroMQ 适配器，覆盖从本地验证到真实硬件部署的统一开发流程。

```text
ACETele/
├── acetele/
│   ├── core/         厂商无关的状态与命令契约
│   ├── specification/ 静态总线、控制与机器人规格
│   ├── config/       TOML 加载、预置配置与资源目录
│   ├── model/        URDF 资源与 Pinocchio 模型
│   ├── control/      无线程位置与笛卡尔控制算法
│   ├── estimation/   鲁棒关节状态估计
│   ├── hardware/     总线、设备适配器、输入与仿真器
│   ├── runtime/      预检、生命周期、安全与遥操作会话
│   └── tools/        检查、标定与统一 TUI
├── ros2/             第一方 ROS 2 功能包
├── zeromq/           独立的直接 TCP 遥操作适配器及其原生 PX4 XRCE 组件
├── third_party/      PX4、RealSense 与固定 XRCE 子模块
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
- [ZeroMQ](https://zeromq.org/)（可选遥操作传输）

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 快速开始

### 环境要求

- [Python 3.10 及以上](https://www.python.org/downloads/)
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

确认 Python 版本为 3.10 或更高。

2. 选择 HTTPS 或 SSH 克隆仓库并初始化子模块：

```bash
# HTTPS
git clone --recursive https://github.com/Xiangyuan-Xie/ACETele.git

# SSH
git clone --recursive git@github.com:Xiangyuan-Xie/ACETele.git

cd ACETele
git submodule update --init --recursive
```

以上两条 `git clone` 命令只需执行一条。子模块使用相对 URL，会自动跟随主仓库选择 HTTPS
或 SSH，不需要分别维护子模块地址。

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

如需使用不依赖 ROS 2 的双机 ZeroMQ 遥操作，请额外安装独立适配器：

```bash
python -m pip install -e apps/ace_operator_ui
# Follower 图传
python -m pip install -e "zeromq/ace_robot_zmq[camera]"
# Leader 可视化
python -m pip install -e "zeromq/ace_robot_zmq[visualization]"
```

Jetson 等无法从 PyPI 安装 `pyrealsense2` 的平台需按 Intel librealsense 文档安装对应
Python binding；ZMQ 包会在打开相机前明确检查该能力。

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

rosdep install --from-paths ros2 third_party/px4_msgs \
  third_party/realsense_ros --ignore-src -r -y
colcon build --symlink-install \
  --base-paths ros2 third_party/px4_msgs third_party/realsense_ros
source install/setup.bash
```

### 硬件自检

项目提供 HLS TTL 的 Leader/Follower 真实硬件配置；ROS 2 通用 launch 默认使用 Leader
配置。首次运行前请核对端口、舵机 ID、型号和机械状态，再通过统一 TUI 完成不会打开
串口的静态预检：

```bash
python -m acetele.tools.tui
```

内置配置会在进入主菜单前完成预检；自定义配置会在选择时完成预检。预检通过只表示配置、
URDF、型号能力和总线预算一致，不代表已经完成真机安全验证。无硬件环境请将自有配置的
`basic.backend` 设为 `mock`，或直接运行测试套件。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

## 使用

### Python API

`RobotRuntime` 是唯一的 Python 机器人入口。构造函数只执行静态预检；`connect()` 才会
创建串口与 Actor 线程，`disconnect()` 负责有界清理资源。

```python
from acetele.config import load_robot_spec
from acetele.runtime import RobotRuntime

spec = load_robot_spec("acetele/config/presets/ace_follower/feetech_sms_rs485.toml")
runtime = RobotRuntime(spec)
runtime.connect()
try:
    state = runtime.read()
    print(state.joints["single"].positions)
finally:
    runtime.disconnect()
```

`RobotState` 按装配名保存只读 `JointState`。物理舵机位置与速度经过带 NIS 门控和物理
创新约束的低延迟状态估计器；`runtime.diagnostics()` 提供总线周期、状态年龄、命令覆盖和
滤波诊断，不需要访问厂商寄存器或总线 ID。

#### 多厂商 RS485 Runtime

新的硬件运行层使用“每个物理端口一个 Actor”，将安全事务 FIFO、最新运动命令邮箱、
周期状态读取和慢速遥测统一到单一串口所有者中。当前提供以下协议级适配：

- FEETECH HLS packet TTL；
- FEETECH SMS/SM packet RS485；
- FEETECH Modbus-RTU RS485；
- FashionStar UART/RS485 packet；
- Linker Hand 通用 RS485 协议，当前 profile 覆盖 O6、L6、L7 和 L10，不绑定单一型号。

新配置必须显式声明总线、每个关节及型号 profile。统一 TUI 会在不打开串口的情况下完成
URDF、型号、固件能力和总线占用率预检，并在选择配置时显示结果。

每个 `port` 只能声明一个 bus；同一串口上的机械臂和末端执行器必须归入同一个 bus，
确保该端口始终只有一个 Actor 所有者。未知 TOML 字段会直接报错，不会静默回退到默认值。

ROS 2 启动入口只接受 `buses + joints` schema，并使用组合持有 `RobotRuntime` 的
Leader/Follower 节点。不传 `config_path` 时默认启动 HLS TTL Leader：

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/presets/ace_follower/feetech_sms_rs485.toml"
```

HLS TTL 配置位于 `ace_leader/feetech_hls_ttl.toml` 和
`ace_follower/feetech_hls_ttl.toml`。通用 launch 默认使用 Leader 配置；启动 Follower
或其他硬件组合时必须通过 `config_path` 显式选择。

Follower 完成完整状态读取后，会无条件以实测位置作为目标使能扭矩并进入 `HOLD`，因此即使
没有 Leader 在线也会保持当前位置。该安全策略没有运行时关闭选项；自动保持仍不能替代独立
硬件急停。

同步时平行夹爪保持无扭矩并作为安全扳机。收到健康的 Follower 状态后，Leader 机械臂自动
带电并对齐 Follower；达到同步姿态后，将夹爪释放到归一化位置 `0.25` 以下再夹到 `0.75`
以上，即进入 `TRACKING`，Leader 机械臂扭矩随即释放。跟踪失联或自动控制异常只进入
`HOLD`，不会自动重新锁住 Leader；恢复时仍需一次完整夹爪手势授权新的同步周期。旧触发
状态不会跨同步周期复用。
没有平行夹爪的 Leader 不会自动带电或开始跟踪，必须显式调用
`/ace_leader/authorize_alignment` 和 `/ace_leader/start_tracking` 两个 `std_srvs/Trigger`
service。显式急停入口为 `/ace_leader/emergency_stop` 和 `/ace_follower/emergency_stop`。

高频 arm/gripper command 与 state 话题使用带有限 lifespan 的
`BEST_EFFORT + KEEP_LAST(1)`，同步状态话题使用 `RELIABLE + KEEP_LAST(1)`。合法命令在订阅
回调中直接进入最新值邮箱，不会等待额外控制定时器。平行夹爪保留
`/ace_leader/gripper/command` 和
`/ace_follower/gripper/state`；灵巧手使用独立的 `/ace_leader/end_effector/command` 和
`/ace_follower/end_effector/state`，不会被当作夹爪同步扳机。

每条总线的 Actor 独立执行命令 watchdog：合法运动心跳超时后清空旧 generation 并进入
`HOLD`；连续运动写失败或快速状态持续超时会清空待执行命令、尽力保持最后可信姿态，再锁存
总线故障。通信及状态可信度故障保持最后一次成功命令；设备明确报告过温、过载等硬件状态
故障时请求协议支持的最强禁用动作；无法软件禁用的设备会明确要求外部急停。显式 `STOP`/急停
始终请求最强停止动作。
总线诊断中的
`p95_motion_end_to_end_s` 和 `p99_motion_end_to_end_s` 从命令接收时间统计到协议写调用成功
返回，区别于仅表示进入最新值邮箱的 Runtime stage 耗时。Actor 仍与主进程共享生命周期，
无法覆盖进程被强制终止、主机掉电或内核失效，因此物理急停仍是生产部署的必要条件。

FashionStar 与 Linker Hand 当前完成了官方协议帧和测试替身验证，接入生产系统前仍必须完成
对应型号的真机身份、断线 HOLD、急停和持续负载测试。FEETECH packet、FashionStar 和
Linker Hand 路径均无法逐设备验证扭矩禁用结果，因此其物理配置必须设置
`external_estop = true`，并实际配备独立硬件急停。该字段只是安全前提声明，不会代替硬件回路。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### 配置系统

多厂商 RS485 运行层使用 `buses + joints` schema。`joint.name` 是 URDF/ROS 2 运动学名称，
`servo_id` 仅是总线地址，二者不得互相推导。示例：

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

运行时启动前会按最坏情况估算线速、帧长、响应和换向间隔；占用率超过 70% 时拒绝启动。
HLS 型号尚无可信的公开型号寄存器对照值，因此配置必须显式选择精确型号 Profile，连接时
读取并记录舵机报告值，但不会拿猜测值做匹配。获得可信值后可为关节增加可选的
`expected_model_number`，启用严格身份校验。对于协议无法读取型号的物理总线，还必须显式设置
`allow_unverified_identity = true`，预检会持续显示 `verified_identity=false`；这只是对协议限制的
明确确认，不代表型号已经由软件验证。FashionStar 会另外读取并严格比对 `firmware_version`。
已知 HLS profile 会使用型号对应的 KT 和空载电流估计关节输出力矩；缺少官方参数的
RS485 型号不会套用近似型号常数。

关键字段：

| 字段 | 作用 |
| --- | --- |
| `basic.model` | 机器人模型与打包 URDF 名称 |
| `basic.backend` | `physical` 或 `mock` |
| `basic.urdf_path` | 可选的显式 URDF 路径；省略时查找打包模型 |
| `buses.<name>.type` | 厂商协议与物理总线类型 |
| `buses.<name>.port` | 由单个 Actor 独占的串口 |
| `buses.<name>.cycle_hz` | 目标总线周期频率 |
| `buses.<name>.external_estop` | 已实际配备独立硬件急停；无法验证关断的物理总线必须为 `true` |
| `buses.<name>.allow_unverified_identity` | 明确接受协议无法读取产品型号；仅在人工核对硬件后设置 |
| `arms.<name>.bus` | 机械臂所属总线 |
| `arms.<name>.tool_frame` | 末端位姿控制使用的 URDF TCP link；`ee_pose` 模式必须配置 |
| `arms.<name>.joints` | 按 URDF 顺序声明的关节列表 |
| `joint.name` | URDF/ROS 2 运动学名称 |
| `joint.servo_id` | 厂商总线地址 |
| `joint.servo_model` | 必须存在对应官方 Profile 的型号 |
| `joint.direction` | 关节方向，只允许 `-1` 或 `1` |
| `joint.home_position_rad` | 舵机位于机械 Home 姿态时应写入的关节角，用于非易失标定 |
| `joint.expected_model_number` | 可选的 uint16 型号寄存器期望值；提供后连接时严格校验 |
| `joint.firmware_version` | FashionStar 物理舵机的预期固件版本；连接时读取并严格校验 |

`mock` 和 `physical` 使用同一份 typed schema。未知字段、重复端口、未知型号、错误关节顺序、
无效限位或超过 70% 的总线预算都会在打开硬件前失败。FEETECH packet 舵机完成机械 Home
对齐并确认所有关节可安全静止后，推荐打开统一终端工具：

```bash
python -m acetele.tools.tui
```

选择 `Calibrate FEETECH Home` 后，界面会显示所有关节的舵机 ID、机械 Home、方向和 Home
目标原始位置。确认完整写入计划后按 Enter，TUI 才会退出并执行标定。该流程只接受
physical FEETECH packet 配置，不允许单独遗漏某个机械臂或夹爪关节。

TUI 使用确认页持有的同一份不可变配置完成全量预检和标定，只允许在 `SAFE_DISABLED`
状态写入非易失偏置。FashionStar、FEETECH Modbus 和 Linker Hand 不会套用该标定流程。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### ROS 2 部署

第一方 ROS 2 包位于 `ros2/`，外部依赖位于 `third_party/`：

| 包 | 作用 |
| --- | --- |
| `ace_robot_ros2` | 根据 `config_path` 启动 leader 或 follower 机器人节点 |
| `data_collector_ros2` | 根据遥控通道状态触发 rosbag 数据录制 |
| `visualization_ros2` | 用于显示 RGB-D 图像、关节状态和 topic 运行状态 |
| `third_party/px4_msgs` | PX4 消息定义子模块 |
| `third_party/realsense_ros` | RealSense 相机 ROS 2 驱动子模块 |

构建示例：

```bash
colcon build --symlink-install \
  --base-paths ros2 third_party/px4_msgs third_party/realsense_ros
source install/setup.bash
```

常用启动命令：

```bash
python -m acetele.tools.tui
```

选择 `Launch ROS 2 Robot` 后，可以选择内置或自定义 RobotSpec、关节/末端位姿模式及位姿
缩放比例。按 Enter 确认后，TUI 会退出 curses 并直接启动经过预检的 `ros2 launch` 进程；
终端输出、信号和退出码均由该进程继承。最近一次启动和标定选择保存在 XDG state 目录中。

也可以直接使用以下命令：

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/presets/ace_follower/feetech_hls_ttl.toml"
ros2 launch data_collector_ros2 data_collector.launch.py
ros2 launch visualization_ros2 visualization.launch.py
```

默认 `teleop_mode:=joint` 保持关节空间遥操作。末端位姿遥操作需要在 Leader 和 Follower
两侧使用相同模式启动：

```bash
ros2 launch ace_robot_ros2 ace_robot.launch.py \
  config_path:="$PWD/acetele/config/presets/ace_follower/feetech_hls_ttl.toml" \
  teleop_mode:=ee_pose translation_scale:=2.0 rotation_scale:=1.0
```

同步阶段仍使用关节位置；进入 `TRACKING` 后，第一帧 `PoseStamped` 建立 Leader/Follower
末端相对锚点，不产生位置跳变。之后 Leader 平移按 `2.0` 倍映射，旋转按 `1.0` 倍映射。
当前 4-DOF 机械臂无法精确跟踪任意六维 SE(3) 位姿，因此逆运动学优先跟踪可达平移，
再在剩余零空间中减小姿态误差。该模式不会绕过 URDF 限位、命令 deadline、同步心跳或
总线安全状态机。

通用输入 `/ace_teleop/arm/ee_pose/command` 使用 `geometry_msgs/PoseStamped` 和
`BEST_EFFORT + KEEP_LAST(1)`；Follower 实际 TCP 位姿发布到
`/ace_follower/arm/ee_pose/state`。后续 VR 只需作为该输入话题的唯一有效发布者，并提供
稳定、非空的参考坐标系；Follower 的映射和 IK 路径无需修改。

常见 topic：

- `/ace_leader/arm/command`
- `/ace_teleop/arm/ee_pose/command`
- `/ace_follower/arm/state`
- `/ace_follower/arm/ee_pose/state`
- `/ace_leader/gripper/command`
- `/ace_follower/gripper/state`
- `/ace_leader/arm/sync_mode`
- `/ace_follower/arm/sync_status`
- `/fmu/in/arm_joint_state`

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>

### ZeroMQ 部署

`ace-robot-zmq` 是与 ROS 2 平行的独立适配器，直接复用相同的 `RobotRuntime`、同步状态机、
逆运动学、命令 deadline 和失联 `HOLD`。它不启动 ROS 2 或 `rclpy`，但 ZMQ Follower
会强制启动隔离的 Micro XRCE-DDS Agent 2.4.2 和原生 sidecar，将实测机械臂状态发布到
PX4 `/fmu/in/arm_joint_state`。默认遥操作链路使用两个最新值 TCP 流：Leader 在 `5555`
端口发布命令，Follower 在 `5556` 端口发布状态。
运行中的 ZMQ 进程收到 `SIGUSR1` 时执行显式 STOP；普通 `SIGINT/SIGTERM` 仍走有界退出流程。

首次使用前构建固定版本的原生栈：

```bash
git submodule update --init --recursive
cmake -S zeromq/ace_robot_zmq/xrce -B build/ace_robot_zmq-xrce \
  -DACETELE_XRCE_PREFIX="$HOME/.local/lib/acetele/xrce-2.4.2"
cmake --build build/ace_robot_zmq-xrce --parallel
```

该构建不读取或覆盖 `/usr/local` 中可能存在的 Agent 3.x。Follower 启动时核对 Agent、
Client 和 `ArmJointState` schema；版本不匹配、UDP `8888` 被占用或实体创建失败时会在
打开机械臂串口前退出。

TUI 的 `Launch ZMQ Robot` 只启动遥操作控制。图传作为独立进程运行，因此相机或界面故障
不会刷新或中断机械臂心跳。Follower 先查看设备并发布双路压缩 RGB-D，Leader 再订阅显示：

```bash
python -m ace_robot_zmq cameras
python -m ace_robot_zmq camera \
  --front-serial FRONT_REALSENSE_SERIAL \
  --wrist-serial WRIST_REALSENSE_SERIAL

# Leader 主机，FOLLOWER_IP 替换为 Follower 的有线网地址
python -m ace_robot_zmq visualize --follower-host FOLLOWER_IP
```

ZMQ Follower 同样在硬件连接后无条件保持实测位置，不提供绕过该启动安全策略的参数。

控制只使用 `5555/5556`，图传只使用 `5562`。图传发送 JPEG 彩色图、Zstd 压缩深度图及
相机标定元数据，不保存 MCAP、不发送关节遥测，也不订阅 PX4 输出。原有 `ArmJointState`
XRCE Publisher 仍是关键安全链路，故障会使 Follower 进入 `HOLD`。

PX4 通过有线网连接 Follower 主机上的 Agent：

```text
UXRCE_DDS_CFG=Ethernet
UXRCE_DDS_AG_IP=<Follower Ethernet IP>
UXRCE_DDS_PRT=8888
UXRCE_DDS_DOM_ID=0
```

PX4 的 `UXRCE_DDS_KEY` 默认保持 `1`；sidecar 默认使用独立的 `0xACED0001`，两者必须非零
且不同。防火墙需要允许 Leader/Follower 的 ZMQ TCP `5555/5556/5562`，并允许 PX4 到
Follower UDP `8888`。Follower 只聚合 RobotSpec 中的
机械臂关节，按装配顺序发布 4 至 14 个滤波后实测位置和速度；夹爪及灵巧手不会进入该消息。

两侧必须使用相同的 `--teleop-mode joint|ee_pose`。位姿模式的
`--translation-scale` 和 `--rotation-scale` 由 Follower 应用。每次进程启动都会生成新的
session ID；断线、乱序帧或对端重启都会清除旧命令并要求重新同步。只有合法 arm 命令会刷新
本机 100 ms 心跳，因此损坏或伪造帧不能阻止进入 `HOLD`。

明文模式仅适用于可信有线局域网。非可信网络应在两台主机分别生成证书，并为每一侧配置
“本机私钥 + 对端公钥”：

```bash
ace-robot-zmq keygen --output keys --name leader
ace-robot-zmq keygen --output keys --name follower

# Leader 参数
--curve-secret-key keys/leader.key_secret --curve-peer-key keys/follower.key

# Follower 参数
--curve-secret-key keys/follower.key_secret --curve-peer-key keys/leader.key
```

CURVE 会加密连接并只允许配置的对端公钥。私钥权限必须禁止组用户和其他用户读取。
VR 或其他位姿源可使用 `ace_robot_zmq.PoseLeaderClient` 发布最新
`EndEffectorPose`；调用方负责采样循环与稳定参考坐标系，客户端负责 session、序列和同步握手，
Follower 仍走相同的相对锚点、IK 与硬件安全路径。

Agent、sidecar、IPC ACK 或 DDS session 在运行中失效时，Follower 会清空旧命令、进入
`HOLD` 并以非零状态退出。XRCE 写成功只确认本地发布链路可用，不替代 PX4 应用层健康检查。

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

如果需要修改 `third_party/px4_msgs/`、`third_party/realsense_ros/` 或其他子模块，请先在对应子模块内完成修改并提交：

```bash
cd third_party/px4_msgs
git checkout -b feat/your-change
git add .
git commit -m "feat: your change"
```

随后回到 ACETele 父仓库，更新并提交对应的 gitlink：

```bash
cd ..
git add third_party/px4_msgs
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
