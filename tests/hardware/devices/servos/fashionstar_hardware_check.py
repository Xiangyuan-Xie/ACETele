"""Read-only check for one RX18-R100W-M (ID 1) and four RX8-R501W-M (IDs 2-5).

Run from the ACETele repository with its Python environment activated:
    python tests/hardware/devices/servos/fashionstar_hardware_check.py
    python tests/hardware/devices/servos/fashionstar_hardware_check.py --connect
    python tests/hardware/devices/servos/fashionstar_hardware_check.py --connect --ids 1

The default only validates the plan. --connect requires interactive confirmation.
Only PING, READ_DATA and QUERY_MONITOR packets can be transmitted. There are no
position, enable, release, stop, zero-setting or persistent-write commands.
Product model identity cannot be read through the generic protocol; model labels
below describe the user's ID mapping, not verified hardware model identity.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

# Prefer this checkout when invoked directly without an editable installation.
REPO_ROOT = Path(__file__).resolve().parents[4]
if (REPO_ROOT / "acetele").is_dir():
    sys.path.insert(0, str(REPO_ROOT))

from acetele.hardware.buses import (  # noqa: E402
    BusActor,
    FatalBusError,
    SerialDirectionControl,
    SerialTransport,
    calculate_bus_budget,
)
from acetele.hardware.devices.servos.fashionstar import (  # noqa: E402
    FashionStarBusProtocol,
    FashionStarCommand,
    FashionStarPacketCodec,
    FashionStarProtocolError,
    fashionstar_rs485_profiles,
)

MODELS = {1: "RX18-R100W-M", **dict.fromkeys(range(2, 6), "RX8-R501W-M")}
READ_COMMANDS = frozenset(
    (FashionStarCommand.PING, FashionStarCommand.READ_DATA, FashionStarCommand.QUERY_MONITOR)
)


class ReadOnlyFashionStarProtocol(FashionStarBusProtocol):
    """Reuse the project's codecs, deadlines and monitoring without torque writes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device_information: dict[int, dict[str, int]] = {}

    def _send(self, frame: bytes, *, deadline_ns: int | None = None) -> None:
        # Also guards against inherited methods accidentally sending a control packet.
        if len(frame) < 5 or frame[2] not in READ_COMMANDS:
            raise FatalBusError("只读检测拒绝发送舵机控制或写入指令")
        super()._send(frame, deadline_ns=deadline_ns)

    def _read_parameter(self, servo_id: int, data_id: int, size: int) -> int:
        packet = self._request_response(
            FashionStarPacketCodec.encode_read_data(servo_id, data_id),
            expected_command=FashionStarCommand.READ_DATA,
        )
        value = FashionStarPacketCodec.decode_read_data(
            packet, servo_id=servo_id, data_id=data_id, size=size
        )
        return int.from_bytes(value, byteorder="little", signed=False)

    def connect(self) -> None:
        # Do not call the base connect(): it sends STOP_CONTROL(RELEASE).
        self.device_information.clear()
        self._last_states.clear()
        self._observed_firmware_versions.clear()
        self._enabled_ids.clear()
        self._sync_capable = False
        self._last_command_ns = None
        try:
            self._transport.connect()
            for servo_id in self._devices:
                try:
                    packet = self._request_response(
                        FashionStarPacketCodec.encode_request(
                            FashionStarCommand.PING, bytes((servo_id,))
                        ),
                        expected_command=FashionStarCommand.PING,
                    )
                    if packet.parameters != bytes((servo_id,)):
                        raise FashionStarProtocolError("PING 回应的 ID 不匹配")
                    firmware = self._read_parameter(
                        servo_id, FashionStarPacketCodec.firmware_version_data_id, 2
                    )
                    if firmware <= 0:
                        raise FashionStarProtocolError("固件版本读数无效")
                    response_switch = self._read_parameter(
                        servo_id, FashionStarPacketCodec.response_switch_data_id, 1
                    )
                    self._observed_firmware_versions[servo_id] = firmware
                    self.device_information[servo_id] = {
                        "firmware_version": firmware,
                        "response_switch": response_switch,
                    }
                except Exception as exc:
                    raise RuntimeError(f"ID {servo_id} 通信检查失败：{exc}") from exc
        except BaseException:
            self.disconnect()
            self.device_information.clear()
            raise

    def write_motion(self, targets: Any) -> None:
        raise FatalBusError("这个检测脚本不支持运动指令")

    def execute_safety(self, label: str, payload: Any) -> Any:
        # A read-only diagnostic must not alter an actuator's existing torque state,
        # including when the actor attempts HOLD after a communication fault.
        raise FatalBusError(f"只读检测不执行硬件动作：{label}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RX18 + RX8 舵机只读通信检测")
    parser.add_argument("--connect", action="store_true", help="确认后打开串口；默认仅检查计划")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Ubuntu 中的串口路径")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--duration", type=float, default=5.0, help="监测秒数，默认 5 秒")
    parser.add_argument("--rate", type=float, default=2.0, help="监测频率 Hz，默认 2 Hz")
    parser.add_argument("--timeout", type=float, default=0.5, help="完整监测循环的通信超时秒数")
    parser.add_argument("--direction-control", choices=["auto", "rts"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not args.port.strip() or args.baudrate <= 0:
        parser.error("串口不能为空，波特率必须大于 0")
    if len(set(args.ids)) != len(args.ids) or any(i not in MODELS for i in args.ids):
        parser.error("--ids 必须是 1–5 中的不重复 ID")
    for field in ("duration", "rate", "timeout"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{field} 必须是有限正数")
    if args.timeout > 5:
        parser.error("--timeout 最大为 5 秒")

    profiles = {
        servo_id: fashionstar_rs485_profiles.require(MODELS[servo_id], context=f"ID {servo_id}")
        for servo_id in args.ids
    }
    # Use the project's conservative unicast budget; no sync capability is assumed.
    budget = calculate_bus_budget(
        baudrate=args.baudrate,
        cycle_hz=args.rate,
        wire_bytes_per_cycle=45 * len(profiles),
        turnaround_s_per_cycle=2 * sum(p.minimum_command_interval_s for p in profiles.values()),
    )
    if not budget.feasible:
        parser.error(f"频率过高：占用率 {budget.utilization:.1%}，最高 {budget.maximum_cycle_hz:.2f} Hz")

    print(f"串口：{args.port}，波特率：{args.baudrate}，监测频率：{args.rate:g} Hz")
    for servo_id in args.ids:
        print(f"  ID {servo_id}: {MODELS[servo_id]}（按配置标注，型号未由硬件验证）")
    print(f"估算总线占用率：{budget.utilization:.1%}")
    if not args.connect:
        print("离线检查通过。添加 --connect 后，确认才会打开串口。")
        return 0

    print("请关闭占用此串口的程序，确认串口、供电、机械空间和独立急停。")
    print("本脚本只读取信息，保持舵机现有扭矩状态，不提供运动或急停控制。")
    confirmation = f"CONNECT {args.port}"
    try:
        answer = input(f"确认检测这组舵机，请输入 {confirmation}：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n检测取消，串口未打开。")
        return 130
    if answer != confirmation:
        print("检测取消，串口未打开。")
        return 130

    transport = SerialTransport(
        args.port, args.baudrate,
        direction_control=SerialDirectionControl(args.direction_control),
    )
    protocol = ReadOnlyFashionStarProtocol(
        transport, profiles, operation_timeout_s=args.timeout
    )
    actor = BusActor(
        protocol, cycle_hz=args.rate,
        state_timeout_ns=round(max(3 / args.rate, 3 * args.timeout) * 1e9),
    )
    exit_code = 0
    try:
        actor.connect()
        for servo_id, info in protocol.device_information.items():
            print(
                f"ID {servo_id} 在线：firmware_version = {info['firmware_version']}, "
                f"response_switch = {info['response_switch']}"
            )
            if info["response_switch"] != 0:
                print("  提示：ACETele 正常控制连接要求 response_switch = 0；此处不修改。")
        snapshot = actor.wait_for_snapshot(timeout=args.timeout + 1)
        finish = time.monotonic() + args.duration
        last_timestamp = None
        print(" ID    angle_deg    voltage_V    current_A    power_W    status_hex")
        while True:
            diagnostics = actor.diagnostics()
            if diagnostics.fault:
                raise RuntimeError(diagnostics.fault)
            if diagnostics.recoverable_error_count:
                raise RuntimeError("监测发生通信错误，请检查波特率、接线和串口占用")
            if not snapshot or set(snapshot) != set(args.ids):
                raise RuntimeError("监测回应未包含全部预期 ID")
            timestamp = min(state.timestamp_ns for state in snapshot.values())
            if timestamp != last_timestamp:
                last_timestamp = timestamp
                for servo_id in args.ids:
                    state = snapshot[servo_id]
                    print(
                        f"{servo_id:3d} {math.degrees(state.position_rad):12.2f} "
                        f"{state.voltage_v:12.3f} {state.current_a:12.3f} "
                        f"{state.power_w:10.3f}    0x{state.status:02X}"
                    )
                    if state.status & 0xFE:
                        raise RuntimeError(f"ID {servo_id} 报告硬件状态异常 0x{state.status:02X}")
            remaining = finish - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
            snapshot = actor.get_snapshot()
        print("检测完成：全部预期 ID 均返回监测信息。固件读数可填写到 TOML。")
    except KeyboardInterrupt:
        print("\n检测已中断。")
        exit_code = 130
    except Exception as exc:
        print(f"检测失败：{exc}", file=sys.stderr)
        print("检查供电、A/B 接线、实际 ID、波特率和串口占用；可用 --ids 1 单独检测。", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            actor.disconnect(timeout=max(1.0, args.timeout + 0.5))
        except Exception as exc:
            print(f"关闭串口失败：{exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
