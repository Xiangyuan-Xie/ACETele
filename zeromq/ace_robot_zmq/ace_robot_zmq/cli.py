"""Command-line entry point for direct ACETele ZeroMQ peers."""

from __future__ import annotations

import argparse
import math
import signal
import sys
from pathlib import Path
from typing import Optional, Sequence

from ace_robot_zmq.application import FollowerApplication, LeaderApplication
from ace_robot_zmq.options import CurveCredentials, PeerRole, ZmqTeleopOptions
from ace_robot_zmq.px4_xrce import Px4XrceOptions
from ace_robot_zmq.security import generate_curve_certificates

from acetele.config import load_robot_spec
from acetele.runtime.teleop import TeleopMode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ace-robot-zmq",
        description="Direct low-latency ZeroMQ teleoperation for ACETele",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    role_parsers = {}
    for role in PeerRole:
        role_parser = subparsers.add_parser(role.value)
        role_parsers[role] = role_parser
        role_parser.add_argument("--config", required=True, type=Path)
        role_parser.add_argument("--peer-host", required=True)
        role_parser.add_argument("--bind-host", default="0.0.0.0")
        role_parser.add_argument("--command-port", type=int, default=5555)
        role_parser.add_argument("--state-port", type=int, default=5556)
        role_parser.add_argument("--rate", type=float, default=100.0)
        role_parser.add_argument("--heartbeat-timeout", type=float, default=0.1)
        role_parser.add_argument(
            "--teleop-mode",
            choices=tuple(mode.value for mode in TeleopMode),
            default=TeleopMode.JOINT.value,
        )
        role_parser.add_argument("--translation-scale", type=float, default=2.0)
        role_parser.add_argument("--rotation-scale", type=float, default=1.0)
        role_parser.add_argument("--curve-secret-key", type=Path)
        role_parser.add_argument("--curve-peer-key", type=Path)
    follower = role_parsers[PeerRole.FOLLOWER]
    follower.add_argument("--xrce-prefix", type=Path)
    follower.add_argument("--xrce-agent-port", type=int, default=8888)
    follower.add_argument("--xrce-domain-id", type=int, default=0)
    follower.add_argument("--xrce-namespace", default="")
    follower.add_argument(
        "--xrce-client-key",
        type=lambda value: int(value, 0),
        default=0xACED0001,
    )
    follower.add_argument("--xrce-startup-timeout", type=float, default=3.0)
    keygen = subparsers.add_parser("keygen")
    keygen.add_argument("--output", required=True, type=Path)
    keygen.add_argument("--name", required=True)
    return parser


def _curve(args: argparse.Namespace) -> Optional[CurveCredentials]:
    paths = (args.curve_secret_key, args.curve_peer_key)
    if all(path is None for path in paths):
        return None
    if any(path is None for path in paths):
        raise ValueError(
            "--curve-secret-key and --curve-peer-key must be supplied together"
        )
    return CurveCredentials(args.curve_secret_key, args.curve_peer_key)


def _options(args: argparse.Namespace, role: PeerRole) -> ZmqTeleopOptions:
    if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0.0:
        raise ValueError("--heartbeat-timeout must be finite and positive")
    return ZmqTeleopOptions(
        role,
        args.bind_host,
        args.peer_host,
        args.command_port,
        args.state_port,
        args.rate,
        round(args.heartbeat_timeout * 1e9),
        curve=_curve(args),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate all deployment inputs before opening sockets or robot hardware."""

    args = _parser().parse_args(argv)
    if args.command == "keygen":
        try:
            public_path, secret_path = generate_curve_certificates(
                args.output,
                args.name,
            )
        except (OSError, ValueError) as exc:
            print(f"ace-robot-zmq: {exc}", file=sys.stderr)
            return 2
        print(public_path)
        print(secret_path)
        return 0

    role = PeerRole(args.command)
    try:
        options = _options(args, role)
        spec = load_robot_spec(args.config)
        teleop_mode = TeleopMode(args.teleop_mode)
        if role == PeerRole.LEADER:
            application = LeaderApplication(
                spec,
                options,
                teleop_mode=teleop_mode,
            )
        else:
            xrce_arguments = {
                "agent_port": args.xrce_agent_port,
                "domain_id": args.xrce_domain_id,
                "namespace": args.xrce_namespace,
                "client_key": args.xrce_client_key,
                "startup_timeout_s": args.xrce_startup_timeout,
            }
            if args.xrce_prefix is not None:
                xrce_arguments["prefix"] = args.xrce_prefix
            application = FollowerApplication(
                spec,
                options,
                teleop_mode=teleop_mode,
                translation_scale=args.translation_scale,
                rotation_scale=args.rotation_scale,
                xrce_options=Px4XrceOptions(**xrce_arguments),
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ace-robot-zmq: {exc}", file=sys.stderr)
        return 2

    if options.curve is None:
        print(
            "warning: ZeroMQ teleoperation is using unauthenticated plaintext TCP; "
            "use CURVE outside a trusted wired network",
            file=sys.stderr,
        )
    stopping = False
    received_signal: Optional[int] = None

    def request_stop(signum, _frame) -> None:
        nonlocal received_signal, stopping
        stopping = True
        received_signal = signum

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    primary_error: Optional[BaseException] = None
    cleanup_error: Optional[BaseException] = None
    try:
        application.run(lambda: stopping)
    except BaseException as exc:
        primary_error = exc
    try:
        application.close()
    except BaseException as exc:
        if primary_error is None:
            primary_error = exc
        else:
            cleanup_error = exc
    if primary_error is not None:
        print(f"ace-robot-zmq: {primary_error}", file=sys.stderr)
        if cleanup_error is not None:
            print(f"ace-robot-zmq: cleanup also failed: {cleanup_error}", file=sys.stderr)
        return 1
    if received_signal == signal.SIGINT:
        return 130
    if received_signal == signal.SIGTERM:
        return 128 + signal.SIGTERM
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
