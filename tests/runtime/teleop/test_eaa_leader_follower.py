"""Tests EAA teleoperation sessions with mock buses and no ROS transport.

Physical presets are copied into memory with the mock backend. Only operator
input is injected; alignment, command admission, motion conditioning, heartbeat
loss, and emergency stop run through the existing runtime and session classes.
"""

import collections.abc
import contextlib
import dataclasses
import queue
import time
from unittest import mock

import numpy as np
import pytest

from acetele import config
from acetele import core
from acetele import runtime
from acetele import specification
from acetele.hardware import simulators
from acetele.runtime.teleop import synchronization

_JOINT_NAMES = ("Joint_1", "Joint_2", "Joint_3", "Joint_4")
_WAIT_TIMEOUT_S = 3.0
_POLL_PERIOD_S = 0.01


class _EaaTeleopPair:
    """Exchanges session messages using the original joint teleop interfaces.

    Attributes:
        leader: Existing triggerless leader session under test.
        follower: Existing follower session with a manually clocked motion loop.
    """

    def __init__(
        self,
        leader: runtime.LeaderTeleopSession,
        follower: runtime.FollowerTeleopSession,
        operator_positions: queue.SimpleQueue[dict[int, float]],
    ) -> None:
        """Stores sessions and the mock operator-input mailbox.

        Args:
            leader: Connected EAA leader session using mock devices.
            follower: Connected EAA follower session using mock devices.
            operator_positions: Raw mock positions consumed by the bus actor.
        """
        self.leader = leader
        self.follower = follower
        self._operator_positions = operator_positions
        self._last_delivered_mode: synchronization.LeaderSyncMode | None = None

    def cycle(self, *, link_up: bool = True) -> None:
        """Advances local control and optionally delivers the latest messages.

        Args:
            link_up: Whether peer feedback, modes, and commands are delivered.
        """
        now_ns = time.monotonic_ns()
        follower_state = self.follower.read(now_ns=now_ns)
        if link_up:
            arm = self.follower.runtime.spec.arms[0]
            feedback = follower_state.joints[arm.name]
            self.leader.observe_follower_state(
                feedback.names, feedback.positions, now_ns=now_ns
            )
            self.leader.observe_follower_status(
                self.follower.status, now_ns=now_ns
            )

        leader_state = self.leader.step(now_ns=now_ns)
        self.leader.try_start_tracking(leader_state)
        if not link_up:
            return

        if self.leader.mode != self._last_delivered_mode:
            self.follower.set_mode(self.leader.mode)
            self._last_delivered_mode = self.leader.mode

        if self.leader.mode == synchronization.LeaderSyncMode.TRACKING:
            arm = self.leader.runtime.spec.arms[0]
            target = leader_state.joints[arm.name]
            assert self.follower.write_arm(
                target.names, target.positions, now_ns=now_ns
            )

    def move_leader_by_hand(self, positions: tuple[float, ...]) -> None:
        """Queues a hand-dragged pose for the mock leader bus actor.

        Args:
            positions: Canonical joint positions in radians and URDF order.
        """
        joints = self.leader.runtime.spec.arms[0].joints
        self._operator_positions.put(
            {
                joint.servo_id: position * joint.direction
                for joint, position in zip(joints, positions, strict=True)
            }
        )


def _joint_state(robot_runtime: runtime.FollowerRuntime) -> core.JointState:
    """Reads the first arm through the public runtime interface.

    Args:
        robot_runtime: Connected EAA runtime whose snapshot is requested.

    Returns:
        The latest coherent joint state in canonical order.
    """
    arm = robot_runtime.spec.arms[0]
    return robot_runtime.read().joints[arm.name]


def _wait_for(
    pair: _EaaTeleopPair,
    predicate: collections.abc.Callable[[], bool],
    reason: str,
    *,
    link_up: bool = True,
) -> None:
    """Advances both local sessions until a bounded state condition is met.

    Args:
        pair: Mock EAA sessions whose local loops must keep running.
        predicate: Condition evaluated after each control cycle.
        reason: Failure message if the condition does not become true.
        link_up: Whether the in-memory transport delivers peer messages.
    """
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while True:
        pair.cycle(link_up=link_up)
        if predicate():
            return
        if time.monotonic() >= deadline:
            pytest.fail(reason)
        time.sleep(_POLL_PERIOD_S)


def _align_pair(pair: _EaaTeleopPair) -> None:
    """Explicitly authorizes powered alignment and waits for stable readiness.

    Args:
        pair: Connected mock EAA sessions awaiting initial or renewed sync.
    """
    if pair.leader.mode == synchronization.LeaderSyncMode.HOLD:
        pair.leader.request_sync()
    pair.cycle()
    assert pair.leader.mode == synchronization.LeaderSyncMode.SYNC_REQUEST
    pair.leader.authorize_alignment()
    _wait_for(
        pair,
        lambda: pair.leader.mode == synchronization.LeaderSyncMode.READY,
        "The EAA leader did not converge to the measured follower pose",
    )


def _start_tracking(pair: _EaaTeleopPair) -> None:
    """Starts tracking through the original explicit operator boundary.

    Args:
        pair: Connected mock EAA sessions to align and start.
    """
    _align_pair(pair)
    pair.leader.start_tracking()
    _wait_for(
        pair,
        lambda: (
            pair.follower.status
            == synchronization.FollowerSyncStatus.TRACKING
            and pair.follower.runtime.diagnostics().safety.state
            == runtime.RuntimeSafetyState.ACTIVE
        ),
        "The EAA follower did not admit the synchronized leader command",
    )


@pytest.fixture(name="teleop_pair")
def teleop_pair_fixture() -> collections.abc.Iterator[_EaaTeleopPair]:
    """Connects original EAA sessions while blocking every physical serial path.

    Yields:
        Leader and follower sessions backed only by in-memory mock buses.
    """
    blocked_transport = mock.Mock(
        side_effect=AssertionError("A mock test attempted physical transport")
    )
    sessions: list[
        runtime.LeaderTeleopSession | runtime.FollowerTeleopSession
    ] = []
    with contextlib.ExitStack() as cleanup:
        serial_constructor = cleanup.enter_context(
            mock.patch(
                "serial.Serial",
                side_effect=AssertionError("A mock test attempted serial I/O"),
            )
        )
        for model_name in ("eaa_leader", "eaa_follower"):
            preset = config.packaged_robot_spec(
                model_name, "fashionstar_rs485.toml"
            )
            physical_spec = config.load_robot_spec(preset)
            assert physical_spec.model == model_name
            assert physical_spec.backend == specification.Backend.PHYSICAL
            mock_spec = dataclasses.replace(
                physical_spec, backend=specification.Backend.MOCK
            )
            robot_runtime = runtime.RobotRuntime(
                mock_spec,
                transport_factory=blocked_transport,
                command_timeout_ns=500_000_000,
            )
            session: (
                runtime.LeaderTeleopSession | runtime.FollowerTeleopSession
            )
            if model_name == "eaa_leader":
                session = runtime.LeaderTeleopSession(robot_runtime)
            else:
                session = runtime.FollowerTeleopSession(
                    robot_runtime, motion_cycle_hz=None
                )
            cleanup.callback(session.close)
            session.connect()
            sessions.append(session)

        leader, follower = sessions
        assert isinstance(leader, runtime.LeaderTeleopSession)
        assert isinstance(follower, runtime.FollowerTeleopSession)
        arm = leader.runtime.spec.arms[0]
        protocol = leader.runtime._actors[arm.bus]._protocol
        assert isinstance(protocol, simulators.MockBusProtocol)
        operator_positions: queue.SimpleQueue[dict[int, float]] = (
            queue.SimpleQueue()
        )
        original_read = protocol.read_fast_state

        def read_with_operator_input(
            *, deadline_ns: int | None = None
        ) -> collections.abc.Mapping[int, simulators.MockDeviceState]:
            """Applies queued input inside the actor that owns the mock bus.

            Args:
                deadline_ns: Optional deadline supplied by the real bus actor.

            Returns:
                A coherent snapshot from the original mock protocol.
            """
            try:
                positions = operator_positions.get_nowait()
            except queue.Empty:
                pass
            else:
                assert not protocol._enabled_ids
                protocol._positions.update(
                    {
                        device_id: (position,)
                        for device_id, position in positions.items()
                    }
                )
            return original_read(deadline_ns=deadline_ns)

        cleanup.enter_context(
            mock.patch.object(
                protocol,
                "read_fast_state",
                side_effect=read_with_operator_input,
            )
        )
        follower.hold_position()
        yield _EaaTeleopPair(leader, follower, operator_positions)

    blocked_transport.assert_not_called()
    serial_constructor.assert_not_called()
    assert not leader.runtime.connected
    assert not follower.runtime.connected


def test_follower_starts_in_hold_before_synchronization(
    teleop_pair: _EaaTeleopPair,
) -> None:
    """Checks measured-pose holding before the first leader command.

    Args:
        teleop_pair: Connected EAA sessions using only mock devices.
    """
    follower = teleop_pair.follower
    measured = _joint_state(follower.runtime)
    assert measured.names == _JOINT_NAMES
    assert follower.runtime.diagnostics().safety.state == (
        runtime.RuntimeSafetyState.HOLD
    )
    assert follower.status == synchronization.FollowerSyncStatus.IDLE
    assert not follower.write_arm(
        measured.names, measured.positions, now_ns=time.monotonic_ns()
    )
    teleop_pair.cycle()
    np.testing.assert_allclose(
        _joint_state(follower.runtime).positions, measured.positions
    )


def test_triggerless_leader_requires_alignment_and_explicit_start(
    teleop_pair: _EaaTeleopPair,
) -> None:
    """Checks real alignment motion and both explicit operator boundaries.

    Args:
        teleop_pair: Connected EAA sessions using only mock devices.
    """
    leader = teleop_pair.leader
    initial_pose = (0.08, -0.08, 0.04, -0.04)
    teleop_pair.move_leader_by_hand(initial_pose)
    _wait_for(
        teleop_pair,
        lambda: np.allclose(
            _joint_state(leader.runtime).positions, initial_pose, atol=1e-8
        ),
        "The mock operator input was not sampled",
    )
    assert not leader.uses_start_trigger
    assert not leader.alignment_authorized
    assert leader.mode == synchronization.LeaderSyncMode.SYNC_REQUEST
    assert leader.runtime.diagnostics().safety.state == (
        runtime.RuntimeSafetyState.SAFE_DISABLED
    )
    with pytest.raises(RuntimeError, match="READY"):
        leader.start_tracking()

    _align_pair(teleop_pair)
    np.testing.assert_allclose(
        _joint_state(leader.runtime).positions,
        _joint_state(teleop_pair.follower.runtime).positions,
        atol=0.03,
        rtol=0.0,
    )
    assert teleop_pair.follower.status == (
        synchronization.FollowerSyncStatus.READY
    )
    teleop_pair.cycle()
    assert leader.mode == synchronization.LeaderSyncMode.READY
    leader.start_tracking()
    teleop_pair.cycle()
    assert leader.mode == synchronization.LeaderSyncMode.TRACKING
    assert not leader.effort_assist_active
    assert leader.runtime.diagnostics().safety.state == (
        runtime.RuntimeSafetyState.SAFE_DISABLED
    )


@pytest.mark.parametrize(
    "target",
    ((0.08, -0.06, 0.04, -0.02), (0.04, -0.08, -0.03, 0.05)),
    ids=("positive-wrist", "negative-wrist"),
)
def test_follower_tracks_measured_leader_joints(
    teleop_pair: _EaaTeleopPair, target: tuple[float, ...]
) -> None:
    """Checks four-joint following through both original runtimes and actors.

    Args:
        teleop_pair: Connected EAA sessions using only mock devices.
        target: Hand-dragged leader pose inside all four model limits.
    """
    _start_tracking(teleop_pair)
    teleop_pair.move_leader_by_hand(target)
    _wait_for(
        teleop_pair,
        lambda: np.allclose(
            _joint_state(teleop_pair.follower.runtime).positions,
            target,
            atol=0.002,
            rtol=0.0,
        ),
        "The EAA follower did not converge to measured leader feedback",
    )
    assert teleop_pair.leader.mode == synchronization.LeaderSyncMode.TRACKING
    assert teleop_pair.follower.status == (
        synchronization.FollowerSyncStatus.TRACKING
    )
    np.testing.assert_allclose(
        _joint_state(teleop_pair.leader.runtime).positions,
        target,
        atol=1e-8,
        rtol=0.0,
    )


def test_link_loss_holds_and_requires_explicit_resynchronization(
    teleop_pair: _EaaTeleopPair,
) -> None:
    """Checks lease expiry, rejected stale ownership, and deliberate recovery.

    Args:
        teleop_pair: Connected EAA sessions using only mock devices.
    """
    _start_tracking(teleop_pair)
    follower = teleop_pair.follower
    _wait_for(
        teleop_pair,
        lambda: (
            teleop_pair.leader.mode == synchronization.LeaderSyncMode.HOLD
            and follower.status == synchronization.FollowerSyncStatus.LOST
            and follower.runtime.diagnostics().safety.state
            == runtime.RuntimeSafetyState.HOLD
        ),
        "Peer loss did not hold both EAA control paths",
        link_up=False,
    )
    held_positions = _joint_state(follower.runtime).positions.copy()
    assert not follower.write_arm(
        _JOINT_NAMES, (0.06, -0.06, 0.03, -0.03),
        now_ns=time.monotonic_ns(),
    )

    recovered_pose = (0.06, -0.06, 0.03, -0.03)
    teleop_pair.move_leader_by_hand(recovered_pose)
    _wait_for(
        teleop_pair,
        lambda: np.allclose(
            _joint_state(teleop_pair.leader.runtime).positions,
            recovered_pose,
            atol=1e-8,
        ),
        "The recovered leader feedback was not received",
    )
    assert teleop_pair.leader.mode == synchronization.LeaderSyncMode.HOLD
    assert follower.status == synchronization.FollowerSyncStatus.HOLD
    np.testing.assert_allclose(
        _joint_state(follower.runtime).positions,
        held_positions,
        atol=0.002,
        rtol=0.0,
    )
    with pytest.raises(RuntimeError, match="READY"):
        teleop_pair.leader.start_tracking()

    _start_tracking(teleop_pair)
    teleop_pair.move_leader_by_hand(recovered_pose)
    _wait_for(
        teleop_pair,
        lambda: np.allclose(
            _joint_state(follower.runtime).positions,
            recovered_pose,
            atol=0.002,
            rtol=0.0,
        ),
        "Explicit resynchronization did not restore joint following",
    )


def test_stop_latches_both_runtimes_and_rejects_motion(
    teleop_pair: _EaaTeleopPair,
) -> None:
    """Checks that STOP invalidates motion and cannot silently restart.

    Args:
        teleop_pair: Connected EAA sessions using only mock devices.
    """
    _start_tracking(teleop_pair)
    leader = teleop_pair.leader
    follower = teleop_pair.follower
    leader_generation = leader.runtime.generation
    follower_generation = follower.runtime.generation

    leader.stop()
    teleop_pair.cycle()

    assert leader.mode == synchronization.LeaderSyncMode.STOP
    assert follower.mode == synchronization.LeaderSyncMode.STOP
    assert leader.runtime.diagnostics().safety.state == (
        runtime.RuntimeSafetyState.FAULT
    )
    assert follower.status == synchronization.FollowerSyncStatus.FAULT
    assert leader.runtime.generation > leader_generation
    assert follower.runtime.generation > follower_generation
    assert not follower.write_arm(
        _JOINT_NAMES, (0.05, -0.05, 0.02, -0.02),
        now_ns=time.monotonic_ns(),
    )
    with pytest.raises(RuntimeError, match="stopped leader"):
        leader.request_sync()
    with pytest.raises(RuntimeError, match="faulted follower"):
        follower.set_mode(synchronization.LeaderSyncMode.TRACKING)
