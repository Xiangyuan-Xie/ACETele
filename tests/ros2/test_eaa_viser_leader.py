"""Tests the virtual leader without creating ROS nodes or Viser servers."""

from __future__ import annotations

import pathlib
import sys
import types
import unittest
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "ros2" / "eaa_robot_ros2"))
from eaa_robot_ros2 import viser_leader_node as leader

_LIMITS = ((-2.0, 2.0),) * 4
_POSE = (0.1, 0.2, 0.3, 0.4)


class LeaderControllerTest(unittest.TestCase):
    """Covers synchronization and fail-closed command admission."""

    def setUp(self):
        self.controller = leader._LeaderController(_LIMITS, 0.5, 1.0)
        self.now = 10.0
        self.controller.receive_feedback(leader._JOINT_NAMES, _POSE, self.now)
        self.controller.receive_status("hold")

    def step(self, **kwargs):
        values = dict(
            start=False, hold=False, connected=True, peers_ready=True
        )
        values.update(kwargs)
        return self.controller.step(self.now, _POSE, **values)

    def begin_tracking(self):
        action = self.step(start=True)
        self.assertEqual(action.mode, "idle")
        self.controller.receive_status("idle")
        self.assertEqual(self.step().mode, "sync_request")
        self.controller.receive_status("ready")
        action = self.step()
        self.assertEqual(action.mode, "tracking")
        self.assertEqual(action.seed, _POSE)
        self.assertIsNone(action.command)
        self.controller.receive_status("tracking")
        self.assertEqual(self.step().command, _POSE)

    def test_start_requires_new_barrier_and_ready(self):
        self.controller.receive_status("ready")
        self.assertEqual(self.step(start=True).mode, "hold")
        # A retained READY is insufficient, even when received after Start.
        self.controller.receive_status("ready")
        self.assertIsNone(self.step().mode)
        self.assertNotEqual(self.controller.phase, leader._Phase.TRACKING)
        self.controller.receive_status("hold")
        self.assertEqual(self.step().mode, "sync_request")
        self.assertIsNone(self.step().mode)
        self.controller.receive_status("ready")
        self.assertEqual(self.step().seed, _POSE)

    def test_first_target_uses_latest_feedback(self):
        self.step(start=True)
        self.controller.receive_status("idle")
        self.step()
        pose = (0.4, 0.3, 0.2, 0.1)
        self.controller.receive_feedback(leader._JOINT_NAMES, pose, self.now)
        self.controller.receive_status("ready")
        self.assertEqual(self.step().seed, pose)

    def test_stationary_target_still_refreshes_heartbeat(self):
        self.begin_tracking()
        for _ in range(4):
            self.now += 0.04
            self.assertEqual(self.step().command, _POSE)

    def test_feedback_timeout_holds_and_never_auto_resumes(self):
        self.begin_tracking()
        self.now += 0.3
        self.step()
        self.now += 0.21
        self.assertEqual(self.step().mode, "hold")
        self.controller.receive_feedback(leader._JOINT_NAMES, _POSE, self.now)
        self.controller.receive_status("hold")
        self.assertIsNone(self.step().command)
        self.assertEqual(self.step(start=True).mode, "idle")

    def test_browser_disconnect_and_reconnect_require_start(self):
        self.begin_tracking()
        self.assertEqual(self.step(connected=False).mode, "hold")
        self.assertIsNone(self.step(connected=True).command)

    def test_hold_has_priority_over_start(self):
        self.begin_tracking()
        self.assertEqual(self.step(start=True, hold=True).mode, "hold")
        self.assertEqual(self.controller.phase, leader._Phase.HOLD)

    def test_missing_or_competing_peer_holds(self):
        self.begin_tracking()
        self.assertEqual(self.step(peers_ready=False).mode, "hold")
        self.assertIsNone(self.step(peers_ready=True).command)

    def test_fault_status_holds(self):
        self.begin_tracking()
        self.controller.receive_status("fault")
        self.assertEqual(self.step().mode, "hold")

    def test_invalid_feedback_is_latched_until_hold(self):
        self.begin_tracking()
        self.controller.receive_feedback(
            leader._JOINT_NAMES, (float("nan"), 0, 0, 0), self.now
        )
        # A second sample in the same executor cycle cannot hide the failure.
        self.controller.receive_feedback(leader._JOINT_NAMES, _POSE, self.now)
        self.assertEqual(self.step().mode, "hold")

    def test_invalid_names_and_positions_are_rejected(self):
        samples = (
            (("Joint_1",) * 4, _POSE),
            (leader._JOINT_NAMES[:3], _POSE[:3]),
            (leader._JOINT_NAMES, (0, 0, 0, float("inf"))),
            (leader._JOINT_NAMES, (0, 0, 0, 2.01)),
        )
        for names, positions in samples:
            with self.subTest(names=names, positions=positions):
                self.controller.receive_feedback(names, positions, self.now)
                self.assertIsNone(self.controller.feedback)

    def test_feedback_is_reordered_by_joint_name(self):
        self.controller.receive_feedback(
            leader._JOINT_NAMES[::-1], _POSE[::-1], self.now
        )
        self.assertEqual(self.controller.feedback, _POSE)

    def test_invalid_target_holds_without_clipping(self):
        self.begin_tracking()
        action = self.controller.step(
            self.now, (3.0, 0, 0, 0), start=False, hold=False,
            connected=True, peers_ready=True,
        )
        self.assertEqual(action.mode, "hold")
        self.assertIsNone(action.command)

    def test_sync_timeout_requires_manual_restart(self):
        self.step(start=True)
        for _ in range(11):
            self.now += 0.1
            self.controller.receive_feedback(
                leader._JOINT_NAMES, _POSE, self.now
            )
            action = self.step()
        self.assertEqual(action.mode, "hold")
        self.assertIn("timed out", self.controller.reason)

    def test_loop_stall_holds_even_after_fresh_feedback(self):
        self.begin_tracking()
        self.now += 0.6
        self.controller.receive_feedback(leader._JOINT_NAMES, _POSE, self.now)
        self.assertEqual(self.step().mode, "hold")
        self.assertIn("stalled", self.controller.reason)

    def test_revoked_tracking_cannot_be_hidden_by_later_ready(self):
        self.begin_tracking()
        self.controller.receive_status("lost")
        self.controller.receive_status("ready")
        self.assertEqual(self.step().mode, "hold")

    def test_ready_after_confirmed_tracking_requires_resync(self):
        self.begin_tracking()
        self.controller.receive_status("ready")
        self.assertEqual(self.step().mode, "hold")

    def test_missing_tracking_ack_times_out(self):
        self.step(start=True)
        self.controller.receive_status("idle")
        self.step()
        self.controller.receive_status("ready")
        self.step()
        for _ in range(11):
            self.now += 0.1
            self.controller.receive_feedback(
                leader._JOINT_NAMES, _POSE, self.now
            )
            action = self.step()
        self.assertEqual(action.mode, "hold")

    def test_fault_and_unknown_status_cannot_start(self):
        for status in ("fault", "not-a-status"):
            with self.subTest(status=status):
                self.controller.receive_status(status)
                self.assertEqual(self.step(start=True).mode, "hold")


class RequestsTest(unittest.TestCase):
    """Checks callback mailbox priority without browser activity."""

    def test_hold_cancels_pending_and_following_start(self):
        requests = leader._Requests()
        requests.start(None)
        requests.hold(None)
        requests.start(None)
        self.assertEqual(requests.take(), (False, True))
        self.assertEqual(requests.take(), (False, False))
        requests.start(None)
        self.assertEqual(requests.take(), (True, False))


class StaticModelTest(unittest.TestCase):
    """Loads the actual model with node, server, and serial creation blocked."""

    def test_load_model_has_no_live_side_effects(self):
        path = (
            _ROOT / "acetele" / "config" / "presets" / "eaa_follower"
            / "fashionstar_rs485.toml"
        )
        with (
            mock.patch.object(
                leader.node.Node, "__init__", side_effect=AssertionError
            ),
            mock.patch.object(
                leader.viser, "ViserServer", side_effect=AssertionError
            ),
            mock.patch("serial.Serial", side_effect=AssertionError),
        ):
            model, limits = leader._load_model(str(path))
        self.assertEqual(tuple(model.actuated_joint_names), leader._JOINT_NAMES)
        controller = leader._LeaderController(limits, 0.5, 2.0)
        self.assertEqual(controller.phase, leader._Phase.HOLD)

    def test_config_path_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, "config_path"):
            leader._load_model("")

    def test_invalid_limits_and_timeouts_are_rejected(self):
        cases = (
            (((0.0, 0.0),) * 4, 0.5, 1.0),
            (((-1, float("inf")),) * 4, 0.5, 1.0),
            (_LIMITS, 0.0, 1.0),
            (_LIMITS, 0.5, float("nan")),
        )
        for limits, feedback_timeout, sync_timeout in cases:
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    leader._LeaderController(
                        limits, feedback_timeout, sync_timeout
                    )


class RosAdapterTest(unittest.TestCase):
    """Checks adapter calls with all ROS entities and UI objects replaced."""

    def setUp(self):
        # Bypass Node.__init__: no DDS participants or sockets are constructed.
        self.adapter = object.__new__(leader.ViserLeaderNode)
        self.adapter._closed = False
        self.adapter._server = mock.Mock()
        self.adapter._server.get_clients.return_value = {1: object()}
        self.adapter._requests = leader._Requests()
        self.adapter._viewer = mock.Mock()
        self.adapter._viewer.target.return_value = _POSE
        self.adapter._controller = leader._LeaderController(_LIMITS, 0.5, 1)
        self.adapter._controller.receive_feedback(
            leader._JOINT_NAMES, _POSE, 10.0
        )
        self.adapter._controller.receive_status("hold")
        self.adapter._command_pub = mock.Mock()
        self.adapter._mode_pub = mock.Mock()
        self.adapter._display_period = 0.2
        self.adapter._last_display = -float("inf")

    def tick(self):
        with (
            mock.patch.object(leader.time, "monotonic", return_value=10.0),
            mock.patch.object(self.adapter, "_peers_ready", return_value=True),
            mock.patch.object(self.adapter, "get_logger"),
            mock.patch.object(self.adapter, "get_clock") as clock,
        ):
            from builtin_interfaces.msg import Time
            clock.return_value.now.return_value.to_msg.return_value = Time()
            self.adapter._tick()

    def test_mode_precedes_commands_and_seed_precedes_tracking(self):
        self.adapter._requests.start(None)
        self.tick()
        self.assertEqual(
            self.adapter._mode_pub.publish.call_args.args[0].data, "idle"
        )
        self.adapter._controller.receive_status("idle")
        self.tick()
        self.assertEqual(
            self.adapter._mode_pub.publish.call_args.args[0].data,
            "sync_request",
        )
        self.adapter._controller.receive_status("ready")
        self.tick()
        self.adapter._viewer.seed.assert_called_once_with(_POSE)
        self.adapter._command_pub.publish.assert_not_called()
        self.tick()
        message = self.adapter._command_pub.publish.call_args.args[0]
        self.assertEqual(message.name, list(leader._JOINT_NAMES))
        self.assertEqual(list(message.position), list(_POSE))
        self.adapter._requests.hold(None)
        self.tick()
        self.assertEqual(
            self.adapter._mode_pub.publish.call_args.args[0].data, "hold"
        )
        self.assertEqual(self.adapter._command_pub.publish.call_count, 1)

    def test_qos_and_timer_are_compatible_with_follower(self):
        defaults = {}
        publishers = []
        subscriptions = []

        def publisher(message_type, topic, profile):
            publishers.append((message_type, topic, profile))
            return mock.Mock()

        def subscription(message_type, topic, callback, profile):
            subscriptions.append((message_type, topic, callback, profile))
            return mock.Mock()

        with (
            mock.patch.object(
                self.adapter, "declare_parameter",
                side_effect=lambda key, value: defaults.setdefault(key, value),
            ),
            mock.patch.object(
                self.adapter, "get_parameter",
                side_effect=lambda key: types.SimpleNamespace(
                    value=defaults[key]
                ),
            ),
            mock.patch.object(
                leader, "_load_model", return_value=(mock.Mock(), _LIMITS)
            ),
            mock.patch.object(
                self.adapter, "create_publisher", side_effect=publisher
            ),
            mock.patch.object(
                self.adapter, "create_subscription", side_effect=subscription
            ),
            mock.patch.object(self.adapter, "create_timer") as timer,
            mock.patch.object(self.adapter, "get_logger"),
            mock.patch.object(leader.viser, "ViserServer"),
            mock.patch.object(leader, "_Viewer"),
        ):
            self.adapter._initialize()
        self.assertAlmostEqual(timer.call_args.args[0], 0.04)
        clock = timer.call_args.kwargs["clock"]
        self.assertEqual(
            clock.clock_type, leader.ros_clock.ClockType.STEADY_TIME
        )
        command_profile = publishers[0][2]
        sync_profile = publishers[1][2]
        self.assertEqual(
            command_profile.reliability,
            leader.qos.ReliabilityPolicy.BEST_EFFORT,
        )
        self.assertEqual(command_profile.lifespan.nanoseconds, 50_000_000)
        self.assertEqual(command_profile.depth, 1)
        self.assertEqual(
            sync_profile.durability, leader.qos.DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.assertEqual(
            sync_profile.reliability, leader.qos.ReliabilityPolicy.RELIABLE
        )
        self.assertEqual(subscriptions[0][1], leader._ARM_STATE_TOPIC)
        self.assertEqual(subscriptions[1][1], leader._SYNC_STATUS_TOPIC)

    def test_peer_checks_allow_observers_but_reject_competing_leader(self):
        self.adapter._command_pub.get_subscription_count.return_value = 2
        self.adapter._mode_pub.get_subscription_count.return_value = 1
        self.adapter._command_pub.topic_name = leader._ARM_COMMAND_TOPIC
        self.adapter._mode_pub.topic_name = leader._SYNC_MODE_TOPIC
        self.adapter._state_sub = types.SimpleNamespace(
            topic_name=leader._ARM_STATE_TOPIC
        )
        self.adapter._status_sub = types.SimpleNamespace(
            topic_name=leader._SYNC_STATUS_TOPIC
        )
        with mock.patch.object(
            self.adapter, "count_publishers", return_value=1
        ) as count:
            self.assertTrue(self.adapter._peers_ready())
            count.side_effect = lambda topic: (
                2 if topic == leader._ARM_COMMAND_TOPIC else 1
            )
            self.assertFalse(self.adapter._peers_ready())

    def test_close_cancels_timer_and_is_idempotent(self):
        self.adapter._timer = mock.Mock()
        with mock.patch.object(type(self.adapter), "context") as context:
            context.ok.return_value = True
            self.adapter.close()
            self.adapter.close()
        self.adapter._timer.cancel.assert_called_once()
        self.adapter._server.stop.assert_called_once()
        message = self.adapter._mode_pub.publish.call_args.args[0]
        self.assertEqual(message.data, "hold")


class ViewerStatusTest(unittest.TestCase):
    """Checks that cached status is marked stale after feedback loss."""

    def setUp(self):
        self.controller = leader._LeaderController(_LIMITS, 0.5, 1.0)
        self.controller.receive_feedback(leader._JOINT_NAMES, _POSE, 10.0)
        self.controller.receive_status("tracking")
        self.viewer = object.__new__(leader._Viewer)
        self.viewer._robot = mock.Mock()
        self.viewer._sliders = [mock.Mock() for _ in leader._JOINT_NAMES]
        self.viewer._feedback = mock.Mock()
        self.viewer.target = mock.Mock(return_value=_POSE)

    def test_stale_feedback_marks_cached_tracking_as_stale(self):
        with mock.patch.object(leader.time, "monotonic", return_value=10.6):
            self.viewer.render(self.controller)
        content = self.viewer._feedback.content
        self.assertIn("offline/stale (last received: tracking)", content)
        self.assertIn("Last feedback (rad; stale)", content)
        self.assertIn("0.1000", content)
        self.assertTrue(all(item.disabled for item in self.viewer._sliders))

    def test_fresh_feedback_displays_live_status(self):
        with mock.patch.object(leader.time, "monotonic", return_value=10.1):
            self.viewer.render(self.controller)
        content = self.viewer._feedback.content
        self.assertIn("Follower: tracking", content)
        self.assertNotIn("offline/stale", content)

    def test_missing_feedback_is_explicitly_unavailable(self):
        self.controller.feedback = None
        with mock.patch.object(leader.time, "monotonic", return_value=10.1):
            self.viewer.render(self.controller)
        content = self.viewer._feedback.content
        self.assertIn("offline/stale", content)
        self.assertIn("unavailable", content)


if __name__ == "__main__":
    unittest.main()
