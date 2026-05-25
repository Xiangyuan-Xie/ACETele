from typing import Optional, Sequence, Tuple

import numpy as np
from px4_msgs.msg import VehicleLandDetected
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.feetech_driver import TorqueEnable
from acetele.robot.ace_leader.ace_leader import AceLeaderRobot
from acetele.utils.teleop_sync import (
    FollowerSyncStatus,
    LeaderSyncMode,
)


class AceLeaderROS2Robot(Node, AceLeaderRobot):
    def __init__(self, config_loader: ConfigLoader):
        Node.__init__(self, "ace_leader_robot_node")
        AceLeaderRobot.__init__(self, config_loader)
        self.declare_parameter("control_rate", 100.0)
        self._control_rate = self.get_parameter("control_rate").value
        self.declare_parameter("publish_rate", 100.0)
        self._publish_rate = self.get_parameter("publish_rate").value
        self.declare_parameter("sync_status_timeout", 0.5)
        self._sync_status_timeout_ns = int(self.get_parameter("sync_status_timeout").value * 1e9)
        self.declare_parameter("follower_state_timeout", 0.5)
        self._follower_state_timeout_ns = int(self.get_parameter("follower_state_timeout").value * 1e9)
        self.declare_parameter("sync_position_tolerance", 0.03)
        self._sync_position_tolerance = self.get_parameter("sync_position_tolerance").value
        self.declare_parameter("sync_stable_duration", 0.2)
        self._sync_stable_duration_ns = int(self.get_parameter("sync_stable_duration").value * 1e9)
        self.declare_parameter("sync_profile_velocity", 2.0)
        self._sync_profile_velocity = self.get_parameter("sync_profile_velocity").value
        self.declare_parameter("sync_profile_acceleration", 3.0)
        self._sync_profile_acceleration = self.get_parameter("sync_profile_acceleration").value
        self.declare_parameter("ready_lock_rate", 20.0)
        self._ready_lock_period_ns = int((1.0 / self.get_parameter("ready_lock_rate").value) * 1e9)
        self.declare_parameter("ready_resync_threshold", max(self._sync_position_tolerance * 2.0, 0.08))
        self._ready_resync_threshold = self.get_parameter("ready_resync_threshold").value

        qos = QoSProfile(depth=10)
        self._arm_command_pub = self.create_publisher(
            JointState,
            "/ace_leader/arm/command",
            qos,
        )
        self._sync_mode_pub = self.create_publisher(
            String,
            "/ace_leader/arm/sync_mode",
            qos,
        )
        self._gripper_command_pub = self.create_publisher(
            JointState,
            "/ace_leader/gripper/command",
            qos,
        )
        self._state_sub = self.create_subscription(
            JointState,
            "/ace_follower/arm/state",
            self._state_callback,
            qos,
        )
        self._sync_status_sub = self.create_subscription(
            String,
            "/ace_follower/arm/sync_status",
            self._sync_status_callback,
            qos,
        )
        self._land_detected_sub = self.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._landed_callback,
            qos_profile_sensor_data,
        )

        control_period = 1.0 / self._control_rate
        self._control_timer = self.create_timer(control_period, self._control_loop)
        publish_period = 1.0 / self._publish_rate
        self._arm_command_publish_timer = self.create_timer(publish_period, self._publish_arm_command_loop)

        self._sync_mode = LeaderSyncMode.IDLE
        self._follower_sync_status = FollowerSyncStatus.IDLE
        self._last_follower_sync_status_ns: Optional[int] = None
        self._latest_follower_state: Optional[
            Tuple[Sequence[float], Sequence[float], Sequence[float]]
        ] = None
        self._last_follower_state_ns: Optional[int] = None
        self._sync_target_position: Optional[list[float]] = None
        self._sync_stable_since_ns: Optional[int] = None
        self._last_ready_lock_ns: Optional[int] = None
        self._is_landed = False
        self._latest_arm_command: Optional[
            Tuple[Sequence[float], Sequence[float], Sequence[float]]
        ] = None
        self._latest_gripper_command: Optional[
            Tuple[Sequence[float], Sequence[float], Sequence[float]]
        ] = None

        self.get_logger().info("Leader arm controller node started.")

    def _state_callback(self, msg: JointState):
        self._latest_follower_state = (
            list(msg.position),
            list(msg.velocity),
            list(msg.effort),
        )
        self._last_follower_state_ns = self.get_clock().now().nanoseconds

    def _sync_status_callback(self, msg: String):
        try:
            sync_status = FollowerSyncStatus(msg.data)
        except ValueError:
            self.get_logger().warn(f"Ignoring invalid sync status: {msg.data}")
            return
        self._follower_sync_status = sync_status
        self._last_follower_sync_status_ns = self.get_clock().now().nanoseconds

    def _landed_callback(self, msg: VehicleLandDetected):
        # self._is_landed = msg.landed
        self._is_landed = False

    def _has_recent_follower_state(self, now_ns: int) -> bool:
        latest_follower_state = getattr(self, "_latest_follower_state", None)
        last_follower_state_ns = getattr(self, "_last_follower_state_ns", None)
        follower_state_timeout_ns = getattr(self, "_follower_state_timeout_ns", 0)
        return (
            latest_follower_state is not None
            and last_follower_state_ns is not None
            and now_ns - last_follower_state_ns <= follower_state_timeout_ns
        )

    def _request_sync(self, message: str):
        if self._sync_mode != LeaderSyncMode.SYNC_REQUEST:
            self.get_logger().info(message)
        self._sync_mode = LeaderSyncMode.SYNC_REQUEST
        self._sync_target_position = None
        self._sync_stable_since_ns = None
        self._last_ready_lock_ns = None

    @staticmethod
    def _shortest_angle_errors(current: Sequence[float], target: Sequence[float]) -> np.ndarray:
        current_array = np.asarray(current, dtype=float)
        target_array = np.asarray(target, dtype=float)
        return (target_array - current_array + np.pi) % (2 * np.pi) - np.pi

    def _lock_to_sync_target(self, now_ns: Optional[int] = None):
        if self._sync_target_position is None:
            return
        if now_ns is not None and self._last_ready_lock_ns is not None:
            if now_ns - self._last_ready_lock_ns < self._ready_lock_period_ns:
                return
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        arm_ids = np.asarray(getattr(single_arm, "ids", getattr(self, "ids", []))).astype(int).tolist()
        if arm_ids:
            self.set_position(
                self._sync_target_position,
                ids=arm_ids,
                velocities=self._sync_profile_velocity,
                accelerations=self._sync_profile_acceleration,
            )
            self._last_ready_lock_ns = now_ns

    def _update_sync_request(self, joint_pos: Sequence[float], now_ns: int):
        if not self._has_recent_follower_state(now_ns):
            return

        latest_follower_state = self._latest_follower_state
        if latest_follower_state is None:
            return

        if self._sync_target_position is None:
            follower_pos, _, _ = latest_follower_state
            self._sync_target_position = np.asarray(follower_pos, dtype=float).tolist()
            single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
            arm_ids = np.asarray(getattr(single_arm, "ids", getattr(self, "ids", []))).astype(int).tolist()
            if arm_ids:
                self.set_position(
                    self._sync_target_position,
                    ids=arm_ids,
                    velocities=self._sync_profile_velocity,
                    accelerations=self._sync_profile_acceleration,
                )
            return

        current = np.asarray(joint_pos, dtype=float)
        target = np.asarray(self._sync_target_position, dtype=float)
        if current.shape != target.shape:
            self._sync_stable_since_ns = None
            return
        errors = self._shortest_angle_errors(current, target)
        position_ok = bool(np.all(np.abs(errors) <= self._sync_position_tolerance))
        if position_ok:
            if self._sync_stable_since_ns is None:
                self._sync_stable_since_ns = now_ns
            elif now_ns - self._sync_stable_since_ns >= self._sync_stable_duration_ns:
                self._sync_mode = LeaderSyncMode.READY
                self._lock_to_sync_target(now_ns)
                self.get_logger().info("Leader arm synchronized. Waiting for gripper release.")
        else:
            self._sync_stable_since_ns = None

    def _control_loop(self):
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        if single_arm is not None and hasattr(single_arm, "act"):
            joint_pos, joint_vel, joint_effort = single_arm.act()
        else:
            joint_pos, joint_vel, joint_effort = self.act()
        gripper = getattr(getattr(self, "_equipments", None), "gripper", None)
        if gripper is not None:
            self._latest_gripper_command = gripper.act()
        else:
            self._latest_gripper_command = None
        self._latest_arm_command = (joint_pos, joint_vel, joint_effort)
        if self._sync_mode == LeaderSyncMode.STOP:
            return
        if self._is_landed:
            self._sync_mode = LeaderSyncMode.STOP
            self.get_logger().info("Landing detected. Teleoperation stopped.")
            return

        now_ns = self.get_clock().now().nanoseconds
        if (
            self._sync_mode == LeaderSyncMode.TRACKING
            and self._follower_sync_status in (FollowerSyncStatus.LOST, FollowerSyncStatus.FAULT)
        ):
            self._request_sync("Follower reported lost sync. Requesting synchronization.")
            return

        if (
            self._sync_mode == LeaderSyncMode.TRACKING
            and getattr(self, "_last_follower_sync_status_ns", None) is not None
            and now_ns - self._last_follower_sync_status_ns > self._sync_status_timeout_ns
        ):
            self._request_sync("Follower sync status timed out. Requesting synchronization.")
            return

        if (
            self._sync_mode == LeaderSyncMode.TRACKING
            and getattr(self, "_last_follower_state_ns", None) is not None
            and not self._has_recent_follower_state(now_ns)
        ):
            self._request_sync("Follower state timed out. Requesting synchronization.")
            return

        if self._sync_mode == LeaderSyncMode.IDLE:
            if self._has_recent_follower_state(now_ns):
                self._request_sync("Follower arm state received. Requesting synchronization.")
                self._update_sync_request(joint_pos, now_ns)
            return

        if self._sync_mode == LeaderSyncMode.SYNC_REQUEST:
            self._update_sync_request(joint_pos, now_ns)
            return

        if self._sync_mode == LeaderSyncMode.READY:
            ready_error_exceeds_threshold = False
            if self._sync_target_position is not None:
                current = np.asarray(joint_pos, dtype=float)
                target = np.asarray(self._sync_target_position, dtype=float)
                if current.shape != target.shape:
                    ready_error_exceeds_threshold = True
                else:
                    errors = self._shortest_angle_errors(current, target)
                    ready_error_exceeds_threshold = bool(np.any(np.abs(errors) > self._ready_resync_threshold))
            if ready_error_exceeds_threshold:
                self._request_sync("Leader drifted from synchronized pose. Requesting synchronization.")
                return
            self._lock_to_sync_target(now_ns)
            gripper_value = None
            if gripper is not None:
                gripper_pos, _, _ = self._latest_gripper_command
                gripper_value = float(np.asarray(gripper_pos, dtype=float)[0])
            if gripper_value is None or gripper_value >= 1.0:
                arm_ids = np.asarray(getattr(single_arm, "ids", getattr(self, "ids", []))).astype(int).tolist()
                if arm_ids:
                    self.set_torque_enable(TorqueEnable.Disable, ids=arm_ids)
                self._sync_mode = LeaderSyncMode.TRACKING
                self.get_logger().info("Leader gripper released. Teleoperation tracking started.")

    def _publish_arm_command_loop(self):
        mode_msg = String()
        mode_msg.data = self._sync_mode.value
        self._sync_mode_pub.publish(mode_msg)
        if self._latest_arm_command is None or self._sync_mode != LeaderSyncMode.TRACKING:
            return

        joint_pos, joint_vel, joint_effort = self._latest_arm_command
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        arm_ids = np.asarray(getattr(single_arm, "ids", getattr(self, "ids", range(len(joint_pos))))).astype(int)
        msg.name = [f"joint_{i+1}" for i in arm_ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self._arm_command_pub.publish(msg)

        if self._latest_gripper_command is None:
            return
        gripper_pos, gripper_vel, gripper_effort = self._latest_gripper_command
        gripper_msg = JointState()
        gripper_msg.header.stamp = msg.header.stamp
        gripper_id = getattr(self, "gripper_id", None)
        gripper_msg.name = [] if gripper_id is None else [f"joint_{int(gripper_id) + 1}"]
        gripper_msg.position = np.asarray(gripper_pos, dtype=float).tolist()
        gripper_msg.velocity = np.asarray(gripper_vel, dtype=float).tolist()
        gripper_msg.effort = np.asarray(gripper_effort, dtype=float).tolist()
        self._gripper_command_pub.publish(gripper_msg)
