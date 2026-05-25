import json
from typing import Optional, Sequence, Tuple

import numpy as np
from geometry_msgs.msg import WrenchStamped
from px4_msgs.msg import ArmJointState
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from acetele.config.config_loader import ConfigLoader
from acetele.robot.ace_follower.ace_follower import AceFollowerRobot
from acetele.utils.teleop_sync import (
    FollowerSyncStatus,
    LeaderSyncMode,
)


class AceFollowerROS2Robot(Node, AceFollowerRobot):
    def __init__(self, config_loader: ConfigLoader):
        Node.__init__(self, "ace_follower_robot_node")
        AceFollowerRobot.__init__(self, config_loader)
        self.declare_parameter("control_rate", 100.0)
        self._control_rate = self.get_parameter("control_rate").value
        self.declare_parameter("publish_rate", 100.0)
        self._publish_rate = self.get_parameter("publish_rate").value
        self.declare_parameter("heartbeat_timeout", 1.0)
        self._heartbeat_timeout_ns = int(self.get_parameter("heartbeat_timeout").value * 1e9)
        qos = QoSProfile(depth=10)
        self._arm_state_pub = self.create_publisher(
            JointState,
            "/ace_follower/arm/state",
            qos,
        )
        self._external_joint_torque_pub = self.create_publisher(
            JointState,
            "/ace_follower/arm/external_joint_torque",
            qos,
        )
        self._external_wrench_pub = self.create_publisher(
            WrenchStamped,
            "/ace_follower/arm/external_wrench",
            qos,
        )
        self._px4_arm_state_pub = self.create_publisher(
            ArmJointState,
            "/fmu/in/arm_joint_state",
            qos,
        )
        self._sync_status_pub = self.create_publisher(
            String,
            "/ace_follower/arm/sync_status",
            qos,
        )
        self._gripper_state_pub = self.create_publisher(
            JointState,
            "/ace_follower/gripper/state",
            qos,
        )
        self._gripper_force_state_pub = self.create_publisher(
            String,
            "/ace_follower/gripper/force_state",
            qos,
        )
        self._arm_command_sub = self.create_subscription(
            JointState,
            "/ace_leader/arm/command",
            self._arm_command_callback,
            qos,
        )
        self._gripper_command_sub = self.create_subscription(
            JointState,
            "/ace_leader/gripper/command",
            self._gripper_command_callback,
            qos,
        )
        self._sync_mode_sub = self.create_subscription(
            String,
            "/ace_leader/arm/sync_mode",
            self._sync_mode_callback,
            qos,
        )

        control_period = 1.0 / self._control_rate
        self._control_timer = self.create_timer(control_period, self._control_loop)
        publish_period = 1.0 / self._publish_rate
        self._state_publish_timer = self.create_timer(publish_period, self._publish_state_loop)

        current_pos, _, _ = self.act()
        self.set_position(current_pos)
        self._sync_mode = LeaderSyncMode.IDLE
        self._sync_status = FollowerSyncStatus.IDLE
        self._last_command_ns = None
        self._heartbeat_lost = False
        self._latest_arm_command: Optional[list[float]] = None
        self._latest_arm_state: Optional[
            Tuple[Sequence[float], Sequence[float], Sequence[float]]
        ] = None
        self._last_external_estimate_ns: Optional[int] = None
        self._external_joint_torque: Optional[np.ndarray] = None
        self._external_wrench: Optional[np.ndarray] = None
        self._external_wrench_frame_id = getattr(
            getattr(getattr(self, "_equipments", None), "single_arm", None),
            "external_wrench_frame_name",
            "link_5",
        )
        self._latest_gripper_state: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None
        self._warned_invalid_px4_arm_state_length = False

        self.get_logger().info("Follower arm controller node started.")

    def _arm_command_callback(self, msg: JointState):
        if self._sync_mode != LeaderSyncMode.TRACKING:
            return
        self._last_command_ns = self.get_clock().now().nanoseconds
        self._heartbeat_lost = False
        self._latest_arm_command = list(msg.position)

    def _gripper_command_callback(self, msg: JointState):
        if self._sync_mode != LeaderSyncMode.TRACKING:
            return
        gripper = getattr(getattr(self, "_equipments", None), "gripper", None)
        if gripper is None or not msg.position:
            return
        command_position = float(msg.position[0])
        if not gripper.set_fragile_position(command_position):
            gripper.set_position(command_position)

    def _sync_mode_callback(self, msg: String):
        try:
            sync_mode = LeaderSyncMode(msg.data)
        except ValueError:
            self.get_logger().warn(f"Ignoring invalid sync mode: {msg.data}")
            return
        self._sync_mode = sync_mode
        if sync_mode == LeaderSyncMode.IDLE:
            self._sync_status = FollowerSyncStatus.IDLE
        elif sync_mode == LeaderSyncMode.SYNC_REQUEST:
            self.get_logger().info("Holding follower arm pose for leader synchronization.")
            self._sync_status = FollowerSyncStatus.READY
        elif sync_mode == LeaderSyncMode.READY:
            self._sync_status = FollowerSyncStatus.READY
        elif sync_mode == LeaderSyncMode.STOP:
            self._sync_status = FollowerSyncStatus.LOST

    def _control_loop(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._sync_status == FollowerSyncStatus.TRACKING and self._last_command_ns is not None:
            if now_ns - self._last_command_ns > self._heartbeat_timeout_ns:
                if not self._heartbeat_lost:
                    self.get_logger().info("Heartbeat lost. Entering lost sync state.")
                self._sync_status = FollowerSyncStatus.LOST
                self._heartbeat_lost = True
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        if single_arm is not None and hasattr(single_arm, "get_linker_state"):
            arm_state = single_arm.get_linker_state()
            joint_pos = arm_state.public_positions
            joint_vel = arm_state.velocities
            joint_effort = arm_state.motor_torque_magnitude
            self._latest_arm_state = (joint_pos, joint_vel, joint_effort)
            self._external_wrench_frame_id = single_arm.external_wrench_frame_name
            if not single_arm.external_torque_estimation_enabled:
                self._external_joint_torque = None
                self._external_wrench = None
            else:
                dt = (
                    0.0
                    if self._last_external_estimate_ns is None
                    else (now_ns - self._last_external_estimate_ns) * 1e-9
                )
                self._last_external_estimate_ns = now_ns
                arm_external_joint_torque = np.asarray(
                    single_arm.estimate_joint_external_torque(
                        arm_state.raw_positions,
                        arm_state.velocities,
                        arm_state.motor_torque_signed,
                        dt,
                    ),
                    dtype=float,
                )
                self._external_joint_torque = arm_external_joint_torque
                self._external_wrench = np.asarray(
                    single_arm.external_wrench_from_joint_torque(arm_state.raw_positions, arm_external_joint_torque),
                    dtype=float,
                )
        else:
            joint_pos, joint_vel, joint_effort = self.act()
            self._latest_arm_state = (joint_pos, joint_vel, joint_effort)
        gripper = getattr(getattr(self, "_equipments", None), "gripper", None)
        if gripper is not None:
            gripper_state = gripper.get_state()
            self._latest_gripper_state = (
                np.array([gripper_state.public_position], dtype=float),
                np.array([gripper_state.velocity], dtype=float),
                np.array([gripper_state.motor_torque_magnitude], dtype=float),
            )
        else:
            self._latest_gripper_state = None
        if self._sync_mode == LeaderSyncMode.STOP:
            self._sync_status = FollowerSyncStatus.LOST
            return
        if self._sync_mode == LeaderSyncMode.IDLE:
            self._sync_status = FollowerSyncStatus.IDLE
            return
        if self._sync_mode in (LeaderSyncMode.SYNC_REQUEST, LeaderSyncMode.READY):
            self._sync_status = FollowerSyncStatus.READY
            return
        if self._sync_mode == LeaderSyncMode.TRACKING:
            if self._latest_arm_command is None:
                return
            if self._sync_status in (FollowerSyncStatus.READY, FollowerSyncStatus.TRACKING):
                self._equipments.single_arm.set_position(self._latest_arm_command)
                self._sync_status = FollowerSyncStatus.TRACKING

    def _publish_state_loop(self):
        if self._latest_arm_state is None:
            return

        joint_pos, joint_vel, joint_effort = self._latest_arm_state
        now = self.get_clock().now()
        msg = JointState()
        msg.header.stamp = now.to_msg()
        single_arm = getattr(getattr(self, "_equipments", None), "single_arm", None)
        arm_ids = np.asarray(getattr(single_arm, "ids", getattr(self, "ids", range(len(joint_pos))))).astype(int)
        msg.name = [f"joint_{i+1}" for i in arm_ids]
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = joint_effort.tolist()
        self._arm_state_pub.publish(msg)

        external_joint_torque = getattr(self, "_external_joint_torque", None)
        if external_joint_torque is not None:
            external_msg = JointState()
            external_msg.header.stamp = now.to_msg()
            external_msg.name = msg.name
            external_msg.effort = external_joint_torque.tolist()
            self._external_joint_torque_pub.publish(external_msg)

        external_wrench = getattr(self, "_external_wrench", None)
        if external_wrench is not None:
            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = now.to_msg()
            wrench_msg.header.frame_id = getattr(self, "_external_wrench_frame_id", "link_5")
            wrench_msg.wrench.force.x = float(external_wrench[0])
            wrench_msg.wrench.force.y = float(external_wrench[1])
            wrench_msg.wrench.force.z = float(external_wrench[2])
            wrench_msg.wrench.torque.x = float(external_wrench[3])
            wrench_msg.wrench.torque.y = float(external_wrench[4])
            wrench_msg.wrench.torque.z = float(external_wrench[5])
            self._external_wrench_pub.publish(wrench_msg)

        status_msg = String()
        status_msg.data = self._sync_status.value
        self._sync_status_pub.publish(status_msg)

        gripper_state = getattr(self, "_latest_gripper_state", None)
        if gripper_state is not None:
            gripper_pos, gripper_vel, gripper_effort = gripper_state
            gripper_msg = JointState()
            gripper_msg.header.stamp = now.to_msg()
            gripper_id = getattr(self, "gripper_id", None)
            gripper_msg.name = [] if gripper_id is None else [f"joint_{int(gripper_id) + 1}"]
            gripper_msg.position = gripper_pos.tolist()
            gripper_msg.velocity = gripper_vel.tolist()
            gripper_msg.effort = gripper_effort.tolist()
            self._gripper_state_pub.publish(gripper_msg)

        publisher = getattr(self, "_gripper_force_state_pub", None)
        state = self.get_gripper_force_control_state() if hasattr(self, "get_gripper_force_control_state") else None
        if publisher is not None and state is not None:
            gripper_msg = String()
            gripper_msg.data = json.dumps(
                {
                    "status": getattr(state, "status", None),
                    "command_position": getattr(state, "command_position", None),
                    "hold_position": getattr(state, "hold_position", None),
                    "measured_torque_nm": getattr(state, "measured_torque_nm", None),
                    "contact_torque_nm": getattr(state, "contact_torque_nm", None),
                    "hold_torque_nm": getattr(state, "hold_torque_nm", None),
                }
            )
            publisher.publish(gripper_msg)

        px4_positions = list(msg.position)
        if gripper_state is not None:
            px4_positions = px4_positions + list(np.asarray(gripper_state[0], dtype=float))

        if len(px4_positions) != 5:
            if not self._warned_invalid_px4_arm_state_length:
                self.get_logger().warn(
                    "Skipping PX4 arm joint state publish: ArmJointState expects 5 joints."
                )
                self._warned_invalid_px4_arm_state_length = True
            return

        px4_msg = ArmJointState()
        px4_msg.timestamp = now.nanoseconds // 1000
        px4_msg.arm_position = px4_positions
        self._px4_arm_state_pub.publish(px4_msg)
