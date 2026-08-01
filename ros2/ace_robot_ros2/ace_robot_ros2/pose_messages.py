"""Lossless conversion between core Cartesian poses and ROS ``PoseStamped``."""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped

from acetele.core import EndEffectorPose


def pose_message(pose: EndEffectorPose, stamp) -> PoseStamped:
    """Build a ROS pose message without leaking ROS types into the core contract."""

    if not isinstance(pose, EndEffectorPose):
        raise ValueError("pose_message requires an EndEffectorPose")
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = pose.frame_id
    message.pose.position.x, message.pose.position.y, message.pose.position.z = (
        float(value) for value in pose.position_m
    )
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = (float(value) for value in pose.quaternion_xyzw)
    return message


def pose_from_message(message: PoseStamped, *, timestamp_ns: int) -> EndEffectorPose:
    """Validate one received ROS pose using its local monotonic receive timestamp."""

    if not isinstance(message, PoseStamped):
        raise ValueError("pose_from_message requires a PoseStamped")
    return EndEffectorPose(
        timestamp_ns,
        message.header.frame_id,
        (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ),
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ),
    )


__all__ = ["pose_from_message", "pose_message"]
